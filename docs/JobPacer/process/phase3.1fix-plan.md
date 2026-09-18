# JobPacer Phase 3.1 修复实施计划

日期：2026-09-17

依据：[原实施计划](phase3.1.md)和[原设计计划](../plan/phase3.1.md)。本计划只修复现有 Phase 3.1 runtime 的正确性、可验证性和必要的资源管理问题，保留中心 coordinator、独立 TCP 控制通道、rank-local runtime、单全局 inflight 和四种策略的现有分层。

## 1. 目标和完成边界

本轮完成后需要满足：

1. 对每个 rank，`SUBMITTED` 必定先于 `COMPLETED` 到达 coordinator；
2. Dynamic FIFO 按任务首次真正 eligible 的顺序调度，不受 DECLARE 顺序或字典顺序影响；
3. bounded lookahead 的 deadline 在空闲和持续有消息两种情况下都不会刷新或明显超时；
4. `wait_host()` 只等待 runtime 的 `COMPLETED/FAILED` 状态，timeout 和失败唤醒语义统一；
5. lookahead 的 ready 预测以当前 frontier 的 producer 开始时刻为基准，未知时间不按立即 ready
   处理；
6. input close、terminal state、GRANT 和本地 tensor/ProcessGroup 都经过明确校验；
7. trace 足以还原每个任务的等待、grant、launch、完成和每个 job 的 makespan；
8. 修复后的 CPU/Gloo 两 rank replay 和故障场景有确定性测试，原 Phase 3.1 结果重新生成。

本轮不实现 DAG、多 inflight、多资源、重连、跨 host coordinator 或 NCCL stream 语义。NCCL
仍标记为未验收；CLI 可以保留入口，但结果文档不得把它列为已完成能力。

## 2. 修复后的关键状态关系

### 2.1 rank-local 任务状态

`RuntimeHandle` 保留现有状态机：

```text
PENDING -> GRANTED -> BOUND -> COMPLETED
    \          \         \
     +----------+---------+-> FAILED
```

状态含义收紧为：

- `PENDING`：OFFER 已在本地登记，尚未收到合法 GRANT；
- `GRANTED`：完整 GRANT 已校验，任务已进入 launch queue；
- `BOUND`：backend 已返回合法异步 Work；
- `COMPLETED`：runtime completion probe 已确认物理完成；
- `FAILED`：本任务或整个 runtime 已失败。

`BOUND` 不再表示 completion worker 可以立刻探测。是否允许探测由 runtime 内部的 active 集合
控制，只有 `SUBMITTED` 成功发送后才把 task 放入 active 集合。

### 2.2 coordinator 成员状态

每个 task member 仍按以下顺序推进：

```text
DECLARE（可选） -> OFFER -> GRANT -> SUBMITTED -> COMPLETED
```

coordinator 继续用 `_Member` 保存每个 rank 的进展，但所有进度事件都记录 coordinator
单调时间。非法跳转、重复事件、错误 decision sequence 和 terminal 后的新事件全部失败。

### 2.3 runtime 生命周期

在 `RankRuntime` 中明确以下生命周期：

```text
CREATED -> RUNNING -> INPUT_CLOSED -> CLOSED
                \-> FAILED -> CLOSED
```

不需要建立通用状态机框架；一个小型 enum 或现有布尔字段加集中校验即可。行为要求如下：

| 状态 | 允许操作 |
| --- | --- |
| `CREATED` | `register_group()`、`start()` |
| `RUNNING` | `register_group()`、`declare()`、`submit()`、`finish_epoch()` |
| `INPUT_CLOSED` | 等待已有任务完成；拒绝新的 group、DECLARE 和 OFFER |
| `FAILED` | 所有新操作抛出首个 runtime 错误 |
| `CLOSED` | 所有新操作明确失败；`close()` 本身幂等 |

coordinator 对每个 endpoint 同样在 `INPUT_CLOSED` 后拒绝 `REGISTER_GROUP`、`DECLARE` 和
`OFFER`。全局 `finished` 后除连接关闭外不再接受业务事件。

## 3. P0：先修四个正确性问题

### 3.1 消除 `COMPLETED -> SUBMITTED` 竞态

修改文件：

- `src/runtime_comm_scheduler/runtime/runtime.py`
- `tests/unit/runtime/test_rank_runtime.py`（新增）

在 `RankRuntime` 中增加只保存可探测任务 ID 的 `_active_task_ids` 集合。launch worker 的顺序
固定为：

```text
检查 runtime 未失败
-> executor.launch(binding)
-> handle.bind(work)
-> transport.send(SUBMITTED)
-> 将 task_id 加入 _active_task_ids
```

completion worker 只遍历 `_active_task_ids`，不再扫描 `_tasks` 中所有历史任务。探测完成后先从
active 集合删除，再调用 `mark_completed()` 和发送 `COMPLETED`。如果发送 SUBMITTED 失败，
任务永远不会进入 active 集合，而是直接走统一 `_fail()`。

这一顺序形成明确的 happens-before：completion worker 能看到任务之前，SUBMITTED 的
`send()` 已经在同一个 `ControlClient` 上完成。不能只依赖 `handle.state == BOUND`，因为 bind
发生在 SUBMITTED 上报之前。

确定性测试使用 fake executor、立即完成的 fake Work 和可阻塞 fake transport：

1. 让 Work 在 `bind()` 时已经 `is_completed() == True`；
2. 人为阻塞 SUBMITTED send；
3. 同时运行 completion loop；
4. 断言解除阻塞前没有 COMPLETED；
5. 最终发送序列严格为 `OFFER, SUBMITTED, COMPLETED`。

### 3.2 修正 Dynamic FIFO 的 eligible 顺序

修改文件：

- `src/runtime_comm_scheduler/runtime/coordinator.py`
- `tests/unit/runtime/test_runtime_core.py`

把当前 `_eligible()` 的两个职责拆开：

- `_refresh_eligibility()`：按照事件处理顺序，为刚满足条件的任务一次性分配
  `eligible_seq`；
- `_eligible_candidates()`：只读取已经分配的 eligible 状态并构造 `Candidate`。

每次 `apply()` 完成事件状态更新后，无论当前是否有 inflight，都调用
`_refresh_eligibility()`。`tick()` 也可调用一次作为防御，但不得重复分配序号。容量已满只阻止
`GRANT`，不能阻止记录 eligibility。

任务首次 eligible 的条件保持为：

```text
未 grant
AND group_seq 等于该 group 当前 frontier
AND group 全部成员均已 OFFER
```

回归测试严格复现问题场景：X inflight；A 先 DECLARE，B 后 DECLARE；B 先全部 OFFER，A 后全部
OFFER；X 完成后断言 FIFO grant B。另测同一 group 的后续序号只能在前项完成、frontier 推进后
获得 eligible_seq。

### 3.3 修正 lookahead deadline

修改文件：

- `src/runtime_comm_scheduler/runtime/coordinator.py`
- `src/runtime_comm_scheduler/runtime/policy.py`
- `src/runtime_comm_scheduler/runtime/transport.py`
- `tests/unit/runtime/test_runtime_core.py`
- `tests/unit/runtime/test_transport_deadline.py`（新增）

Coordinator 增加集中式 deadline 更新方法，在 `apply()` 和 `tick()` 进入决策前都执行：

- epoch deadline 到期：进入 `epoch_timeout`；
- active lookahead deadline 到期：设置 `must_dispatch=True`；
- deadline 一旦到期，不允许策略创建新的等待轮次，直到发射当前 eligible 候选。

等待期间每个新事件都重新构造 eligible/anticipated snapshot 并调用策略，以便新候选到达时
重新选择；若策略仍返回 `Wait`，新 deadline 必须取：

```text
min(策略建议 deadline, 当前 active_wait.deadline)
```

因此可以更换等待目标，但不能延长本轮最初的预算。到期时如果暂时没有 eligible task，保持
`must_dispatch=True`；第一个合法候选到达后立即发射，不再开始新一轮等待。

`CoordinatorState` 暴露只读的 `next_deadline`，返回 active wait deadline 和 epoch deadline
中的最近值。`CoordinatorServer._event_loop()` 的 queue timeout 改为：

```text
min(固定最大检查间隔, max(0, next_deadline - monotonic_now))
```

每处理一个 inbound message 后仍执行 deadline 检查，因此持续消息不会饿死 timer。固定最大
间隔保留为连接和停止检查上限，不再决定 lookahead 精度。

测试使用 fake clock 覆盖：

- deadline 前无目标到达：保持等待；
- deadline 后由 `tick()` 触发 fallback；
- deadline 后先到一个无关事件：同一次 `apply()` 必须 fallback；
- 等待中出现更优候选：允许重算但 deadline 不变；
- 持续输入消息：epoch timeout 仍按时发生；
- 5 ms wait budget 不被 50 ms queue timeout 放大。

### 3.4 统一 `wait_host()` 完成语义

修改文件：

- `src/runtime_comm_scheduler/runtime/handle.py`
- `tests/unit/runtime/test_handle.py`（新增）

`wait_host()` 不再调用底层 `work.wait()`。它只在 condition 上等待：

```text
COMPLETED -> True
FAILED    -> 抛出保存的原始异常
deadline  -> False
```

这使完成定义与 completion probe 完全一致，也保证 `fail()` 的 `notify_all()` 能立刻唤醒正在
等待的应用线程。删除针对 backend `wait(timeout=timedelta(...))` 和 `TypeError` fallback 的
分支，也删除不再使用的 `timedelta` import。

测试覆盖：

- PENDING、GRANTED、BOUND 均继续等待；
- `mark_completed()` 唤醒并返回 True；
- BOUND 后调用 `fail()` 立即唤醒并抛出同一异常；
- timeout 返回 False，且不会调用 fake Work 的 `wait()`；
- 负 timeout 仍报错；重复 complete/fail 保持幂等语义。

## 4. P1：修正 lookahead 输入和协议边界

### 4.1 只声明当前 job frontier

修改文件：

- `examples/jobpacer/runtime_worker.py`
- `examples/jobpacer/runtime_adapter.py`
- 对应 replay 单元测试

删除 job 开始时一次性 DECLARE 全部通信的循环。每个 task 的时间线改为：

```text
生成当前 TaskSpec/TaskHint
-> runtime.declare(current)
-> 记录 producer_start
-> producer sleep
-> runtime.submit(current)
-> consumer-independent sleep
-> handle.wait_host()
-> 推进下一项
```

这样 `TaskHint.ready_after_s` 始终相对于当前 task 的 DECLARE 时刻，值仍为本项
`producer_compute_s`，不需要引入累计时间模型。线性 job 一次只暴露一个 frontier，避免声明
一个实际上依赖前序通信的远期任务。

`CoordinatorState._anticipated()` 对 `ready_after_s=None` 的 hint 不构造 `Anticipated`；如果
任一必要 member 的 ready 时间未知，则该任务不能参与基于时间的 lookahead。不得继续使用
`ready_after_s or 0.0`。多个 member 都有预测时，使用最晚的 `declared_at + ready_after_s`
作为 collective 的预计 ready 时间。

测试覆盖第二项不会在第一项完成前成为 anticipated，以及 `None` 不会被当成当前时刻。

### 4.2 GRANT 完整校验

修改文件：

- `src/runtime_comm_scheduler/runtime/runtime.py`
- `tests/unit/runtime/test_rank_runtime.py`

control loop 收到 GRANT 后先解析完整 `TaskSpec`，再校验：

- message epoch 已由 transport/protocol 校验；
- task_id 在本地存在且已 OFFER；
- GRANT 中的完整 TaskSpec 与本地保存值完全相等；
- 当前 rank 是已注册 group 的成员；
- decision_seq 是正整数，handle 仍为 PENDING；
- 同一 task 不得重复 grant。

全部通过后才能调用 `handle.grant()` 和进入 launch queue。测试注入错误 epoch、group_seq、
collective metadata、未知 task 和重复 GRANT，均应在 backend launch 前失败。

### 4.3 校验 LocalBinding

修改文件：

- `src/runtime_comm_scheduler/runtime/runtime.py`
- `src/runtime_comm_scheduler/runtime/model.py`（若需要一个小型校验 helper）
- `tests/unit/runtime/test_rank_runtime.py`

`register_group(group, process_group)` 不再忽略实际对象。runtime 同时保存 `GroupSpec` 和本地
ProcessGroup；真实 replay 注册时 process_group 必须非空。`submit()` 校验：

- `binding.process_group` 与该 group 注册的本地对象相同；
- `binding.launch` 可调用；
- tensor 的 shape、numel、element size/num_bytes 和 dtype 与 `CollectiveSpec` 一致；
- binding device 与 tensor device 不矛盾；
- producer dependency 若当前 backend 不支持则必须为空，不得静默忽略。

校验采用 duck typing，核心 runtime 不引入 torch import。dtype 只需规范化常见字符串，如
`torch.float32 -> float32`。首版不尝试从 opaque ProcessGroup 反查全局成员，成员关系仍以
`GroupSpec` 和对象身份一致性共同保证。

测试用轻量 fake tensor/process group 覆盖 shape、dtype、bytes、错误 group 和空 launch。

### 4.4 输入关闭和失败后停止发射

修改文件：

- `src/runtime_comm_scheduler/runtime/runtime.py`
- `src/runtime_comm_scheduler/runtime/coordinator.py`
- `tests/unit/runtime/test_rank_runtime.py`
- `tests/unit/runtime/test_runtime_core.py`

`finish_epoch()` 在发送 `INPUT_CLOSED` 前原子地把本地状态设为 input closed，之后
`declare()`、`submit()` 和 `register_group()` 都拒绝新输入。coordinator 收到某 endpoint 的
INPUT_CLOSED 后拒绝该 endpoint 的新 REGISTER_GROUP/DECLARE/OFFER；重复 INPUT_CLOSED 明确
报协议错误。

`_fail()` 除标记所有 handle 失败外，还设置 stop event，并向 launch queue 放终止标记。launch
worker 从队列取出 task 后、调用 backend 前再次检查 runtime failure/stop；失败后已经排队但尚未
launch 的任务不得触碰 backend。当前正在执行的 backend call 无法抢占，本阶段只保证不发射新
任务。

测试构造两个已 grant 的 fake task（不改变正式 max_inflight=1 限制，可直接对白盒 queue
测试），在第一个失败后断言第二个 launch closure 从未调用。

## 5. P2：资源管理和 telemetry

### 5.1 active 集合与 LocalBinding 释放

修改文件：`src/runtime_comm_scheduler/runtime/runtime.py`。

P0 引入的 `_active_task_ids` 同时解决 completion loop 每毫秒扫描全部历史 task 的问题。任务
物理完成并成功形成完成记录后，释放 `_LocalTask.binding`，从而释放 tensor、launch closure 和
`keepalive` 引用。保留 TaskSpec、TaskHint、handle 和必要时间戳用于 trace。

失败路径也清理 active 集合和未再需要的 binding。清理不能早于 completion probe，因为
backend Work 及 closure 可能依赖 keepalive。

单元测试用 weakref 或带析构标记的对象确认完成前仍被持有，完成后 binding 被置空；历史 task
数量增加时 completion probe 调用次数只与 active 数量相关。

### 5.2 接入 rank-local EventLog

修改文件：

- `src/runtime_comm_scheduler/runtime/telemetry.py`
- `src/runtime_comm_scheduler/runtime/runtime.py`
- `examples/jobpacer/runtime_worker.py`

`RankRuntime` 接收可选 `EventLog`，默认创建 `source="runtime"`、当前 endpoint 的日志。由于
control、launch、completion 和应用线程都会写日志，`EventLog.record()` 增加内部锁，
`as_dict()` 返回一致快照。

至少记录以下 rank-local 事件：

| 事件 | 记录时点 |
| --- | --- |
| `group_registered` | 本地 group 校验通过 |
| `declared` | DECLARE 发出后 |
| `offered` | OFFER 发出后 |
| `grant_received` | 完整 GRANT 校验通过 |
| `launch_start` | 调用 executor 前 |
| `submitted_sent` | SUBMITTED 成功发送后 |
| `completion_observed` | probe 首次确认物理完成 |
| `completed_sent` | COMPLETED 成功发送后 |
| `input_closed` | INPUT_CLOSED 发出后 |
| `finished_received` | 收到 FINISHED |
| `failed` | 首个 runtime 错误被保存 |

事件包含 `task_id`、`decision_seq`、本地 `time_us`；不记录 tensor 内容或不可序列化对象。

### 5.3 补齐 coordinator 时间和等待区间

修改文件：`src/runtime_comm_scheduler/runtime/coordinator.py`。

所有 records 使用传入的 coordinator `now`，包括 eligible、decision、submitted、completed、
finished 和 failed。`_progress()` 增加 `now` 参数。对于状态原因，只在原因变化时记录区间边界，
避免 event loop 每次 tick 重复写 idle：

- `NO_ELIGIBLE`
- `STATIC_HEAD_BLOCKED`
- `ACTIVE_LOOKAHEAD`
- `CAPACITY_FULL`

最小实现可保存当前 idle reason 和开始时间，在 reason 切换、dispatch、finish 或 fail 时关闭区间。
不建设新的 telemetry 框架。

Coordinator trace 必须包含每次 policy snapshot 的完整输入：eligible 的 task ID、estimated comm、
remaining tail、eligible_seq；anticipated 的预测 ready 时间和估值；active wait、must_dispatch、
最终 action 和 reason。这样测试可以从 JSON 重建 `PolicySnapshot`。

### 5.4 修正 replay 时间戳和汇总

修改文件：

- `examples/jobpacer/runtime_worker.py`
- `examples/jobpacer/run_runtime_replay.py`

worker 分别记录：

```text
producer_start_ts
ready_ts
submit_call_ts
submit_return_ts
consumer_start_ts
first_wait_ts
consumer_end_ts
job_start_ts
job_end_ts
```

当前错误命名的 `submit_ts` 删除或改成真正的 `submit_call_ts`，不能继续把 producer sleep 前的
时间标成提交时间。rank-local runtime events 一并写入每 rank 输出，rank 0 继续输出 coordinator
records。

driver 计算并验证：

- 每个 task 结果正确；
- 所有 rank 的相同 group 实际 launch 投影一致；
- 每个 group 的 launch 顺序与 `group_seq` 一致；
- grant sequence 在各 endpoint 上的投影等于该 endpoint 的 launch sequence；
- 每个 job duration 和全局 makespan；
- coordinator 时钟上的 eligible→grant、grant→all submitted、all submitted→all completed；
- rank 本地的 submit call→grant、grant→launch、consumer wait 阻塞时间。

不同进程的 `perf_counter` 原始值不直接相减。跨 rank 阶段只使用 coordinator 接收事件的同一
时钟。

## 6. 测试文件和场景

建议保留现有 `test_runtime_core.py` 的模型与纯 coordinator 测试，并按职责新增三个文件：

| 文件 | 覆盖范围 |
| --- | --- |
| `tests/unit/runtime/test_handle.py` | handle 状态、完成、失败唤醒和 timeout |
| `tests/unit/runtime/test_rank_runtime.py` | GRANT、launch/complete 顺序、本地 binding、失败停止 |
| `tests/unit/runtime/test_transport_deadline.py` | event-loop deadline 唤醒和持续消息 |

集成测试新增或扩展：

| 场景 | 核心断言 |
| --- | --- |
| 两 rank、多 job、立即完成 fake Work | 每个 rank 始终 SUBMITTED 在 COMPLETED 前 |
| FIFO arrival inversion | B 先 eligible 时 B 先 grant，与 DECLARE 顺序无关 |
| lookahead 5 ms budget | fallback 不产生第二轮等待，误差使用宽松调度容差而非固定 50 ms |
| delayed frontier | 第二项只在第一项完成后声明和预测 |
| late runtime failure | BOUND 状态的 `wait_host()` 被立即唤醒并抛错 |
| input closed 后提交 | backend 不执行，所有进程有界失败 |
| metadata/binding mismatch | collective 发射前失败 |
| 正常 Gloo replay | tensor、grant/launch/group 投影、finish handshake 全部正确 |
| missing task/disconnect | coordinator 广播失败，子进程按 timeout 回收 |

不要对线程调度的绝对微秒值做硬断言。并发测试用 Event、Barrier 和 fake clock 控制交错；真实
Gloo 测试只断言顺序、不变量、有限正时长和有界退出。

## 7. 分步实施和每步验收

### F0：固定复现用例

先把 `phase3.1fix.md` 中四个高优先级问题写成当前实现必定失败的回归测试，不改生产代码。

通过条件：测试分别稳定复现 completion 顺序、FIFO、deadline 刷新和 wait_host 失败唤醒问题；
测试不依赖 sleep 猜测线程交错。

### F1：协议顺序与 handle

实现 active 集合、SUBMITTED 边界和 condition-only `wait_host()`。

通过条件：对应回归测试转绿；已有正常/失败 handle 行为不回退；completion probe 只访问 active
任务。

### F2：eligibility 与 deadline

拆分 eligibility 更新，集中检查 deadline，调整 event-loop timeout 和 active wait 预算继承。

通过条件：FIFO arrival inversion 和全部 fake-clock lookahead 用例通过；策略的纯函数测试覆盖
deadline fallback。

### F3：生命周期和边界校验

实现当前 frontier DECLARE、unknown ready、runtime/coordinator input close、完整 GRANT 和
LocalBinding 校验，以及失败后停止 launch。

通过条件：所有错误在 backend launch 前被拒绝；正常 replay 的应用接口不变；故障场景有界
退出。

### F4：资源释放和 telemetry

释放完成 binding，接入 EventLog，修正 worker 时间戳和 driver 投影验证。

通过条件：长任务序列不会被 completion loop 全量扫描，完成任务不再持有 tensor；输出 JSON
包含计划中的事件、耗时和投影，且可序列化。

### F5：CPU/Gloo 重新验收

依次执行：

```bash
PYTHONPATH=src pytest -q tests/unit/runtime tests/unit
PYTHONPATH=src pytest -q tests/integration/test_runtime_*.py

PYTHONPATH=src python examples/jobpacer/run_runtime_replay.py \
  --policy static_fifo --workload delayed --backend gloo --timeout 20 \
  --output /tmp/jobpacer-runtime-static.json

PYTHONPATH=src python examples/jobpacer/run_runtime_replay.py \
  --policy fifo --workload balanced --backend gloo --timeout 20 \
  --output /tmp/jobpacer-runtime-fifo.json

PYTHONPATH=src python examples/jobpacer/run_runtime_replay.py \
  --policy ltf --workload tail --backend gloo --timeout 20 \
  --output /tmp/jobpacer-runtime-ltf.json

PYTHONPATH=src python examples/jobpacer/run_runtime_replay.py \
  --policy lookahead --workload delayed --backend gloo --timeout 20 \
  --output /tmp/jobpacer-runtime-lookahead.json
```

再执行 metadata mismatch、missing task、断连、launch failure 和 completion probe failure。

通过条件：所有正常场景 tensor 正确、协议顺序正确、投影一致并正常 FINISHED；所有故障场景
返回非零、有明确根因、在进程级 timeout 内退出且没有遗留子进程。

### F6：更新结果文档

修改 `docs/JobPacer/result/phase3.1.md`，把原结果明确标为修复前结果，新增：

- 修复项及对应回归测试；
- 新的测试总数和命令；
- 四策略修复后的实际 grant/launch 顺序；
- workload、seed、backend、软件版本和输出文件；
- makespan、job duration 和等待分解；
- 仍未验收的 NCCL/GPU 边界。

只有 F0–F6 全部完成后，才能把 Phase 3.1 CPU/Gloo 标记为完成。

## 8. 兼容性和最小改动原则

- 保留 `GroupSpec`、`TaskSpec`、`TaskHint`、`LocalBinding`、`RuntimeHandle` 和
  `RankRuntime` 的现有外部角色；只收紧错误语义和生命周期。
- 保留 coordinator 单事件循环和 `max_inflight=1`，不为未来多资源调度提前引入资源图。
- 保留现有 NDJSON/TCP 协议；如消息字段未变化，不提升 protocol version。
- 不复用旧 Phase 2 `AdmissionScheduler`、`ScheduledWork` 或 Plan。
- 不为测试暴露新的生产 API；线程交错通过 fake transport/executor 和现有状态观察完成。
- `wait_on()` 的 GPU 语义本轮不伪装修复。CPU/Gloo 验收不调用它；NCCL 阶段再结合 CUDA
  event/stream 设计和测试。

## 9. 最终验收清单

- [ ] 任意可控线程交错下，每个 endpoint 都是 SUBMITTED 先于 COMPLETED；
- [ ] FIFO 按首次 eligible 顺序，DECLARE 顺序不影响结果；
- [ ] lookahead deadline 不刷新，持续消息不会饿死 tick/epoch timeout；
- [ ] `wait_host()` 可被 completion、failure 和 timeout 正确唤醒；
- [ ] 当前 frontier 声明基准正确，未知 ready 不参与时间预测；
- [ ] input close/finish/fail 后没有新的 task 或 backend launch；
- [ ] GRANT 与本地 TaskSpec 完全一致，LocalBinding 与 collective 元数据一致；
- [ ] completion probe 只扫描 active task，完成后释放 LocalBinding；
- [ ] telemetry 可解释所有 grant，并能验证 endpoint/group launch 投影；
- [ ] CPU/Gloo 四策略正常 replay 和全部故障注入有界结束；
- [ ] `docs/JobPacer/result/phase3.1.md` 已用修复后结果更新；
- [ ] NCCL/GPU 仍明确标记为未验收。
