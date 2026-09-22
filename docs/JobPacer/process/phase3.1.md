# JobPacer Stage 3.1 实际代码实现方案

本文把 [设计讨论](../plan/discussion.md) 和 [Phase 3.1 计划](../plan/phase3.1.md)
落实为代码结构、状态机、协议、线程模型、策略算法和验收步骤。实施范围仅覆盖
Stage 3.1：线性 job、一个全局竞争资源、`max_inflight=1`、首先支持
`all_reduce`。DAG、多资源、多 host、恢复和批量 grant 不在本次实现中。

## 1. 对现有代码的处理

### 1.1 新旧边界

新 runtime 放入 `src/runtime_comm_scheduler/runtime/`，不依赖旧的 `Plan`、
`AdmissionScheduler`、`ScheduledWork`、`TaskKey` 和旧 executor。旧静态实现暂时
保留，用于历史测试和结果对照；新代码使用独立命名，避免语义混用。

此前的 `examples/jobpacer/dynamic/` 是早期原型，存在以下与新设计冲突的地方；该原型
已在新 runtime 完成后删除：

- `TaskCatalog` 要求启动前给出完整任务全集，不能在线注册；
- 继续使用旧七字段 `TaskKey`，job、group 和 endpoint 身份未分离；
- `Agent.offer()` 阻塞到 grant，不满足 `submit()` 立即返回 handle；
- coordinator 状态被多个 socket 线程和 timer 线程直接调用，不是单事件循环所有权；
- grant 没有 endpoint 级 `delivery_seq`，结束协议也只有不完整的 `done`；
- 线性前驱被建模为完整 catalog DAG，提前引入了 Stage 3.2 的结构。

因此不在该原型上继续增加分支。M0–M2 完成后，删除
`examples/jobpacer/dynamic/{catalog,coordinator,policies,protocol,transport}.py`，
示例目录只保留 workload 转换、进程启动和结果汇总。现有原型测试同步替换为新核心
测试，而不是继续维持兼容层。

### 1.2 建议文件布局

```text
src/runtime_comm_scheduler/runtime/
├── __init__.py       # 只导出稳定应用接口
├── model.py          # GroupSpec、TaskSpec、TaskHint、LocalBinding
├── protocol.py       # 控制消息、编解码、版本和协议校验
├── policy.py         # Snapshot、Action、Policy Protocol 与四种纯策略
├── coordinator.py    # 单线程状态机、合法候选、容量、结束和错误
├── transport.py      # TCP NDJSON，server/client 收发，不含调度逻辑
├── handle.py         # RuntimeHandle 的等待、绑定、完成和失败
├── executor.py       # 单 launch worker、backend launch 和完成探测
├── runtime.py        # RankRuntime 公共 API 与本地各线程编排
└── telemetry.py      # 本地/中央事件记录与 JSON 导出

examples/jobpacer/
├── workloads.py            # 保留 workload 定义
├── runtime_adapter.py       # workload → GroupSpec/TaskSpec/TaskHint/LocalBinding
├── runtime_worker.py        # 每 rank 的 Stage 3.1 replay
└── run_runtime_replay.py    # 拉起 ranks、收集结果和公共校验

tests/
├── unit/runtime/            # 模型、协议、coordinator、policy、handle
├── integration/test_runtime_gloo.py
├── integration/test_runtime_failures.py
└── gpu/test_runtime_nccl.py
```

不要求机械地为每个概念建抽象基类。首版 TCP transport 和 Gloo/NCCL executor 可以是
具体实现；只有策略需要统一的小接口，因为同一路径必须运行四种策略。

## 2. 数据模型

模型全部使用不可变 dataclass。控制面对象必须能无损转换为 JSON；本地 tensor、
ProcessGroup、CUDA event 和 callable 只能存在于 `LocalBinding`。

```python
@dataclass(frozen=True)
class GroupSpec:
    epoch: int
    group_id: str
    ranks: tuple[int, ...]

@dataclass(frozen=True)
class CollectiveSpec:
    op: Literal["all_reduce"]
    numel: int
    num_bytes: int
    dtype: str
    shape: tuple[int, ...]
    reduction: str = "sum"

@dataclass(frozen=True)
class TaskSpec:
    epoch: int
    job_id: str
    task_id: str                 # epoch 内全局唯一
    group_id: str
    group_seq: int               # 该 group 从 0 开始的规范通信序号
    collective: CollectiveSpec

@dataclass(frozen=True)
class TaskHint:
    ready_after_s: float | None
    estimated_comm_s: float
    remaining_tail_s: float      # 当前通信完成后的剩余 job 时间

@dataclass
class LocalBinding:
    tensor: Any
    process_group: Any
    launch: Callable[[], Any]
    producer_event: Any = None
    device: Any = None
    keepalive: tuple[Any, ...] = ()
```

必须在构造或注册边界校验：

- ID 非空，`epoch/group_seq/numel/num_bytes` 合法；
- group 成员非空、唯一且按全局 rank 规范化；
- `task_id` 在 epoch 内唯一；
- `group_seq` 对同一 group 不能映射到两个 task；
- `num_bytes` 与 shape/dtype/numel 一致；
- hint 数值非负，但 hint 不进入任务身份和成员匹配。

`task_id` 建议由 adapter 生成，例如 `job-0/comm-3`。它是协议主键，不从本地对象
地址、线程到达顺序或随机数生成。`group_seq` 由 workload 的共同语义顺序产生。

### 2.1 Coordinator 保存的最小状态

```python
@dataclass
class MemberProgress:
    declared: bool = False
    offered: bool = False
    submitted: bool = False
    completed: bool = False

@dataclass
class TaskRecord:
    spec: TaskSpec
    hints: dict[int, TaskHint]
    members: dict[int, MemberProgress]
    registered_seq: int
    eligible_seq: int | None = None
    grant_seq: int | None = None

@dataclass
class EndpointState:
    next_event_seq: int = 1
    next_delivery_seq: int = 1
    input_closed: bool = False
```

Coordinator 另外维护：已注册 group、`(group_id, group_seq) -> task_id`、每个 group
下一可 grant 的序号、当前唯一 inflight task、全局 `decision_seq`、主动等待状态和
事件日志。只有 coordinator event-loop 线程可以修改这些对象。

## 3. 应用 API 与本地语义

```python
runtime = RankRuntime(
    rank=rank,
    epoch=epoch,
    transport=client,
    executor=executor,
    completion_probe=probe,
)

runtime.register_group(group_spec, local_process_group)
runtime.declare(spec, hint)                    # 可选
handle = runtime.submit(spec, binding, hint)   # 立即返回
handle.wait_host(timeout=20.0)
handle.wait_on(torch.cuda.current_stream())    # NCCL 消费依赖
runtime.finish_epoch(timeout=20.0)
runtime.close()
```

### 3.1 `declare()`

`declare()` 保存本地声明并发送 `DECLARE`，用于 lookahead 看见尚未 ready 的线性前沿。
它不创建可执行绑定，也不使任务 eligible。相同声明可以幂等重放；相同 task_id 的
冲突元数据必须失败。已声明任务在 `INPUT_CLOSED` 前必须 `submit()`，首版不支持撤销。

### 3.2 `submit()`

`submit()` 只做本地校验、保存 `LocalBinding`、创建 handle、发送 `OFFER`，随后立即
返回。它不得等待成员到齐或 GRANT。job 线程可以继续独立计算，并在真正消费结果的
位置等待 handle。

CPU replay 在 producer sleep 完成后调用 `submit()`。GPU adapter 在 producer stream
记录 CUDA event 后调用 `submit()`，把 event 放入 binding；OFFER 表示 producer
依赖已经可表达，不表示 kernel 已物理完成。

### 3.3 `RuntimeHandle`

handle 内部状态为 `PENDING -> GRANTED -> BOUND -> COMPLETED`，任意非终态可进入
`FAILED`。Condition 只保护本 handle，不承担 coordinator 或 worker 调度。

- `bind(work)`：launch worker 成功取得底层异步 Work 后调用；只允许一次；
- `wait_host(timeout)`：先等绑定，再使用经过验收的 host completion 路径等待；失败
  原样抛出；总 timeout 同时覆盖等 grant/绑定和 backend 等待；
- `wait_on(stream)`：先等绑定，然后在指定消费 stream 上建立对通信完成的依赖；不得
  把它当作物理完成回执；
- `mark_completed()`：仅 completion worker 调用，记录物理完成并唤醒等待者；
- `fail(exc)`：幂等保存首个错误并唤醒所有等待者。

Gloo 的 `wait_host()` 可以调用底层 `Work.wait()`。NCCL 路径需要在 M5 根据目标
PyTorch/NCCL 版本验证 `Work.is_completed()`、`Work.wait()` 和 CUDA stream/event 的
语义后实现，不能用 CPU 返回时间冒充设备物理完成。

## 4. 控制协议

### 4.1 传输格式

使用单条 TCP 连接连接每个 endpoint 与 rank 0 coordinator，消息采用一行一个 JSON
对象（NDJSON），UTF-8 编码。每条消息包含：

```json
{
  "protocol": 1,
  "kind": "OFFER",
  "epoch": 7,
  "endpoint": 1,
  "event_seq": 12,
  "payload": {}
}
```

连接建立先发送 `HELLO(protocol, epoch, endpoint)`。Coordinator 收齐本 epoch 所有
endpoint 后回复 `READY`。TCP 只负责连接内字节有序，应用层仍检查 `event_seq` 和
`delivery_seq`，用于发现重复处理、逻辑丢失和错误连接复用。

### 4.2 上下行消息

| 方向 | 消息 | 关键 payload |
| --- | --- | --- |
| 上行 | `REGISTER_GROUP` | `GroupSpec` |
| 上行 | `DECLARE` | `TaskSpec`, `TaskHint` |
| 上行 | `OFFER` | `TaskSpec`, `TaskHint` |
| 上行 | `SUBMITTED` | `task_id`, `decision_seq` |
| 上行 | `COMPLETED` | `task_id`, `decision_seq` |
| 上行 | `INPUT_CLOSED` | 本地声明/提交计数摘要 |
| 上行 | `FAILED` | task、stage、错误类型和文本 |
| 下行 | `READY` | epoch |
| 下行 | `GRANT` | task_id、decision_seq、delivery_seq |
| 下行 | `FINISHED` | 最终 decision_seq 和任务数 |
| 下行 | `FAILED` | 中央错误摘要、最后 decision_seq |

控制消息不携带 tensor、ProcessGroup、event、backend Work 或 closure。

### 4.3 序号与幂等规则

- endpoint 每发送一个上行事件，`event_seq += 1`；coordinator 只接受严格连续的值，
  不是简单的“大于上次”；
- coordinator 每产生一个不可撤销 grant，`decision_seq += 1`；
- 每个 endpoint 只对发给自己的消息维护连续 `delivery_seq`，不参与某 task 的 endpoint
  不会因全局 decision_seq 跳号而报错；
- 完全相同 `(endpoint, event_seq)` 的网络重放首版也视为协议错误，因为 TCP 模式不做
  重连恢复；task 级 `DECLARE` 可在不同合法 event_seq 下幂等；
- 本地 runtime 对已经处理过的同一 `decision_seq/task_id` grant 不重复 launch；相同
  decision_seq 指向不同 task 或反之立即失败。

## 5. 线程与队列模型

### 5.1 Coordinator 端

Coordinator 只有一个拥有状态的 event-loop：

```text
socket reader(rank 0) ─┐
socket reader(rank 1) ─┼─> inbound Queue ─> coordinator event-loop
deadline timer --------┘                         │
                                                └─> per-endpoint outbound Queue
                                                          │
                                                    socket writer
```

reader 只负责解析和基本 envelope 校验，然后投递 `InboundEvent`；writer 只负责发送。
它们不能调用 coordinator 状态方法。event-loop 使用 `queue.Queue.get(timeout=...)`，
timeout 取当前 lookahead deadline 与 epoch deadline 的最近值，不创建多个
`threading.Timer`。这样 deadline 处理和新事件处理天然串行，也不会重复刷新等待预算。

rank 0 本地 runtime 也通过同样的 client/connection 路径接入，不走直接函数调用捷径，
保证两 rank 的协议和时间记录一致。

### 5.2 Rank Runtime 端

每个 rank 至少包含：

1. 一个 control reader：校验 delivery_seq，将 GRANT/FINISHED/FAILED 投递本地队列；
2. 一个 control writer 或带锁发送队列：串行发送所有上行事件；
3. 一个 launch worker：唯一允许调用受管理 group 的 collective；
4. 一个 completion worker：非阻塞探测所有已绑定 Work，完成后更新 handle 并发送
   COMPLETED；
5. 若干应用 job 线程：仅 declare/submit/wait，不直接调用 managed collective。

虽然 Stage 3.1 全局最多一个 inflight，runtime 仍用 `dict[task_id, LocalTask]` 表示本地
任务，避免把协议正确性绑死在单元素变量上。launch worker 严格按本地 GRANT 队列处理：

```text
收到 GRANT
  -> 校验 task 已 OFFER、未执行、group/spec 一致
  -> 等待/桥接 producer dependency
  -> executor.launch(binding)
  -> 校验异步 Work
  -> handle.bind(work)
  -> 发送 SUBMITTED
  -> 交给 completion worker
```

任何 launch 或 completion 错误先原子地使 runtime 进入 FAILED，停止接收新 grant，失败
所有未终态 handle，再发送 `FAILED`。不得由异常 job 线程继续独立发射后续 collective。

## 6. Coordinator 状态机和候选构造

### 6.1 事件处理

`CoordinatorState.apply(event, now)` 是唯一状态修改入口，按以下顺序执行：

1. 校验协议版本、epoch、endpoint、连续 event_seq；
2. 校验 GroupSpec 已由所有 endpoint 一致注册；
3. DECLARE/OFFER 首见时注册 TaskSpec；后续成员必须给出完全相同的 TaskSpec；
4. 更新该成员进度，拒绝非成员、倒退、跳转和重复终态；
5. SUBMITTED/COMPLETED 校验 task 已 grant 且 decision_seq 匹配；
6. 所有成员 COMPLETED 后释放唯一 inflight；
7. 更新 eligible、anticipated frontier 和结束条件；
8. 若容量可用，调用 policy；提交合法决定后再发布消息。

消息状态允许路径为：

```text
无记录 --DECLARE--> DECLARED --OFFER--> OFFERED
无记录 ---------------------OFFER-----> OFFERED
OFFERED --GRANT--> GRANTED --SUBMITTED--> SUBMITTED --COMPLETED--> COMPLETED
```

DECLARE 后直接 INPUT_CLOSED 而没有 OFFER 属于缺失任务；OFFER 后连接关闭、launch 失败或
完成超时均进入 epoch 失败，不能静默删除。

### 6.2 eligible

任务只有同时满足以下条件才进入 eligible：

- TaskSpec 已由 GroupSpec 的所有成员 OFFER，且 collective 元数据一致；
- 未 grant、未完成，且全局容量为空；
- `group_seq == next_group_seq[group_id]`；不存在同 group 的序号缺口；
- 它是该线性 job 当前 frontier。Stage 3.1 的 frontier 在上一 task 全体 COMPLETED 后
  推进，而不是仅凭后项提前 DECLARE/OFFER 推进。

任务第一次成为 eligible 时分配单调 `eligible_seq`，之后重复计算不能改变它。

### 6.3 anticipated frontier

anticipated 只包含每个未结束 job 的当前 frontier，且该任务已经 DECLARE、尚未全部
OFFER。不能把同 job 更远的后继放进去。Coordinator 以首次看见 hint 的本地接收时间
为基准计算预计到达：

```text
predicted_ready_at = declare_received_at + ready_after_s
```

成员 hint 不一致时，取所有成员预测中的最大值作为 collective 可 eligible 的最早估计，
同时在日志中保留各成员值。该预测仅供 policy 使用，不改变合法性。

### 6.4 grant 提交点

Policy 返回 `Dispatch(task_id)` 后，coordinator 必须再次校验任务仍 eligible，然后按
以下原子顺序处理：

1. 增加 `decision_seq`；
2. 设置唯一 inflight、task `grant_seq`，推进对应 group 的 grant 序列；
3. 为每个成员分配其 `delivery_seq`；
4. 记录完整 DecisionRecord；
5. 把 GRANT 放入成员 outbound queue。

第 2 步完成后 grant 不可撤销。某个 writer 发送失败意味着整个 epoch 失败，而不是回滚
并改选其他任务。

## 7. Policy 实现

Policy 是纯函数/小型状态对象，只接收不可变快照，不接触 socket、锁、tensor 或真实
时钟：

```python
@dataclass(frozen=True)
class PolicySnapshot:
    now: float
    eligible: tuple[Candidate, ...]
    anticipated: tuple[Anticipated, ...]
    active_wait: ActiveWait | None

Action = Dispatch | Wait | Idle | Done

策略决策接口统一为 `decide(snapshot)`；静态策略额外保存自己的 order/cursor，
其他策略不共享这些状态。Coordinator 只通过统一策略接口工作，并自行维护 lookahead 的
等待轮次。策略还提供统一的任务完整性检查入口，静态策略用它报告缺失的静态 task。
```

Coordinator 使用可注入的 `Clock.monotonic()` 生成 `now`，单元测试使用 fake clock。

### 7.1 StaticOrder

构造参数是完整 `task_id` 顺序，只供此策略使用。维护下标 `cursor`：

- 当前 task eligible：Dispatch；
- 当前 task 已完成：推进 cursor；
- 当前 task 尚未 eligible：返回 `Idle(STATIC_HEAD_BLOCKED)`；
- 不能绕过静态队首，也不能要求启动时所有任务已注册。

静态序列中的 task 在 INPUT_CLOSED 后仍未注册或提交，应作为缺失输入失败。

### 7.2 DynamicFIFO

选择 `(eligible_seq, task_id)` 最小的候选。`eligible_seq` 在首次 eligible 时固化，因此
后续状态重算不会把老候选移到新候选之后；task_id 只用于确定性破同分。

### 7.3 DynamicLTF

选择 `(-remaining_tail_s, eligible_seq, task_id)` 最小的候选。remaining tail 的统一定义
为“当前 collective 完成以后，job 尚余的预计时间”，不包含已经完成的 producer 和
当前通信。各成员 hint 不一致时使用注册时确定的规范值；首轮实验不在线学习更新。

### 7.4 BoundedLookahead

先用 DynamicLTF 得到当前候选 `c`，再从 anticipated frontier 中选 tail 最大且预计在
`wait_budget_s` 内到达的目标 `t`。令：

```text
r = max(0, predicted_ready_at(t) - now)
C = estimated_comm_s(c)
T = estimated_comm_s(t)
Rc = remaining_tail_s(c)
Rt = remaining_tail_s(t)

dispatch_score = max(C + Rc, C + T + Rt)
wait_score     = max(r + T + Rt, r + T + C + Rc)
```

只有 `wait_score < dispatch_score` 且 `r <= wait_budget_s` 才返回 Wait。第一次 Wait 建立：

```text
ActiveWait(target=t, deadline=now + min(r, wait_budget_s), round_id=n)
```

后续 OFFER 等事件可以提前重新决策，但同一 round 的 deadline 不得向后移动。目标在期限
内 eligible 时重新比较并通常优先 Dispatch；deadline 到期而仍有 eligible 时，设置
`must_dispatch=True`，本轮必须按 LTF 发射一个现有候选，不能立即开始另一轮等待。若无
eligible，则记录 `NO_ELIGIBLE` 并等待新事件/epoch timeout，不算主动等待续期。

所有策略决定记录候选、anticipated、排序 key、输入估计、score、deadline 和 reason，
保证能用日志离线重放。

## 8. Executor 与完成探测

### 8.1 CPU/Gloo 首版

`GlooExecutor.launch(binding)` 直接调用 `binding.launch()`，并校验返回对象至少提供
`wait()` 和 `is_completed()`。completion worker 周期调用 `is_completed()`；首次验收
中另用 backend 测试确认它代表本项目需要的物理完成边界。

轮询间隔可配置，默认 1 ms。它只影响完成发现延迟；日志同时记录 launch submit 时间和
completion observed 时间，不把两者差值称为精确通信 kernel 时间。

### 8.2 CUDA/NCCL 补充

每个本地 group 使用独立 gate stream。launch worker 在 gate stream 上等待
`producer_event`，再调用 asynchronous collective。handle 的 `wait_on(consumer_stream)`
在消费 stream 上建立 backend 完成依赖。completion probe 独立确认物理完成后才能发送
COMPLETED、释放全局容量。

M5 必须用真实测试分别验证：

- producer stream 未完成时 collective 不读取半成品 tensor；
- consumer stream 不会在 collective 完成前读取结果；
- 应用不调用 `wait_host()` 时 completion 仍会上报；
- 应用延迟消费不会阻止容量释放；
- 不同 group 的 gate stream 不引入无关的跨 group 依赖。

## 9. 结束与失败

### 9.1 正常结束

每个 RankRuntime 在不再产生任务时发送一次 INPUT_CLOSED。Coordinator 仅在以下条件
全部成立时发布 FINISHED：

- 所有 endpoint 已 INPUT_CLOSED；
- 每个已 DECLARE 的成员都已 OFFER，没有成员或 group_seq 缺口；
- 所有已接受任务已由全部成员 COMPLETED；
- 没有 inflight、active wait 或未发送的 grant。

`finish_epoch()` 发送 INPUT_CLOSED 后等待 FINISHED/FAILED；它不能因为本地任务完成就
自行成功。FINISHED 到达后 runtime 停止 worker、关闭 transport，并检查没有 pending
handle。

### 9.2 失败

以下任一事件使 epoch fail-stop：协议/元数据错误、非成员事件、序号缺口、重复 launch、
连接 EOF、launch 异常、completion probe 异常、声明缺失或 epoch timeout。

Coordinator 记录首个根因，停止产生 grant，向仍连接的 endpoint 广播 FAILED。Runtime
收到 FAILED 后失败所有本地非终态 handle、停止 launch，并让 `submit()`、
`finish_epoch()` 和等待接口观察同一根因。关闭必须有 timeout；若 backend collective
已经卡死，测试 harness 到期终止 rank 进程并将实验标记为失败，不能伪造取消成功。

失败报告至少包含 epoch、stage、task_id、endpoint、最后 event/delivery/decision seq、
inflight、缺失成员和未完成声明。

## 10. Telemetry 和结果格式

本地事件使用本进程 `monotonic_ns`：declare、submit、offer sent、grant received、launch
start、backend submitted、completion observed、first host wait、consumer wait、failed。
中央事件使用 coordinator `monotonic_ns`：message received、all offers received、first
eligible、decision、all submitted、all completed、input closed、finished/failed。

不同进程的原始 monotonic 时间不直接相减。每个事件带 source、endpoint、task_id、
seq 和本地时间；需要跨 rank 的阶段耗时由 coordinator 同一时钟上的接收事件计算。

每次运行输出：

- 配置、软件版本、epoch、group 和 workload digest；
- 全局 decision sequence 及各 endpoint delivery projection；
- 每 job duration、全局 makespan；
- 每 task eligible wait、主动等待、grant→submitted、submitted→completion-observed；
- `NO_ELIGIBLE`、`STATIC_HEAD_BLOCKED`、`ACTIVE_LOOKAHEAD`、`CAPACITY_FULL` 时间分解；
- tensor 正确性、结束状态和失败摘要。

决策日志是纯 JSON，可由测试重新构造 PolicySnapshot 并验证同策略得到同决定。

## 11. JobPacer adapter 和 replay

`runtime_adapter.py` 只做映射：

- `job.job_id` 同时生成独立 `group_id`，但模型中仍保存为两个字段；
- `CollectiveComm.id -> task_id=f"{job_id}/comm-{id}"`；
- 线性序号生成 `group_seq=id`；
- workload ranks 转为 GroupSpec.ranks；
- op、shape、dtype、num_bytes 转为 CollectiveSpec；
- producer 估计、通信估计和通信后 tail 转为 TaskHint；
- 本地创建 tensor、ProcessGroup 和 launch closure，形成 LocalBinding。

Stage 3.1 CPU replay 时间线：

```text
可选 declare 当前 frontier
-> producer sleep（按固定 seed 样本）
-> submit 并立即得到 handle
-> consumer-independent sleep
-> handle.wait_host()
-> 校验 all-reduce 结果
-> 推进同 job 下一任务
```

随机扰动用 `(seed, epoch, job_id, task_id, rank)` 经稳定 hash 派生各自 RNG，不能共享
全局 RNG 或按线程运行顺序取样。StaticOrder、DynamicFIFO、DynamicLTF 和 Lookahead
使用完全相同的 workload 样本、runtime、transport、executor 和完成探测，仅替换 policy。

进程启动器另外保留 bare 并发模式作为外部性能基线，但它不与 managed group 混用；
同一次 epoch 内所有受调度 collective 必须经过 RankRuntime。

## 12. 分里程碑实现顺序

### M0：模型、协议和 API 骨架

新增 `model.py`、`protocol.py`、`handle.py` 的状态骨架和 JSON round-trip。此阶段不连
socket、不调用 torch。

通过条件：

- GroupSpec/TaskSpec/CollectiveSpec 校验；
- TaskSpec 元数据匹配和冲突检测；
- 所有消息 round-trip 后相等，未知字段策略和协议版本明确；
- handle 的立即返回、绑定、完成、失败和总 timeout 单测通过。

### M1：纯 coordinator 与策略

实现 `CoordinatorState.apply()`、候选构造、四策略和 fake clock。测试直接喂事件，
不启动线程。

通过条件：覆盖成员到齐、group_seq 缺口、动态选择、StaticOrder 阻塞、FIFO 首次
eligible 顺序、LTF、lookahead 期限内/外、deadline 不刷新、容量释放和结束判定。

### M2：TCP、RankRuntime 与 Gloo executor

实现 transport 的 reader/event-loop/writer、RankRuntime、单 launch worker、completion
worker 和正常结束。增加两 rank subprocess Gloo 测试。

通过条件：真实 all-reduce 正确；`submit()` 在 grant 前返回；成员到达顺序相反仍执行
共同序列；job 晚消费不妨碍 completion 上报和下一个 grant。

### M3：故障和确定性机制

补充协议注入和 subprocess 故障场景。

通过条件：元数据冲突、非成员、event/delivery 序号错误、重复 grant、旧 epoch、声明
缺失、断连、launch/completion 失败全部有界退出；错误不会被报告为成功或静默跳过。

### M4：最小策略对照

新增无扰动和静态队首晚到 workload，用同一 execution path 运行四策略。输出结果 JSON
和 `docs/JobPacer/result/phase3.1.md`。

通过条件：StaticOrder 忠实等待；Dynamic 策略能服务另一个已 ready job；FIFO/LTF 在
构造快照上产生预期不同选择；lookahead 的等待/回退可由日志重放解释。动态策略不要求
每项指标必胜。

### M5：NCCL

实现/启用 CUDA producer、gate、consumer stream 和经过验收的 completion probe。GPU
不可用时测试显式 skip，结果文档标记未验收，不能宣称 NCCL 完成。

## 13. 测试矩阵

| 层次 | 重点用例 |
| --- | --- |
| model/protocol | JSON round-trip、版本、epoch、成员、metadata、event/delivery seq |
| coordinator | 在线注册、直接 OFFER、成员到齐、group_seq、一次 grant、全员完成释放容量 |
| policy | static 队首、FIFO eligible_seq、LTF tail、lookahead score/deadline/fallback |
| handle/runtime | submit 非阻塞、grant 前等待、一次 bind/launch、晚消费、错误广播 |
| Gloo integration | 两 rank 多 job、相反 arrival、子 group、结果、finish handshake |
| failure integration | missing offer、disconnect、metadata mismatch、launch/probe failure、timeout |
| replay | 相同 seed 输入、四策略同路径、日志重放、makespan/等待分解 |
| NCCL | producer/consumer stream、物理完成释放、无 wait 也上报、tensor 正确 |

最关键的跨 rank 断言不是所有 rank 拥有完全相同的全局日志，而是：

```text
每个 endpoint 的 actual_submit_sequence
    == 全局 grant sequence 在该 endpoint 所属任务上的前缀/最终投影
```

同一 group 的成员最终投影必须完全一致，且与 group_seq 的规范顺序一致。

## 14. 完成与非目标

Stage 3.1 完成时应具备：在线 submit、中央动态选择、共同有序 grant、真实 Gloo
collective、独立完成上报、四策略、正常结束、失败有界退出以及可解释的小规模对照。

本阶段明确不做：通用 DAG、多个 collective 同时在途、链路资源图、抢占、重连恢复、
跨 host coordinator、独立 Job Agent、在线估计学习和批量 grant。后续阶段可以修改 API，
不为尚未验证的扩展提前引入插件或兼容层。
