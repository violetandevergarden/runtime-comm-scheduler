# Runtime Communication Scheduler：runtime 与 DAG 实现说明

本文按当前源码说明 Phase 3 通信 runtime 和其上层 DAG runner，重点是状态所有权、数据转换、发射顺序和完成边界。实验命令、GPU workload JSON 与七臂流程见 [JobPacer README](../../examples/jobpacer/README.md)；本包不负责生成实验输入或组织性能批次。

## 1. 包结构与设计边界

```text
runtime_comm_scheduler/
├── runtime/       # 新中心化通信准入协议与 rank 本地执行
├── dag/           # 通用依赖图、顺序/估计计算、job 节点推进
├── adapters/      # 框架适配相关模块
├── intent.py      # 以下为旧执行体系
├── plan.py
├── scheduler.py
├── work.py
├── executor.py
├── telemetry.py
├── validate.py
└── __init__.py
```

根层的 Plan、AdmissionScheduler、ScheduledWork 与 `runtime/` 是两套执行体系。新 runtime 不调用旧 scheduler 来完成准入；旧实现仍可作为基线，由上层 adapter 接入共同 DAG runner。`runtime/executor.py` 与根层 `executor.py` 也不是同一个模块。

当前通信核心限定 all-reduce、中心化 coordinator、单个全局通信容量 `max_inflight=1`。这不是通用多资源调度器，不提供任意多在途准入、自动故障恢复、跨 epoch 状态复用或完整训练框架接入。上层 bare 多在途参考不经过这里的中央准入，其实现位于 examples。

层次关系为：

```text
上层 workload / 训练框架 / examples
  ├─ 提供 TaskSpec、TaskHint、LocalBinding
  └─ 或提供 DagGraph、compute_fn、make_binding
                         ↓
                    dag/DagRunner
                         ↓
                   runtime/RankRuntime
              控制消息 ↙          ↘ 本地执行
           CoordinatorState       executor → PyTorch collective
                 ↓                     ↓
               policy             完成探测 → handle
```

核心不反向导入 examples。CUDA program、tensor 初始化、GPU buffer hazard、schema-v2 JSON、application terminals 的实验定义在上层；`dag/` 只接收图对象和执行回调。

## 2. Runtime 架构设计

### 2.1 三类信息分离

| 类型 | 关键字段 | 所属位置与作用 |
| --- | --- | --- |
| `GroupSpec` | epoch、group_id、ranks | 可序列化的逻辑通信组；成员排序并去重校验 |
| `CollectiveSpec` | op、numel、num_bytes、dtype、shape、reduction | 跨成员必须匹配的 collective 参数，校验元素数与字节数 |
| `TaskSpec` | epoch、job_id、task_id、group_id、group_seq、collective | 任务共同身份与组内规范顺序；不包含 tensor |
| `TaskHint` | ready_after_s、estimated_comm_s、remaining_tail_s | 策略估计，不是任务的真实未来执行结果 |
| `LocalBinding` | tensor、process_group、launch、producer_event、device、keepalive、timing_hook | 只留在当前 rank，定义真实执行及资源生命周期 |
| `RuntimeHandle` | 状态、decision_seq、本地 Work/receipt、错误 | 应用侧异步等待和 consumer stream 依赖接口 |

`epoch` 是本次调度窗口标识，不必等同训练 epoch。`task_id` 在该窗口内标识同一个逻辑 collective；DAG 默认生成 `job_id/node_id`。`group_seq` 是共同输入确定的组内顺序，不能由各 rank 的线程到达次序独立生成。

DECLARE/OFFER 只传 TaskSpec 和 TaskHint 的字典表示。coordinator 不持有 tensor、ProcessGroup、CUDA event 或 Python closure。GRANT 下发 TaskSpec 与顺序号，rank 再查找已经保存的 LocalBinding。

### 2.2 状态所有权与线程

| 组件 | 持有的状态 | 执行方式 |
| --- | --- | --- |
| `CoordinatorState` | groups、tasks、各成员进度、eligible 次序、inflight、等待 deadline、epoch 终态 | 由唯一中央事件循环修改；自身不操作 socket |
| `CoordinatorServer` | 连接、入站队列、各 endpoint 出站队列 | 接收线程入队，事件循环调用 apply/tick，每 endpoint writer 顺序写出 |
| `ControlClient` | 上行 event_seq、下行 delivery_seq、接收队列 | 发送锁序列化消息，reader 读 socket，receive 校验身份与投递顺序 |
| `RankRuntime` | 本地 group/任务/binding、发射队列、active task 集合、错误与生命周期 | control、launch、completion 三个工作线程，公共状态由 condition 保护 |
| `RuntimeHandle` | 单任务状态、receipt、错误 | 独立 condition 唤醒应用等待者 |

应用线程调用 submit 不直接发起 collective。control 线程只处理决定并入队，唯一 launch 线程执行队列；completion 线程只观察已跨过 SUBMITTED 发送边界的任务。这样应用 job 的线程交错不会直接决定 collective 发射顺序。

异步 submit 的含义是“不等待 grant 或 collective 完成”，不是保证没有任何 host 开销：输入校验、锁和 OFFER 发送仍在调用路径中。

### 2.3 五种顺序号不能混用

| 名称 | 范围 | 用途 |
| --- | --- | --- |
| `group_seq` | 单个逻辑 group | collective 规范先后约束 |
| `event_seq` | 单 endpoint 上行消息 | coordinator 检查事件连续性和重复/乱序 |
| `eligible_seq` | coordinator 内任务首次合法就绪次序 | 动态 FIFO 排序，容量满时仍更新 |
| `decision_seq` / task 的 `grant_seq` | 全局准入决定 | 关联 GRANT、SUBMITTED、COMPLETED |
| `delivery_seq` | 单 endpoint 的 GRANT 投递序列 | 校验该 rank 收到的决定投影连续性 |

全局 decision_seq 在某个 rank 上可以有空缺，因为该 rank 未必属于所有 group；不能要求每个 rank 收到所有全局决定。正确要求是实际 launch 为共同 grant 序列的本地投影前缀。

顺序保证来自“单中央决定者 → endpoint FIFO writer → client 接收校验 → 本地 FIFO launch queue → 唯一 launcher”的整条链。只校验整数序号不足以替代有序发布和执行。

## 3. 状态与数据如何流动

### 3.1 一个通信任务的完整生命周期

1. **连接和注册。** client 发送 HELLO；server 收齐本窗口 endpoint 后发送 READY。rank 注册 GroupSpec，并保存本地 ProcessGroup。
2. **可选 DECLARE。** 上层已知某任务未来可能 ready，发送规范和估计。这里只提供预测，不提供发射权限，也可以不 DECLARE 而直接 OFFER。
3. **submit/OFFER。** rank 校验身份、成员、binding/tensor 参数，保存 `_LocalTask` 和 handle，再发送 OFFER。若此前已 DECLARE，复用该任务的 handle。
4. **汇总成员信息。** coordinator 用 task_id 合并各成员上报，拒绝规范不一致、重复 OFFER 和 `(group_id, group_seq)` 冲突。
5. **首次 eligible。** 所有成员均 OFFER 且任务处于当前 group 序号前沿，分配 eligible_seq。任务 eligible 不意味着容量可用。
6. **决定并 GRANT。** 无 inflight 时 policy 在合法候选中选任务；coordinator 再校验选择、提交 decision_seq、占用容量并向成员发送 GRANT。决定此时不可撤销。
7. **rank 发射。** control 线程校验本地 spec，handle 转 GRANTED，任务进入 launch queue。launcher 调用 executor，绑定 Work/receipt 后发送 SUBMITTED。
8. **物理完成。** SUBMITTED 发送成功后才将任务加入 active 集合。completion probe 确认本 rank 物理完成，标记 handle 完成并发送 COMPLETED。
9. **全局释放。** coordinator 收齐该任务全部成员 COMPLETED，推进 `_next_group_seq`、清空 inflight，才允许下一次 grant。

```text
输入规范 + 本地资源
    │ submit / OFFER
    ▼
本地 PENDING               中央：等待成员 OFFER / group 前沿
    │ GRANT                         │ 合法候选 + 空闲容量
    ▼                               ▼
本地 GRANTED               中央：grant_seq 确定，inflight 占用
    │ executor.launch
    ▼
本地 BOUND ── SUBMITTED ──→ 成员 submitted=True
    │ physical probe
    ▼
本地 COMPLETED ─ COMPLETED → 成员 completed=True
                                    │ 全成员完成
                                    ▼
                           释放容量，推进 group_seq
```

**本地 handle 完成与全局容量释放不是同一时刻。** 当前 completion loop 先 `mark_completed()` 唤醒应用，再发送 COMPLETED。因此本 rank 后续计算可能已经推进，而 coordinator 仍等待消息或其他成员完成。不能把应用 wait 的返回时间当作全局完成时间。

### 3.2 本地 runtime 与 handle 状态

`RuntimeState`：

```text
CREATED → start/READY → RUNNING → finish_epoch → INPUT_CLOSED
                            ↘ 失败 → FAILED
以上生命周期最终通过 close → CLOSED
```

没有单独的 RuntimeState.FINISHED。正常 FINISHED 到达后，`_finish_received` 为真，finish_epoch 返回；对象仍待 close。一个 RankRuntime 绑定一个 rank/epoch，不是可自动重启复用的训练 session。

`HandleState`：

```text
PENDING → GRANTED → BOUND → COMPLETED
    未结束状态 ──错误──→ FAILED
```

- PENDING：请求已保存，尚无 grant；提前 DECLARE 也可以创建该 handle。
- GRANTED：共同决定已收到，本地尚未绑定执行结果。
- BOUND：Work 或 CUDA receipt 已绑定，不表示物理完成。
- COMPLETED：本地 completion probe 已确认完成。
- FAILED：保存失败原因，等待者抛出错误；已完成 handle 不被后续 fail 覆盖。

### 3.3 中央状态不是一个单独 enum

`CoordinatorState` 用多个结构记录不同维度：

| 结构/字段 | 内容 |
| --- | --- |
| `_EndpointState` | 下一个 event_seq、delivery_seq、input_closed |
| `_Member` | declared/offered/submitted/completed、中央收到声明的时刻 |
| `_Task` | 共同 spec、各成员 hint/进度、registered_seq、eligible_seq、grant_seq |
| `groups`、`_group_registered` | 组定义和注册成员 |
| `_group_task_ids`、`_next_group_seq` | 组内序号身份唯一性及当前执行前沿 |
| `inflight` | 当前占用唯一容量的 task_id，空闲为 None |
| `active_wait`、`_must_dispatch` | Lookahead 固定等待窗口与超时后的回退要求 |
| `failed`、`finished`、`_finish_sent` | epoch 终态及终结消息是否已产生 |

`registered_seq` 只是首次看到任务的次序，不能代替 FIFO 的 eligible_seq。不同 rank 的估计通过候选构造取最大 estimated_comm 和最大 remaining_tail；中央不要求每个成员估计值完全相同，但共同 TaskSpec 必须一致。

### 3.4 DECLARE 与预测时间

`TaskHint.ready_after_s` 是相对中央收到声明的时间估计，不发送并直接比较 rank 的绝对时钟。anticipated 要求任务未 grant、处于 group 前沿、尚未全员 OFFER，并且成员预测信息齐全。中央按各成员 `declared_at + ready_after_s` 的最大值估计共同 ready 时刻。

OFFER 可以直接创建任务，并隐含该成员已声明；DECLARE 不意味着将来一定准时 OFFER。实际合法候选仍由 OFFER 和序号决定。

### 3.5 正常结束与失败

`finish_epoch()` 在本地状态锁内关闭输入并发送 INPUT_CLOSED，使关闭消息排在已接受的提交之后。关闭输入后仍允许此前任务上报 SUBMITTED/COMPLETED，但不允许再提交新任务。

coordinator 收齐 INPUT_CLOSED 后，检查缺失 OFFER、缺失 group 序号、静态计划缺项以及未完成任务。只有全部已知任务完成、无 inflight，才向所有 endpoint 发送 FINISHED。rank 的 finish_epoch 等待该消息，不是仅等待本地 handle。

`abort()`/`_fail()` 保留首个错误、失败未完成 handle、唤醒等待者并通知中央。协议错误、传输断开、launch/probe 异常或 deadline 均可触发失败。coordinator 不再分发新 grant；已发射 collective 不能伪装成 admission 层取消。

`close()` 是资源清理，不能替代正常 finish。它关闭传输、停止线程并有界 join。Python 无法强制中断任意阻塞的底层调用或不合作的计算回调；真实后端 hang 的最终进程边界清理由上层 harness 承担，不能仅凭 close 返回宣称 GPU 已终止。

## 4. `runtime/` 每个文件的作用

### 4.1 `model.py`：协议数据与本地绑定

定义 GroupSpec、CollectiveSpec、TaskSpec、TaskHint、LocalBinding。前四者使用 dataclass 及字典序列化；LocalBinding 保存不可跨进程序列化的本地对象。

模型校验负责基本形状/字节数和身份合法性；coordinator 校验跨成员一致性；rank runtime 校验 tensor/binding 与 spec。三处解决的问题不同，不能因为模型构造成功就跳过后两层。

### 4.2 `protocol.py`：消息封装

当前协议版本为 1，使用一行一个 JSON 的 NDJSON。`encode_message()`、`decode_message()` 处理序列化和 envelope 校验，`event_message()`、`hello_message()` 构造消息。具体成员状态转移由 coordinator 完成。

| 方向 | 消息 | 含义 |
| --- | --- | --- |
| rank → server | HELLO | 连接身份及 epoch 握手 |
| rank → coordinator | REGISTER_GROUP | 共同 group 定义 |
| rank → coordinator | DECLARE、OFFER | 预测声明、实际提交就绪 |
| rank → coordinator | SUBMITTED、COMPLETED | 发射回执、本地物理完成回执 |
| rank → coordinator | INPUT_CLOSED | 本 endpoint 不再增加任务 |
| coordinator → rank | READY、GRANT、FINISHED | 连接就绪、执行许可、epoch 正常结束 |
| 双向 | FAILED | 上报或广播失败 |

上行事件包含 protocol、kind、epoch、endpoint、event_seq、payload。GRANT payload 包含 task、decision_seq、delivery_seq。

### 4.3 `coordinator.py`：唯一准入权威

主要类是 `CoordinatorState`，输入是控制事件，输出是 `list[Outbound]`，没有 socket 或 tensor 操作。

| 函数 | 职责 |
| --- | --- |
| `apply()` | 校验 endpoint/event_seq，应用事件，检查 deadline，刷新候选并决定后续动作 |
| `tick()`、`next_deadline` | 无新消息时仍推进超时和主动等待，向 server 提供下一检查时刻 |
| `_register_group()`、`_accept_task()` | 合并组/任务，验证成员和共同规范 |
| `_progress()` | 校验 task/decision_seq 与 SUBMITTED→COMPLETED 转移，收齐完成后释放容量 |
| `_refresh_eligibility()` | 容量满时也记录首次 eligible 次序 |
| `_eligible_candidates()`、`_anticipated()` | 分别构造实际候选和预测前沿 |
| `_decide()` | 调用 policy，再校验 Dispatch/Wait，提交 grant 并产生有序 Outbound |
| `_maybe_finish()` | 输入关闭后的完整性检查及 FINISHED 生成 |
| `_fail()`、`_failure_messages()` | 设置终态并产生失败通知 |

检查 deadline 不只发生在队列空闲时：apply 路径也检查，避免持续输入使超时永不处理。终态不继续零等待轮询。

### 4.4 `policy.py`：候选上的策略

`Candidate`、`Anticipated`、`PolicySnapshot` 是策略输入；`Dispatch`、`Wait`、`Idle`、`Done` 是动作类型。`ActiveWait` 保存等待目标、固定 deadline 和轮次。policy 不创建 tensor、不发送 GRANT、不修改成员进度。

- `StaticPolicy`：维护冻结顺序 cursor；队首未 eligible 时 Idle，不跳过。
- `FifoPolicy` / `select_fifo()`：按 `(eligible_seq, task_id)` 选择。
- `LongestTailFirstPolicy` / `select_ltf()`：优先最大 `estimated_comm_s + remaining_tail_s`，再按 eligible 次序和 task_id 打破平局。
- `BoundedLookaheadPolicy`：先取当前 LTF 候选，再比较立即发射与等待 anticipated 目标的估计代价；只有预测等待处于预算且更有利时返回 Wait。
- `make_policy()`：将策略名和静态顺序/等待预算绑定到实现。

coordinator 对 Wait 管理固定 deadline；重算不能不断延长等待窗口，超时后设置 must_dispatch，存在候选时回退。policy 的 Done 不替代中央的完整结束检查。

### 4.5 `transport.py`：有序消息与状态机驱动

`CoordinatorServer` 接受连接，reader 将事件和错误送入 `_inbound`；`_event_loop()` 是 CoordinatorState 的唯一状态拥有者。`_dispatch()` 将 Outbound 写入对应 endpoint 的 FIFO 队列，`_writer_loop()` 实际发送，避免 socket 写入直接占据中央决策路径。

`ControlClient.connect()` 握手；`send()` 在发送锁内生成递增 event_seq 并写出；reader 将消息送入本地接收队列；`receive()` 检查 epoch/endpoint 和带有 delivery_seq 的控制消息。关闭和传输错误唤醒接收方。

HELLO/READY 只说明控制端连接完成，不等于所有通信 group 和所有任务已注册。控制连接独立于 job collective；当前传输不是认证服务或生产级容错网络协议。

### 4.6 `runtime.py`：rank 侧生命周期

`RankRuntime` 是应用主要接口：

- `register_group(group, process_group)`：保存共同组与本地执行组；启动前保存的定义会在 start 后发送。
- `start()`：连接并等待 READY，启动 control/launch/completion 三线程。
- `declare()`：提前声明可预测任务，不绑定实际 collective。
- `submit()`：保存请求、验证本地 binding、发送 OFFER、返回 RuntimeHandle。
- `finish_epoch()`：有序关闭输入，等待中央 FINISHED。
- `abort()`、`close()`：传播失败与清理；`failure`、`grant_order`、`launch_order` 用于状态和验证。

关键内部函数：`_control_loop()` 将 GRANT 转换为 handle 状态与 launch queue；`_launch_loop()` 调用 executor、bind receipt、发送 SUBMITTED 后才允许 probe；`_completion_loop()` 探测物理完成、唤醒 handle、发送 COMPLETED。

`_validate_binding_locked()` 检查 tensor 参数、ProcessGroup/设备等本地条件。可注入 executor 和 completion_probe；probe 必须声明物理完成语义。`wake_completion_on_submit` 控制提交回执后是否主动唤醒观察线程，不改变 SUBMITTED 必须先于 COMPLETED 的合同。

### 4.7 `executor.py`：真实执行与 CUDA 完成桥接

`DirectExecutor.launch()` 调用 binding.launch 并要求返回异步 Work-like 对象，不接受需要 CUDA producer event 的绑定。

`CudaCollectiveExecutor` 为每个本地 ProcessGroup 维护 gate stream：

```text
producer_event → gate.wait_event
  → binding.launch() → backend Work
  → 在 gate 上 Work.wait() 建立完成依赖
  → 在 gate 上记录 done event
  → CudaCollectiveWork receipt
```

`CudaCollectiveWork.is_completed()` 查询 done event；`wait_on(stream)` 把事件依赖插入 consumer stream。keepalive 保存 tensor、Work、event 和额外引用，避免异步工作完成前资源失效。

这里利用的是受支持 PyTorch/NCCL 配置下的 stream 依赖语义，不把 Work.wait 的 CPU 返回当作物理完成。执行器拒绝 `TORCH_NCCL_BLOCKING_WAIT`。`WorkIsCompletedProbe` 调用 receipt/Work 的 is_completed，实际后端必须满足该方法的物理完成合同；不能随意替换为只表示入队完成的对象。

### 4.8 `handle.py`：两种等待

`wait_host(timeout)` 等待本地 COMPLETED/FAILED，超时返回 False，失败抛出原始错误。

`wait_on(stream, timeout)` 先等 receipt 绑定，再调用其 wait_on 插入 consumer stream 依赖；成功返回不等于 GPU 物理完成，也不释放中央容量。普通 Work 没有此能力时明确抛出 NotImplementedError，不退化为模糊的 host wait。

### 4.9 `telemetry.py` 与 `__init__.py`

`EventLog.record()`、`record_at()` 记录事件，`as_dict()`、`dumps()` 输出。可禁用记录或使用 thread_sharded 缓冲减少记录竞争；分片合并按本进程时间排序，不提供跨主机时钟同步。

coordinator 区分 minimal/diagnostic/full 观测模式，协议状态与可选详细观测不能混为一谈。机制计数、decision_seq、实际 launch 和物理完成事件用于解释行为，不应只根据 grant 日志宣称成功。

`runtime/__init__.py` 导出模型、RankRuntime、handle、policy、executor、coordinator 和 EventLog 等公共符号。`CoordinatorServer`、`ControlClient` 从 `runtime.transport` 导入。

## 5. DAG 的模型与校验

### 5.1 `dag/model.py` 的对象

| 类型 | 字段与意义 |
| --- | --- |
| `ComputeNode` | node_id、deps、estimated_duration_s；只描述依赖和估计，不包含 CUDA kernel |
| `CommNode` | node_id、deps、group_id、group_seq、estimated_comm_s、CollectiveSpec |
| `DagJob` | job_id 与有序 nodes 元组；节点输入顺序参与确定性平局处理 |
| `DagGraph` | groups、jobs；expected_task_ids 只含通信，expected_node_ids 含全部节点 |

deps 是同 job 内 node_id。不同 job 不通过任意跨 job deps 相连，但共享 group 的规范顺序可在联合约束图中建立跨 job 通信先后关系。一个 job 必须包含通信节点，其使用的各 group 成员集合须一致。

`validate_graph()` 检查对象类型、ID 唯一性、依赖存在性/重复/自环、估计非负有限、group 成员与 world_size、同 epoch、通信参数和 group_seq。每组 group_seq 必须从 0 连续且唯一。

除了单 job 无环，还要检查 **显式 DAG 依赖 + group_seq 隐式边** 合并后无环。例如 DAG 要求 B 在 A 前，但同组序号要求 A 在 B 前，应在执行前拒绝，不能等运行时死锁。

### 5.2 联合约束与 tail

`joint_predecessors()` 用完整 `job/node` ID 构建前驱映射，同时为同组相邻 group_seq 加边。该映射用于静态顺序与合法性分析；runner 的普通 remaining_deps 仍来自显式节点 deps，组内准入由通信 runtime 保证。

`compute_tails()` 按 job 逆拓扑计算，包含该 job 内规范 group 顺序边：

```text
tail(v) = max(duration(u) + tail(u) for u in successors(v))
终点 tail = 0
```

tail 排除当前节点自身。计算节点 duration 是 estimated_duration_s，通信节点是 estimated_comm_s。LTF 分数再加当前通信估计。该 tail 是 job 内估计最长后继路径，不是所有分支工作量之和，也不是全局多 job 干扰或真实 GPU 执行时间预测器。

### 5.3 静态 FIFO、LTF 与外部顺序

- `build_layered_fifo_order()`：合并 DAG/group/extra_predecessors，构造拓扑层；每层按 job round-robin，job 内保持输入顺序，再投影出通信节点。可显式传 job_order，结果不依赖时长估计。
- `build_static_order(..., 'static_fifo')`：调用上述 layered FIFO。
- `build_static_order(..., 'static_ltf')`：在合法拓扑前沿上先推进可用计算节点，再按通信 LTF 分数选择，最终输出通信顺序。这是离线顺序构造，不模拟真实 ready 时间。
- `validate_static_order()`：要求全部通信恰好一次，且显式/隐式通信祖先必须排在后继之前；可结合额外提交约束。
- `dag_task_spec()`、`dag_task_hint()`：把 CommNode 转为通信 runtime 接口，保留 group_seq 与 estimated tail。

拓扑层只用于构造默认静态顺序，不在运行时添加全层 barrier。静态顺序也不是执行许可：中央仍检查成员就绪、组前沿和容量。

## 6. `dag/runner.py` 的推进实现

### 6.1 一个 runner 对应一个 job

`DagRunner` 接收 graph、job、runtime、epoch/rank、compute_fn、make_binding、绝对单调时钟 deadline、poll_interval，可另传 tails、submit_after、Lookahead 开关和 stop_event。

每个 runner 有一个 `ThreadPoolExecutor(max_workers=1)`，当前保持一个 active compute；多个 job 可由上层并行运行 runner 并共享通信 runtime。它不是多 GPU compute scheduler，也不会自行启动 rank 进程。

初始化建立：

- `node_by_id`、`states`：节点定义和状态。
- `remaining_deps`、`successors`：尚未完成的显式依赖计数和反向边。
- `future`、`compute_node`、`compute_receipt`：当前计算调用及物理完成观察。
- `handles`、`bindings`：已提交通信的本地资源。
- `declared`、`compute_started`：安全预测前沿和时间估计。
- `completed_at_us`：本地观察到节点完成的时间，供上层终点计量使用。

### 6.2 节点状态与每轮顺序

```text
PENDING ──所有 deps 完成──→ READY ──启动/submit──→ RUNNING
                                                   │
                               callback/receipt 或 handle 完成
                                                   ↓
                                               COMPLETED
错误发生于节点处理时 → FAILED，runner abort runtime
```

`run()` 先将无前驱节点设为 READY；每轮执行：

1. `_check_stop()`：检查共享停止、runtime failure 和固定 deadline。
2. `_collect_compute()`：收集计算 callable 返回值或 receipt 完成。
3. `_collect_comms()`：对 RUNNING 通信调用 `handle.wait_host(0)`，非阻塞检查物理完成。
4. 提交所有 READY 通信：构造 binding/spec/hint，保存 handle，节点进入 RUNNING。
5. 没有 active compute 时，从符合 submit gate 的 READY compute 中按输入顺序启动一个。
6. 可选 `_declare_safe_frontier()`：为安全预测的通信发 DECLARE。
7. 没进展时最多等待 poll_interval 或剩余 deadline，不延长总预算。

`_complete()` 只允许 RUNNING→COMPLETED；记录完成时刻，递减每个后继的 remaining_deps，计数归零则 READY。通信 RUNNING 包括等待中央 grant 的阶段，不等同“已在 GPU 上运行”。

### 6.3 计算返回与物理完成

`compute_fn(node, stop_event)` 有两种实际返回合同：

- 返回 None：callable 返回即代表该节点计算完成，适用于同步 CPU 工作或可合作停止的 host 模拟。
- 返回 receipt：必须有 `is_completed()`；future 返回只表示提交动作结束，runner 继续查询 receipt，完成后才解锁后继。可选 `is_success()`、`elapsed_ms()`、`completion_source` 用于检查与记录。

虽然当前构造函数类型注解仍写返回 None，实现明确支持异步 receipt。GPU callable 不能只发射 kernel 后返回 None，否则 runner 会提前解锁。

### 6.4 `submit_after` 的准确含义

`submit_after` 是计算节点到本 job 通信节点的附加映射。`_submit_gate_open()` 检查对应通信是否已经存在于 handles，即 runtime.submit **已经返回 handle**。

它不等待 GRANT、SUBMITTED 或 collective 物理完成。更准确地说，这是“通信请求已被本地接口接受”的门槛。若 compute 真正读取 collective 结果，必须使用完成依赖/正确设备依赖，不能用 submit_after 冒充数据就绪。

runner 校验引用类型和重复项；GPU workload 的组合约束、buffer hazard 和额外环检查还需由上层输入解析负责。仅调用 runner 构造函数不等于完成全部 workload 校验。

### 6.5 安全 Lookahead 前沿

`_declare_safe_frontier()` 只考虑尚未声明的 PENDING 通信：所有未完成直接前驱必须是正在运行且有起始时间的 ComputeNode。存在未完成通信前驱或尚未开始的计算时，不声明为该安全前沿。

预计剩余时间来自计算估计减去本地已运行时间，取最大值并截断到零；转换为相对 ready_after_s 发送。它是预测，不触发真正 OFFER。正常依赖完成后仍走 submit。

### 6.6 完成、失败与上层职责

runner 正常返回要求该 job 的所有节点 COMPLETED，并返回 completed_node_ids 和最大本地完成观察时间。它不调用整个 runtime 的 finish_epoch，也不定义 schema-v2 application terminals；上层等各 job 完成后处理 epoch drain、实验终点和 tensor 验证。

异常时记录当前失败节点（能定位时）、调用 runtime.abort 并抛出；共享 stop_event 帮助其他 job 停止。compute_fn 必须合作响应停止或返回有界的异步 receipt，runner 不具备强制杀死 Python 线程/底层 GPU 调用的能力。

`dag/__init__.py` 导出图对象、runner、静态顺序及 TaskSpec/Hint 转换。`compute_tails` 和 `joint_predecessors` 当前从 `dag.model` 导入，不假设所有辅助函数都在包顶层导出。

## 7. 接入与阅读建议

一个接入方需要完成以下工作，而不是只实例化 coordinator：

1. 跨 rank 冻结共同 group、任务身份、group_seq 和 collective 参数，调用 validate_graph 等校验。
2. 初始化 PyTorch 分布式与实际 ProcessGroup，准备 tensor、计算程序和本地 binding。
3. 启动 CoordinatorServer，构造每 rank 的 ControlClient/RankRuntime，注册 group 并完成 READY 握手。
4. 选择 DirectExecutor 或 CudaCollectiveExecutor 与正确的物理完成 probe。
5. 直接 submit，或为每 job 构造 DagRunner 的 compute_fn/make_binding。
6. 等待 job、finish_epoch、验证结果，最后 close runtime/server；异常路径调用 abort 并由 harness 承担进程边界清理。

这里的顺序是接入职责清单，不是要求所有 ranks 在单线程依次阻塞执行 start。多 rank 启动需要并发进行，server 的 READY 依赖所有 endpoint 到达。

建议阅读顺序：`model.py` → `handle.py` → `runtime.py` 的 submit/三线程 → `coordinator.py` 的 apply/eligible/decide/progress → `policy.py` → `transport.py`/`executor.py` → `dag/model.py` → `dag/runner.py`。

实际接入示例见 [runtime_worker.py](../../examples/jobpacer/runtime/runtime_worker.py)、[DAG 输入适配](../../examples/jobpacer/runtime/runtime_adapter.py)与 [GPU DAG 资源](../../examples/jobpacer/gpu/gpu_dag_resources.py)。裸发和旧 scheduler 的共同 runner 适配见 [dag_comm_adapters.py](../../examples/jobpacer/runtime/dag_comm_adapters.py)。

## 8. 验证边界

从仓库根目录、已有 `.venv` 运行：

```bash
source .venv/bin/activate
PYTHONPATH=src:. python -m pytest -q tests/unit/runtime
PYTHONPATH=src:. RUN_JOBPACER_RUNTIME_REPLAY=1 \
  python -m pytest -q tests/integration/test_runtime_replay.py
PYTHONPATH=src:. RUN_JOBPACER_RUNTIME_NCCL=1 \
  python -m pytest -q tests/integration/test_runtime_replay_nccl.py
```

DAG 相关单元测试位于 `tests/unit/test_jobpacer_dag*.py` 等文件，图、输入适配与 GPU receipt 应分别验证。真实 GPU 测试前确认分配的可见设备及 UUID，不覆盖设备分配。未启用集成检查产生的 skip 不是通过。

本文是源码架构说明，不是本次 GPU 验收或性能报告。CPU mock、Gloo、NCCL 正常执行、故障退出和策略收益有不同证据要求；尤其不能用有序 host launch 单独证明多 communicator 并发安全，也不能直接修改 max_inflight 绕过当前串行合同。
