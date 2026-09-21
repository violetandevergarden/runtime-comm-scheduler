# JobPacer Phase 3.2：DAG 模型与本地推进

日期：2026-09-20。状态：已实施；验收结果见 [Phase 3.2 结果](../result/phase3.2.md)。

依据：[runtime 设计讨论](discussion.md)、[Phase 3.1 计划](phase3.1.md)、[Phase 3.1 结果](../result/phase3.1.md)，以及当前 `src/runtime_comm_scheduler/runtime/` 和 `examples/jobpacer/runtime_*.py` 实现。

## 1. 目标与完成边界

在现有 Phase 3.1 通信 runtime 上支持手写 DAG：每个 job 可以有分叉、汇合、多前驱和多个通信前沿；根据实际完成情况推进依赖，将 ready 的通信节点提交给现有 coordinator 调度。

本阶段验证依赖推进与通信调度的组合正确性，不以性能超过 bare 或静态策略为完成条件。输入暂时由人工构造，但图描述、运行状态和真实执行对象必须分离，便于后续框架捕获与绑定。

首版范围：

- 有限、执行前已知的 job 内 DAG；节点为 compute 或 comm，通信复用现有 all-reduce 模型。
- CPU/Gloo、真实双 rank collective、多 job；每 job 一个串行计算执行通道，允许计算与通信重叠。
- 保留全局 `max_inflight=1`、固定 group 内规范顺序、共同 grant 和独立完成反馈。
- 本地 DAG 推进、输入校验、关键路径摘要、安全的 Lookahead 前沿、日志及失败收尾。
- 手写 JSON 输入与少量确定性样例，原有线性 replay 保持可运行。

不实现任意在线改图、跨 job 数据依赖、同 group 内动态重排、多通信在途、多计算资源调度、自动求导、框架捕获插件、跨 host 或重连恢复。GPU stream 依赖与 NCCL 完成语义另行验收；CPU DAG 完成不代表 GPU 适配完成。

## 2. Phase 3.1 基础与需要调整的假设

当前通信核心可复用：`TaskSpec / TaskHint / LocalBinding`、异步 `submit()`、`RuntimeHandle`、成员 OFFER 匹配、中央策略和有序 launch。此前本分支检查中 runtime 单测 32 项、双 rank Gloo replay 4 项通过；实施前应重新运行，不将这份计划视为新代码的验收记录。

需要显式处理的现状：

1. `runtime_worker.py` 通过逐项循环推进 job，不能直接表达多个独立前沿。
2. coordinator 的 `_eligible_candidates()` 和 `_anticipated()` 都受 `_next_group_seq` 限制；DAG ready 不等于全局 eligible。
3. `consumer_compute_s` 实际是 submit 后、wait 前的独立计算，不能翻译成“通信完成后才能执行”的节点。
4. `runtime_adapter.remaining_tail()` 按线性段累加，包含可重叠部分；不适合作为 DAG 的关键路径估计。
5. 当前没有完成回调接口；不能为了推进 DAG 给每个通信创建一个永久等待线程。
6. replay 的结果校验仍有“一个 job 对应一个 group”的假设，DAG 支持多个 group 后必须按真实 `group_id` 校验，不能按 job 的任务列表排序推断 launch 顺序。

## 3. 架构与职责

```text
手写 JSON → 图校验 / 本地绑定     后续框架 adapter
              │                       │
              └────── 本地 DAG runner ─┘
                         │ declare / submit
                     RankRuntime
                         │ OFFER / 完成反馈
                     Coordinator → Policy
                         │ GRANT
                   现有单 launch worker
```

| 层 | 职责 | 不承担的职责 |
| --- | --- | --- |
| 图描述 | 节点身份、前驱、通信规范、估计 | tensor、线程、ProcessGroup |
| 本地 DAG runner | 节点状态、计算推进、通信提交、完成解锁 | collective 的最终准入与跨 rank 排序 |
| RankRuntime | 本地绑定、handle、launch、通信完成与失败 | 执行整个计算图 |
| Coordinator / Policy | 成员匹配、顺序与容量、选择或有界等待 | 保存完整计算图、执行计算节点 |

图推进实现放在应用侧，不让通信核心依赖 DAG。先使用具体的数据类和 runner，不建设插件系统或通用计算后端接口。

后续框架可以将捕获图映射到此模型做 replay，也可以继续由自身执行引擎推进计算、只通过 `RankRuntime` 提交通信；不强制真实训练由本 runner 托管。

## 4. 图模型与手写输入

### 4.1 最小对象

图按 job 保存节点；group 在 workload 层单独定义，不从 job ID 推导 group ID。

| 字段 | 语义 |
| --- | --- |
| `schema_version` | 手写格式版本，首版为 1 |
| `job_id` | job 身份 |
| `node_id` | job 内唯一且稳定的节点身份 |
| `kind` | `compute` 或 `comm` |
| `deps` | 同 job 内需要完成的前驱 node ID |
| `estimated_duration_s` | 计算节点估计耗时；通信使用 TaskHint 的估计通信耗时 |
| 通信规范 | 复用 TaskSpec / CollectiveSpec：task ID、group ID、group_seq、shape、dtype、reduction 等 |

运行 epoch 由启动参数注入；通信 task ID 在 epoch 内唯一，可由 job ID 与 node ID 稳定组合产生，不用本地线程到达顺序编号。首版要求 job 的通信 group 覆盖该 job 的全部参与 rank，避免引入 rank-local 缺失节点与跨成员依赖语义；一个 job 仍可使用多个 group。

本地绑定单独保存：compute 的同步 callable、通信的绑定创建函数及其对象引用。通信绑定在依赖满足后创建，从而可以读取前驱产生的数据。JSON 不编码任意 Python 代码，不携带 tensor、ProcessGroup 或 CUDA event。

手写 replay 的实际 sleep 时长或扰动参数属于执行配置，必须与策略估计分开。框架绑定与 sleep 模拟使用相同完成契约：CPU compute callable 正常返回表示该节点完成；异步设备调用返回不适用此契约。

### 4.2 图形样例

```text
producer ──┬── comm-A ──────────┐
           └── independent-C ───┤
                               join → compute-next → comm-B
```

`join` 可用零耗时 compute 表示，无需新增节点类型。所有终点完成表示 job 完成，不强制要求单一终点。

该图对应原线性 workload 的 overlap 语义；不能转换成 `producer → comm-A → independent-C`。纯完成依赖并不强制独立计算恰好在 submit 调用之后开始，因此迁移实验需说明这一点；若保留精确的 submit-before-compute 时序，应作为额外执行顺序约束说明，不能伪装成通信完成边。

### 4.3 执行前校验

优先使用标准库拓扑排序及现有模型校验，不引入图算法依赖。

- 拒绝重复 ID、缺失或重复依赖、自依赖、环、非法节点类型、非有限或负耗时。
- 校验 TaskSpec 所属 job、通信 task ID 唯一、group 存在、参与成员和 collective 参数一致。
- 每个 group 的通信序号必须唯一、从 0 连续编号，作用域是整个 epoch，而不是各 job 分别编号。
- 在所有 job 的图上加入每个 group 的规范顺序边，校验组合约束无环；跨 job 共享 group 也必须纳入。
- 静态调度序列必须覆盖全部通信且无重复，并与计算依赖和 group 顺序相容；不合法顺序执行前报错。
- 所有 rank 从同一份已校验 manifest 启动，保存规范化输入摘要。harness 检查成员一致性；runtime 原有通信匹配仍保留，不能用输入摘要取代它。

例如 DAG 要求 `comm-1 → comm-0`，group 却要求 `comm-0 → comm-1`，即使原 DAG 无环，也必须拒绝，不能依赖 epoch 超时发现。

## 5. 本地推进与完成语义

### 5.1 节点状态

采用最小状态：`PENDING → READY → RUNNING → COMPLETED`，错误进入 `FAILED`。通信的 RUNNING 表示已交给 runtime 管理，不代表已获得 grant 或已实际 launch；后两者继续由现有 runtime 日志记录。

维护每个节点的未完成前驱数、后继列表、ready 集合和运行中对象。节点至多启动一次，完成事件至多解锁后继一次。只有真实完成才能解锁完成依赖；submit 返回、grant 或 backend 返回 Work 都不算节点完成。

### 5.2 首版执行机制

每个 job 一个推进循环和一个串行 compute worker：

1. 收集已完成或失败的 compute future、通信 handle，更新图状态。
2. 把所有 ready 通信依次提交给 RankRuntime，不等待其 grant 或完成。
3. 计算通道空闲时，按稳定 node ID 顺序启动一个 ready compute。
4. 暂无可推进动作时有界等待，再检查完成、失败及 job deadline。

通信完成首版可复用 `RuntimeHandle.state` 和已终态的 `wait_host(0)`；compute 使用标准库 future。轮询间隔显式记录并可配置，不忙等、不新增每通信一个线程，也不调用底层 Work 自行判断完成。若以后测出轮询成本显著，再增加统一通知接口。

计算通道只接收当前要启动的一个计算任务，避免提前排入长队而隐藏其排队状态。compute callable 不在推进线程、runtime 控制线程或完成探测线程中执行。

“每 job 一个计算通道”仅是本次 CPU 模拟的资源模型，不表示多个 job 在同一 GPU 上拥有独立算力。独立计算不一定能并行，通信调度效果结论应限定在此模型内。

### 5.3 就绪与全局准入

区分三个时刻：

- DAG-ready：本地所有完成依赖满足。
- submitted/offered：本地资源绑定完成并已提交 runtime。
- eligible：所有成员 OFFER 到齐，且符合 group 顺序。

runner 不复制 coordinator 的容量门控，也不只挑一个通信交上去；同 group 的后项可以提前 OFFER，但仍不能越过前项执行。不同 group 的 ready 节点可以同时成为策略候选。

不移除 `group_seq`，不按本地 ready 顺序重新编号，不通过每节点新建 group 人为制造可重排空间。同 group 动态重排属于后续协议扩展。

## 6. 策略衔接

### 6.1 FIFO 与静态基线

DynamicFIFO 继续按中央首次 eligible 的顺序选择，不改为本地节点列表顺序。StaticOrder 继续忠实等待队首，但 DAG 静态序列不能调用只理解旧线性 Job 的 plan builder 直接生成。

首版为静态对照提供经联合依赖校验的显式通信顺序或稳定拓扑顺序。所有策略使用相同 DAG、绑定、计算资源模型和容量，不能把改变图语义当作策略收益。

### 6.2 LTF 的 DAG 摘要

在 job 内加入同 job 的 group 顺序约束后，计算通信完成之后的最长估计后继路径。跨 job 共享 group 的顺序用于全局合法性校验，不把其他 job 的计算计入本 job 的 tail：

```text
tail(v) = max(duration(u) + tail(u) for u in successors(v))
tail(sink) = 0
```

当前通信自身耗时不计入 tail；并行分支取最大值，不求和。估计固定于本轮输入，不读取本轮未来实际 sleep 或测量完成时间。

这是关键路径优先级，不宣称等于精确剩余 JCT：汇合的其他分支可能仍未完成，串行计算通道存在排队，其他 job 会竞争通信。首版不做在线最优剩余时间求解，也不把旧线性 remaining_tail 与新指标混合比较；输出记录 tail 的定义与数值。

### 6.3 Lookahead 的安全前沿

先用 FIFO/LTF 验证 DAG，再接 Lookahead。完整图已知不等于全部通信都应立即 DECLARE。

首版允许预测的未 ready 通信满足：所有未完成的直接前驱都是已经启动的本地 compute；其他前驱已完成。尚未启动的 compute、未完成通信或未知完成时间都不产生有限 ready 预测。

对多个运行前驱取预计剩余时间最大值，基于估计耗时与已运行时间计算，不读取真实未来扰动。仅在首次满足条件时声明；首版不新增反复更新预测的协议。预测不可靠时不提前声明，直接 ready 后 submit。

coordinator 保留成员预测齐备、group 下一序号、固定等待 deadline 和到期回退规则。必须验收：不会因等待依赖未服务通信的后继而主动阻塞其前驱；重复事件不延长同轮等待预算。预测可能过期，日志必须保留预测依据，不能把它当准确 ready 时刻。

## 7. 失败、结束与资源生命周期

- 任一 compute、绑定创建或通信失败，停止本 job 新节点执行，并通知共享 runtime 进入现有 fail-stop；不再运行失败节点后继。
- 不把访问私有 `_fail()` 扩散到新代码。实施时增加一个最小公共 abort/fail 入口，复用现有失败广播与 handle 唤醒，并迁移现有 worker 调用。
- 所有本地 job 成功完成后，由 rank 的共同拥有者调用一次 `finish_epoch()`；不能每个 job 分别关闭共享输入。
- 全部图终点完成才算 job 成功；已经声明的通信最终必须提交，否则失败，不新增撤销语义。
- 保持数据对象存活到相关计算、通信与消费者安全结束；通信 binding 由 runtime 管理，但不能因此提前释放 DAG 下游需要的数据。
- 设置整体 deadline，超时报告未完成节点、未满足前驱、运行计算和通信 handle 状态。不能按节点不断重置总预算。
- Python 线程不能安全取消任意阻塞 callable；内置 sleep 使用可中断等待，未知 callable 卡住时由已有进程 harness 有界终止并记为失败。不宣称已提交 collective 被取消。

## 8. 代码落点与兼容范围

| 位置 | 计划改动 |
| --- | --- |
| `src/runtime_comm_scheduler/dag.py`（拟新增） | 具体图模型、校验、关键路径计算、DAG runner；位于 runtime 核心之上，规模确实需要时再拆文件 |
| `examples/jobpacer/runtime_adapter.py` | 保留历史线性 workload 的映射；DAG 映射由正式库模块提供 |
| `examples/jobpacer/runtime_worker.py` | 选择线性或 DAG 模式，复用 group 创建、runtime 生命周期和 tensor 校验 |
| `examples/jobpacer/run_runtime_replay.py` | 增加 DAG 输入并完整传递参数；按 group 校验真实 launch 投影和预期节点全集 |
| `src/runtime_comm_scheduler/runtime/runtime.py` | 必要的最小公共失败入口；不加入图遍历与计算线程 |
| `benchmark/phase3/*.json` | 可复现的手写 DAG benchmark 输入 |
| `tests/unit/`、`tests/integration/test_runtime_replay.py` | 图与推进测试、真实双 rank DAG replay、原线性回归 |

不修改历史 Phase 1/2 原始结果，不替换旧 scheduler，不要求本阶段迁移所有历史 workload。新输入尽量沿用现有参数和结果结构，新增字段说明含义。

结果校验必须比较 manifest 中预期任务与实际完成任务的集合及数量，拒绝缺失、重复和非预期任务，避免空结果的 `all(...)` 被误判成功。每个 group 按实际成员检查完整 launch 投影；不能要求不参与某 group 的 rank 具有相同全局序列。

## 9. 验收与观测

### 9.1 确定性检查

| 场景 | 通过条件 |
| --- | --- |
| 线性链 | 前驱完成前后继不启动，现有 Phase 3.1 回归不受影响 |
| 分叉与汇合 | 独立分支能推进，join 等全部前驱；节点不重复执行 |
| 通信等 grant | 不阻塞同 job 的独立计算和其他 ready 通信提交 |
| 同 job 多 group | 可产生多个候选，最终 launch 仍由共同 grant 决定 |
| 同 group 分支错位 | 后项先 ready 仍不能越序；联合约束死锁在启动前被拒绝 |
| rank 到达顺序相反 | group 的完整 launch 投影一致，all-reduce tensor 正确 |
| Lookahead | 安全前沿内外、预测早晚、固定 deadline 和回退均符合规则 |
| 输入错误 | 环、缺失依赖、序号冲突、非法成员和非有限估计明确拒绝 |
| 执行错误 | compute/绑定/通信失败唤醒等待者，不执行后继，有界退出 |
| 结束 | 预期节点全部完成，输入只关闭一次，未排空不报告成功 |

优先使用事件或可控假 handle 验证推进因果，不依靠极小 sleep 时间差断言线程顺序。真实 Gloo 验收覆盖 diamond、overlap、多 group 与 rank skew；保留原有四个线性 replay 用例。

### 9.2 指标

本地记录节点 ready、计算 start/complete、通信 submit、观察到的完成与失败；通信 grant、launch、中央完成继续使用现有日志。区分：

- 依赖未满足、计算通道等待、runtime eligible 等待和容量等待。
- job 起止与总体 makespan，通信完成不等于整个 job 完成。
- 预测 ready、tail、实际前沿变化、轮询间隔和实际计算样本。

记录规范化 DAG、输入摘要、策略、seed、world size、group 成员、容量和代码版本。跨 rank 原始单调时钟不直接相减；轮询观察时间不伪装成真实设备完成时间，各类等待不未经证明相加成 makespan 分解。

机制通过后做小规模静态/FIFO/LTF/Lookahead 对照，先看选择与依赖是否可解释，再看 JCT 和 makespan；不承诺动态策略普遍获胜，不将 CPU sleep 结果外推到训练 GPU。

## 10. 里程碑与交付物

| 里程碑 | 实施内容 | 完成条件 |
| --- | --- | --- |
| M0：冻结语义 | 图格式、完成边、group 约束、计算通道与示例 | 手写样例可校验，联合环和非法静态顺序能被拒绝 |
| M1：本地推进 | compute worker、异步通信 handle、完成解锁、失败 | 假 runtime 下分叉/汇合/独立推进和错误测试通过 |
| M2：真实通信 | 接入现有 worker 和 replay，FIFO/静态基线 | 双 rank Gloo 结果、预期任务全集及 group 投影正确 |
| M3：策略摘要 | DAG tail、LTF、安全 Lookahead | 可控候选选择符合预期，无依赖诱发的主动等待错误 |
| M4：回归与小对照 | 原线性回归、故障注入、日志及结果文档 | 机制矩阵通过，效果和限制可解释 |

交付代码、手写 DAG 样例、可重复运行命令、测试以及 `docs/JobPacer/result/phase3.2.md`。结果文档应记录实际命令与产物路径，不以本计划中的目标替代已完成证据。

## 11. 后续框架接入边界

现在保留稳定节点身份、通信规范与本地绑定分离、计算与通信职责分离即可；不提前实现 Megatron/PyTorch 图捕获插件。

后续真实框架需另外解决：

1. 从捕获结果生成规范通信身份和所有成员一致的 group 顺序，而不是按各 rank 的 hook 到达顺序编号。
2. 将设备依赖“已建立”和物理完成分开；明确哪些边可以通过 event/stream 依赖推进，哪些必须等待完成。
3. 验证 producer event、consumer stream、tensor 生命周期及实际 NCCL 完成探测。当前 `wait_on()` 等接口存在不等于这些语义已通过验收。
4. 明确共享 GPU 计算资源和多通信并发模型，不能沿用 CPU 每 job 独立计算通道假设解释性能。

本阶段完成标志是：手写 DAG 能在既有 runtime 上正确推进并调度真实通信，多个前沿不退化为逐通信阻塞链，同时边界清楚、可被后续 adapter 复用。
