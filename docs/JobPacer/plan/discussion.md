# JobPacer runtime 设计讨论总结

日期：2026-09-16。实施范围以 [Phase 3.1 计划](phase3.1.md) 为准，后续阶段的设计方向不等于本阶段必须实现的功能。

## 1. 总体决定

新 runtime 独立设计和重写，不要求兼容或复用现有 Plan、scheduler、Work、executor、任务身份字段。先定义正确性、执行语义与实验要求，再实现代码。旧实现可作为历史实验对照，不进入新核心依赖链。

采用中心化 coordinator，依据运行状态决定 collective 准入；各 rank 保存真实执行对象，执行共同 grant。Policy 逐次产生顺序，coordinator 负责合法性检查、提交决定和有序发布，不提前替 policy 排完整序列。

Phase 3.1 聚焦线性 job、单竞争资源和严格串行通信。DAG、多进程部署和多链路调度分阶段展开，不提前建设完整通用框架。

## 2. 动态 runtime 的价值

静态 Plan 根据预估情况提前确定跨 job 顺序，队首未 ready 时可能阻塞其他已 ready 的任务。Runtime 利用实际到达、完成和 job 进度，减少预测错误导致的空闲，也能有界等待预计即将到达的重要任务。

例如 A 在 10 ms ready、B 在 2 ms ready，两项通信各耗时 3 ms，无跨 job 依赖且要求串行：静态 A→B 在 16 ms 全部完成；动态先 B 后 A 可在 13 ms 完成。

Runtime 也允许运行中注册任务，不要求启动前知道全集。但动态调度不保证更快：控制通信、状态汇聚、决策和主动等待均有成本；稳定 workload 的静态顺序可能很好。通信并发也可能优于串行，不能把动态选择收益与限制并发收益混淆。

## 3. 职责与任务模型

```text
Workload / framework adapter
    │ declare（可选） / submit
    ▼
Rank Runtime ─── OFFER / SUBMITTED / COMPLETED ───► Coordinator
    ▲                                                 │
    │                                       合法候选 → Policy
    └────────────────── GRANT ────────────────────────┘
    │
单 host launch worker → collective backend
```

| 对象或组件 | 职责 |
| --- | --- |
| TaskSpec | 跨 rank 一致的任务身份、group、collective 参数与约束 |
| TaskHint | 预计 ready、通信时长、remaining tail；与任务身份分开 |
| LocalBinding | 本地 tensor、ProcessGroup、producer event、执行入口 |
| Rank Runtime | 接收提交、保存绑定和 handle、处理 grant、独立探测完成与传播错误 |
| Coordinator | 成员状态、顺序和容量检查、提交并发布共同决定 |
| Policy | 从合法候选选择或有界等待，不接触线程、网络和 tensor |

submit 保存请求后立即返回 handle，不等待 grant。应用在消费位置等待；runtime 独立上报完成。建立 consumer stream 依赖与等待主机可观察的物理完成需分开定义。

## 4. ProcessGroup 内的顺序一致性

同一 collective 具有跨 rank 一致的任务身份、group 内序号及兼容参数。序号由 adapter 按共同语义产生，不能按本地线程到达顺序独立编号。Coordinator 校验成员信息，收齐所有成员 OFFER，仅让符合 group 顺序的任务进入候选。

选择一旦成为 grant，即追加到不可重排的共同序列。各 rank 接收自己参与的投影，由唯一 launch worker 按序调用底层 collective，不能交回 job 线程自由发射。

全局 decision_seq 标识决定；每 endpoint 连续的 delivery_seq 检查本地消息缺失，避免将不参与的全局任务误判为丢包。必须同时实现有序发布、单接收队列和单执行入口，只有序号或优先队列不够。

核心不变量：每个 rank 实际提交记录始终是其 grant 投影的前缀；同 group 所有成员因此执行同一序列的前缀。可以进度不同，不能交换、跳项或重复提交。受管理 group 的全部 collective 必须经过 runtime。

## 5. 多个 collective 在途与 policy 的能力

顺序一致不要求上一任务完成：rank 0 已提交 A、B，rank 1 才提交 A，仍可保持 A→B。顺序进度、提交进度、完成进度应分开。

但 host 次序一致不独立证明多 communicator 的设备并发安全。Backend 还需满足实际 PyTorch/NCCL/CUDA 配置的设备端排序要求，不能只取消完成屏障。参见 [NCCL 多 communicator 并发说明](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html#using-multiple-nccl-communicators-concurrently)。

Grant 是不可撤销的决定提交点：某成员可能已经执行，其他成员不能撤回或插队。提前 grant 越多，policy 对新到达任务的反应空间越小。若启动时 grant 全部任务，就退化为静态顺序。

后续可分别限制 max_inflight（已 grant 未全局完成）和 max_unsubmitted（未收齐提交回执）。容量是上限，不要求 policy 填满；可以等待未来任务。Phase 3.1 只实现全局一个在途任务，不做撤销或第二套容量控制。

## 6. 候选与有界等待

eligible 是成员和依赖条件已满足的任务；anticipated frontier 是可能很快到达的前沿。线性 job 只考虑当前下一项，不能因远端后继 tail 很大而等待它、阻止其前驱获服务。

- Dynamic FIFO 按首次进入 eligible 的顺序选择，稳定 ID 仅破同分。
- Dynamic LTF 使用统一定义的 remaining tail，估计不读取真实未来扰动。
- Bounded lookahead 为目标建立固定截止时间；状态更新可提前重算，不能反复刷新预算。到期有候选就回退服务。

等待需比较预期收益和空闲成本；仅目标 tail 更大不够。第一版采用透明的简单启发式，并记录估计与实际结果，不追求全局最优。

## 7. 静态与动态效果比较

主要对照在新 runtime 内完成：StaticOrder 与动态策略共用协议、执行 backend、完成探测和日志。StaticOrder 输入任务 ID 序列，下一项未 eligible 就等待，不能跳过；完整序列不是其他策略的启动条件。

旧系统对新系统的整体对照另行进行，用于衡量迁移总效果，不能单独归因 policy 收益。

实验应保持 workload、成员、collective 参数、计算/消费位置和并发限制相同。Profiling 估计与实际扰动分离；首轮策略共享固定估计，在线更新估计另设实验。

按 (seed, epoch, job, task, rank) 固定计算扰动样本，不按线程次序取随机数；固定计算时长，而不强制固定请求绝对 ready 时间，因为调度改变后续推进属于真实反馈。

记录 job 完成时间、总体 makespan、eligible 等待、主动等待、grant 到提交和完成探测时间。区分无候选、静态队首阻塞、主动等待、容量满。中央接收时间与本地执行时间分开，不直接相减各 rank 原始时钟。

先无扰动与确定性错位，再多 seed 扰动。CPU/Gloo 验证机制；真实 GPU 实验才说明相应通信环境的性能，并保留 bare 并发基线。

## 8. Phase 3.2：DAG

DAG 改变合法候选：同 job 可以有多个独立前沿，不能永远用一个 job 队首表示。区分提交顺序约束与完成依赖，不能把所有边都解释为 host 等待物理完成。

DAG executor 位于 workload/framework 层，runtime 管通信准入，policy 可读关键路径等摘要，无需托管计算节点。DAG 独立不代表同 group 内可以由各 rank 自由重排；若未来支持，需中央共同分配执行顺序。

Phase 3.1 只实现线性推进，不做通用 DAG schema、环检测或计算执行器。

## 9. Phase 5：中心化管理与多进程执行

继续采用中心化 coordinator。Phase 3.1 可放在 rank 0 内，Phase 5 再独立部署。控制消息不携带 tensor、ProcessGroup、CUDA event 或 closure，不使用受调度 collective 作为控制 rendezvous。

每 GPU 一个 worker 适合托管式 replay。真实多个 job 进程共享 GPU 时，各进程执行端仍持有自己的 tensor 和 group；可增加 GPU controller 管理提交交接，不强制搬迁执行对象。

跨进程先发 A 许可再发 B 不能保证实际调用顺序。需要交接时，收到 A 实际 SUBMITTED 后才允许 B；不必等待 A 完成。跨 CUDA context 并发需单独验收。

Job Agent 可负责生命周期、注册、状态汇聚，不必每 job 强制新增进程。只有汇聚而非逐条转发才能减少中央成员级事件；中央逐 collective 决策仍有吞吐上限。先测事件速率、排队和 grant 延迟，再决定批处理或按竞争域下放。批量 grant 会提前锁定顺序，扩展吞吐与动态性存在取舍。

身份上分开 job、group、endpoint 与 GPU，不同 job 的 rank 0 不是同一个执行端。本阶段不做服务发现、恢复、sidecar 或独立 Job Agent。

## 10. Phase 6：多链路

多链路改变可同时执行的任务组合。后续用资源需求与占用替代单个在途计数；跨多资源任务由中央一次性检查、预留，避免部分占用后互相等待。

资源共享不必互斥。可以先采用不相交任务并行，再通过测量增加容量与干扰模型；真实路由需要拓扑/backend 信息，不能只凭成员集合推断。

等待链路 X 上的未来任务不应自动阻塞仅用 Y 的任务，后续需扩展等待作用范围。资源可并行与 backend 顺序安全分别校验。中央调度可以继续保留，分域不是必选项。

Phase 3.1 仅采用全局一个抽象竞争资源，不提前实现资源图、链路发现、多资源预留或任务集合优化。允许后续按需要修改 API。
