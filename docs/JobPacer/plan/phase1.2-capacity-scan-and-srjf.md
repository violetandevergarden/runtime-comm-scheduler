# Phase 1/2：FIFO/LTF 容量扫描与最短剩余 job 优先实验

日期：2026-09-19。本文仅规定下一步实施与实验，不代表功能或结果已经完成。

## 1. 目标与顺序

继承 [workload 扩展计划](phase1.2-workload-expansion.md) 的测量口径，以批次 `20260919T083615Z-10a786` 为背景，按以下顺序推进：

1. 先扫描静态 FIFO 和静态 LTF 的在途容量，判断有限并发是否优于单在途或不限在途。
2. 分析容量结果后，增加静态最短剩余 job 优先策略，在相同容量和执行路径下比较排序效果。

将容量选择与策略选择分开，避免同时改变两个因素后无法解释差异。当前数据只说明单在途在多个 CPU/Gloo 场景中代价较大，尚未证明单项通信没有占满带宽，也未证明有限并发能超过 bare。

本轮使用已有线性 workload、两 rank CPU/Gloo、真实 all-reduce。暂不扩展 DAG、GPU/NCCL、自适应带宽控制、抢占或新的在线协调协议。

## 2. 共用测量要求

- 数值校验在应用结束、通信排空后执行，不进入逐任务推进关键路径。
- 使用统一 release、application end、communication drain、validation 和 harness 边界。
- 分开记录应用等待绑定、底层 wait、collective 调用与完成探测；探测时间不称为精确设备完成时间。
- 确认已有等待归因修复有效，切分事件包含 admit、collective call start、completion observation 等全部状态变化边界。
- 每个 workload 各场景共享固定 manifest、profile、成员、计算/消费语义、backend 和软件环境。
- 每个 repetition 内随机交错场景，记录固定 seed；各 replay 顺序运行。
- 使用独立 batch、manifest 和文件哈希；保存相关源码哈希或可还原快照。新报告不覆盖历史原始数据。
- 主实验固定 completion poll interval 为 1 ms；需要检验接近该尺度的差异时，另开 0.1 ms 批次，对所有被比较场景一致修改，不混合统计。

## 3. 第一阶段：FIFO 和 LTF 容量扫描

### 3.1 矩阵

对 FIFO、LTF 分别运行 `max_outstanding = 1、2、3、不限`，另加入 bare：每个 workload 共 9 个场景。沿用当前 CLI 的 `0` 表示不限，manifest 和图表显示为 `unbounded`。

| 选择方式 | 容量 |
| --- | --- |
| bare | 不限；不经过 scheduler |
| 静态 FIFO | 1、2、3、unbounded |
| 静态 LTF | 1、2、3、unbounded |

场景名显式包含策略和容量，例如 `fifo_k2`、`ltf_k3`，不再仅使用 serial/unbounded 两档命名。

沿用已有 8 个 workload：comm-heavy、staggered-compute、long-short-chains、大小消息的 large-first/large-last，以及 overlap-window 的 short/medium/long。优先检查通信密集、大小消息和长短链；其他场景用于观察错峰与计算重叠是否改变容量效果。

当前 job 逐项等待后再推进，实际可用并发受 job 数和 ready 状态限制。2-job 场景中容量 3 与 unbounded 可能不构成有效区别。仍保留初始扫描，并记录实际峰值在途数量；不得把配置容量当作实际并发。

### 3.2 实施检查

- runner 支持显式策略/容量矩阵，并将实际矩阵写入 manifest；finalize、summary 和绘图只读取 manifest。
- FIFO/LTF 在各容量下保持相同静态 Plan，实际提交投影和各 group 顺序仍须匹配。
- 对有限容量 k 验证每个 rank 的占用不超过 k；占用包含已准入未发射任务，直到完成探测后释放。不能只数已经调用 collective 的任务。
- 不将 rank-local 容量验证写成全局完成屏障。保持现有执行语义，不为扫描引入额外跨 rank 同步。
- 在 smoke 中检查 k>1 的正常结束、错误唤醒和排空路径，避免扩大容量后只检查正常退出码。
- 保存各场景 Plan，标记 FIFO/LTF 相同 Plan 的场景；它们的差异不能解释为排序收益。

### 3.3 运行步骤与预算

1. 先运行针对性单元检查和两 rank 小规模通信检查。
2. 每个 workload 每场景 2 次 smoke：完整矩阵最多 `8 × 9 × 2 = 144` 次 replay，按家族分批。
3. smoke 通过后每场景 10 次探索：完整矩阵最多 `8 × 9 × 10 = 720` 次 replay。
4. 每个家族先记录耗时与资源消耗，再安排后续批次。不要并行启动不同 replay 制造额外竞争，也不自动增加规模。

输入相同不意味着可复用旧批次 bare 作为配对基线：本次 bare 应与容量场景一起交错重跑。旧结果仅作历史参考。

### 3.4 指标与判断

必须报告：

- application makespan、每个 job 完成时间、每次运行内所有 job 完成时间的平均值。
- 配置容量、实际峰值在途，以及在途数量随时间的分布；分别说明 admission 占用和实际调用后的在途口径。
- ready-to-admit、队首空档、容量占用、待发射和未知归因时间；不可直接将互相重叠的指标相加。
- 相同策略下 k 与 1、k 与 unbounded 的配对差值；相同 k 下 LTF 与 FIFO 的差值；与 bare 的端到端差值。
- 原始点、样本数、中位数、P10–P90；P10–P90 是样本分布，不是置信区间。

绘制容量—makespan 曲线，同时展示各 job 和平均 job 完成时间。按以下情况解释：

- 若容量从 1 增加到 2/3 明显改善，随后趋平：存在并发重叠收益，但不能直接断言带宽已经饱和。
- 若有限 k 优于 unbounded：作为过度并发干扰的候选证据，使用新 seed 复核后再讨论稳定性。
- 若 unbounded 最好：不强行选一个有限容量，不提前实现自适应控制器。
- 若实际峰值远低于配置容量：解释 workload 的 ready/依赖限制，不把平台期解释为带宽上限。
- 若静态 Plan 挡住 ready 工作：增加容量不能自动消除这种队首阻塞。

通信字节数除以应用时长只能称为逻辑吞吐，不能冒充实际链路带宽利用率；collective 的线上传输量与算法、拓扑有关。

第一阶段交付各 workload 的容量曲线、实际并发观测和第二阶段容量选择理由。不可只汇报事后最快的一个点。

## 4. 第二阶段：最短剩余 job 优先

### 4.1 明确定义

新增策略建议命名 `srjf`（shortest remaining job first）。它是静态、非抢占的通信顺序构造策略，不是最短单次 collective 优先，也不是运行时精确获知剩余时间。

为与当前修正 LTF 可比，复用相同的零 admission delay 关键路径估计。对 job 当前候选通信 i：

```text
remaining(i) = max(estimated_comm_i, consumer_compute_i)
             + sum(producer_compute_j
                   + max(estimated_comm_j, consumer_compute_j)
                   for j > i)
```

每次从各 job 尚未排入 Plan 的下一项中，选择 `remaining(i)` 最小者，随后推进该 job 的计划位置。LTF 取同一分数的最大值，SRJF 取最小值，以便隔离优先级方向的影响。

当前候选 producer 不包含在分数内，这是“候选已经 ready”的评分假设；静态构建时它未必实际 ready。因此策略仍可能等待固定队首，不应称为在线 work-conserving 策略。

估计只读取共同 workload/profile，不读取运行结果或未来真实扰动。相同分数按稳定 job ID、ordinal 破同分，不能由线程到达顺序决定。记录每步候选分数、选择与 tie-break。

### 4.2 最小实现边界

- 在现有 Plan builder 中复用评分和候选构造，增加 SRJF 选择方向及 CLI 注册。
- 更新 replay、runner、summary 和图表的策略名称与 diagnostics；不建立新的策略插件框架。
- 保持 collective 提交、容量、完成探测、消费语义与 FIFO/LTF 一致。
- 回归检查覆盖：长短链选择、分数相同的稳定顺序、job 内顺序、compute/communication overlap 评分，以及跨 rank Plan 一致性。
- 失败或较差结果仍保留，不按观察到的收益修改评分定义。

### 4.3 对照与预算

完成第一阶段分析后冻结第二阶段容量集合，再开始 SRJF 性能实验：

- 保留 k=1 作为语义清晰的参照，以及 unbounded 作为不主动限制并发的参照。
- 若扫描显示 k=2 或 k=3 有值得验证的收益，额外纳入一个或两个有限容量；记录选择依据。未出现该迹象时不额外增加档位。
- FIFO、LTF、SRJF 必须在完全相同的容量集合上比较，并在新批次中重跑；bare 每轮一次作为共同基线。
- 设容量档位数为 C，单 workload 每轮为 `3C + 1` 个场景。先每场景 2 次 smoke，通过后 10 次探索；按家族评估总耗时。

第二阶段覆盖已有 workload，重点分析 long-short-chains 与 mixed-message-sizes，staggered-compute 用于检查是否为了短 job 牺牲整体推进；同构场景用于确认相同 Plan 对照，不必将其包装成策略差异。

用第一阶段选择容量、第二阶段新样本复核，可减少在同一批数据上挑选“最佳容量”再声称收益的问题。不要为每个策略各取不同最佳容量后称为纯排序比较。

### 4.4 主要问题

1. SRJF 是否降低平均 job 完成时间和短 job 延迟？
2. 改善短 job 的代价是否是长 job 延后或整体 makespan 增长？
3. 同一容量下，SRJF/LTF 是否主要重新分配完成先后，而不改善总体吞吐？
4. 容量增加后，排序效应是否被弱化，还是仍存在静态队首阻塞？

平均 job 完成时间应先在每次运行内计算，再跨重复做统计，不能用各 job 中位数的平均值代替。同时保留各 job 结果、最长 ready-to-admit 等待和整体 makespan。

本轮 workload 有限且无持续到达，不能据此证明 SRJF 不会饥饿。在线公平性、aging 或长任务保护留到后续有持续到达需求时研究。

## 5. 完成与报告要求

第一阶段与第二阶段分别交付独立 batch、原始 trace、manifest、summary、图表和 analysis，说明命令、环境、耗时、通过/失败/跳过及未验证边界。

正式样本必须满足：tensor 正确；成员与任务覆盖完整；实际提交顺序符合共同 Plan；有限容量占用合法；所有通信观察完成后才校验；必填时间边界闭合；文件哈希和 summary 样本集一致。

最终报告分开回答“容量带来的效果”和“同容量下排序带来的效果”。没有超过 bare、没有统一最优容量、短 job 改善却整体变慢，都属于有效结果。CPU/Gloo 与 sleep workload 结论不能外推为 GPU/NCCL 或真实训练性能。
