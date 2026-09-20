# Phase 1/2：扩展 workload 与探索性对照实验计划

日期：2026-09-19。计划已于同日实施完成。

实施记录：

- 归因修正与确定性回归检查已完成；历史批次的重算说明保存在 `benchmark/phase1.2/result/batches/20260919T034952Z-9caa6a/attribution-v2.md`，未覆盖旧报告。
- 8 个 workload 已冻结在 `benchmark/phase1.2/workloads/`；smoke 批次 `20260919T082148Z-2dd849` 完成 96/96 次 replay。
- 10 次探索批次 `20260919T083615Z-10a786` 完成 480/480 次 replay，并生成 summary、分析、具体 Plan 和 9 张 SVG 图。
- `overlap-window-long` 的 0.1 ms 定向复核批次 `20260919T091635Z-ceff56` 完成 60/60 次 replay，并生成独立图表；没有与主批次混合统计。
- 当前阶段仍只覆盖单机 CPU/Gloo 与 sleep 计算；GPU/NCCL、多机和真实训练保持为非目标。

## 1. 目标与依据

在完成测量修正的基础上，增加 workload 的结构和负载差异，观察静态顺序、通信容量限制和在线选择在什么条件下有帮助，以及控制开销何时盖过收益。先扩大场景覆盖，不直接扩大现有三个场景的重复次数。

依据：

- [测量语义修正计划](../process/phase1.2-measurement-fix.md)。
- [最新完整六场景批次](../../../benchmark/phase1.2/result/batches/20260919T034952Z-9caa6a/analysis.md)。
- [轮询敏感性批次](../../../benchmark/phase1.2/result/batches/20260919T044046Z-17c1d7-poll-sensitivity/analysis.md)。

以上两个 benchmark 链接应从仓库根目录的 `benchmark/phase1.2/result/batches/` 查阅；结果可能不随源码分发。

已有数据提示：修正后的 FIFO/LTF 整体差异较小；delayed 场景的静态队首阻塞明显损害短 job；ready-first 实现的协调开销不可忽略。新增实验用于检验这些现象的适用条件，不以得到动态策略或 LTF 获胜的结果为目标。

## 2. 实验前置关口：修正等待归因

当前 `visualize.py:scheduler_state_intervals()` 使用 `collective_call_start_ts` 判断在途状态，却没有将该时刻加入区间切分事件。应先修正，防止实际通信区间被错误计入 scheduler idle。

实施要求：

1. 补齐影响分类的事件边界，并区分 admission 后待发射、实际通信在途和完成探测。
2. 容量占用按 scheduler 的实际语义重建，不能简单等同于底层 collective 已开始调用。
3. 分别统计无任务 ready、容量占用、静态队首阻塞、待发射、协调等待和无法归因的时间。
4. ready-first 的未知归因区间不能通过 `work_conserving_idle=0` 表述为没有空闲；记录本地 ready 与全局 ready 的区别。
5. 补一个确定性事件序列回归检查：admit 与 call-start 不同，且两者之间没有其他事件；验证通信区间不计为空闲。
6. 用已有原始 trace 重算并核对归因；保留历史报告或输出带版本的新报告，禁止静默改写旧结论。

这一关只要求重算已有数据，不要求先重跑全部历史实验。makespan 与归因统计分别验证。

## 3. 新增 workload

继续使用线性 job、真实 all-reduce、独立 job ProcessGroup 和现有 manifest schema。计算仍用 sleep 表示，因此结果只说明该 replay 的通信与推进行为，不代表真实 GPU 计算竞争。

下面参数为初始设计。所有 producer/consumer 时长单位均为 ms；通信大小为每项 all-reduce 的 tensor 大小。先检查 profile 与资源预算，再冻结正式输入。不得根据某个策略的获胜情况调整参数。

| 家族 / 建议名称 | 初始配置 | 主要问题 |
| --- | --- | --- |
| 通信密集 `comm-heavy` | 3 个 job，各 4 项 16 MiB；每项 producer 0.5，consumer 0 | 更长通信能否摊薄控制成本？单在途是否仍有代价？ |
| 计算错峰 `staggered-compute` | 3 个 job，各 4 项 1 MiB；producer 分别为 `[1,12,2,12]`、`[6,2,12,2]`、`[12,6,2,6]`；consumer 0 | 静态顺序是否反复阻挡已有 ready 工作？在线选择是否填补空档？ |
| 长短链 `long-short-chains` | job-0 为 8 项、其余 3 个 job 各 2 项，均为 1 MiB；producer 1，consumer 1 | 长链优先对整体结束时间和短 job 完成时间有何取舍？ |
| 大小消息 `mixed-message-sizes` | 3 个 job，各 4 项；一个 job 用 16 MiB、另两个用 1 MiB；producer 1，consumer 0 | 大消息位置如何影响小消息 job？增加一个大 job 位于 manifest 末尾的对应变体，检查 FIFO 固定顺序偏置。 |
| 重叠窗口 `overlap-window` | 2 个 job，各 4 项 1 MiB；producer 1；分别生成短、中、长 consumer 窗口三个变体 | 独立计算能隐藏多少等待与通信？修正后的关键路径评分是否与观测相符？ |

重叠窗口三个变体使用同一份预先测得的通信 profile：设 1 MiB 的无竞争 service p50 为 T，consumer 分别取 `0`、`T`、`3T`。将所得具体秒数写入不可变 workload 文件，并记录 profile digest；运行时不随策略或实际时长更新。

共 5 个家族、8 个初始 manifest。避免全参数笛卡尔积：先验证上述有限配置，再决定是否需要额外变体。

输入约束：

- 每个变体在各策略间保持相同的消息大小、计算时长、成员、profile 和消费语义。
- 错峰由相对计算时长产生，不固定每项通信的绝对 ready 时刻；调度改变后续 ready 属于真实反馈。
- 消息大小来自显式配置，不按耗时反推字节数。
- 保持每个 task 独立 tensor，将数值校验留在应用结束和通信排空之后。
- 预估全部预分配 tensor、校验临时 tensor 与进程内存。资源不足时在正式批次前统一缩减并记录，不能只为某个策略改变输入。
- 多 job 场景要检查图表的 job 标签、颜色和偏移是否仍正确，不能沿用只支持 job-0/job-1 的假设。

## 4. 对照矩阵与控制变量

沿用以下六个场景：

| 场景 | 选择方式 | 在途约束 |
| --- | --- | --- |
| `phase1_bare` | 裸发，到达后调用 collective | 不限 |
| `ready_first_unbounded` | 跨 rank 协调的 ready-first | 不限 |
| `ready_first_serial` | 跨 rank 协调的 ready-first | 当前实现的全局完成屏障 |
| `fifo_unbounded` | 静态 FIFO | 不限 |
| `fifo_serial` | 静态 FIFO | rank-local 单在途 |
| `ltf_serial` | 修正评分的静态 LTF | rank-local 单在途 |

固定 CPU/Gloo、world size 2、相同参与成员和当前软件环境，不安装或升级通信依赖。GPU/NCCL 留作独立阶段。

每个 workload 的六场景使用同一 profile。维持已有 3 次 warmup、10 次正式 profile 测量，保存原始样本和测量边界。重叠窗口变体必须复用用于生成窗口的 profile。

主批次 completion poll interval 固定为 1 ms，与当前修正批次一致。对于通信较短或差异接近该尺度的场景，另做 0.1 ms 敏感性批次；所有被比较策略一起改变 interval，不能混合两种 interval 的样本。额外消耗和系统干扰也属于敏感性结果的一部分。

ready-first 与静态路径的控制协议及完成屏障不同，差值只能作为系统级对照，不称为纯策略收益。未记录为统一执行路径前，不将 `fifo_serial - ready_first_serial` 全部归因于队首阻塞。

## 5. 分阶段运行

### A. 准备与静态检查

- 完成第 2 节归因修复和回归检查。
- 构造并校验 8 份 manifest，保存参数说明、profile 和 digest。
- 验证 task ID、job 内顺序、成员覆盖和预分配内存预算。
- 输出 FIFO/LTF 的具体 Plan 与 LTF 候选评分；相同 Plan 标注为没有排序差异的对照。
- 为 runner 增加最小的 workload 选择能力（例如可重复的 `--workload`），同时确保新建批次和 finalize 都以 manifest 定义的样本集为准，不由当前 workload 目录重新推断数量。该选项是待实现能力，不是现有可直接运行命令。

### B. Smoke：每种 2 次

8 个 workload × 6 场景 × 2 次，共 96 次 replay，按家族分批运行。复用预先冻结的 profile，不与探索批次混合统计。

每个家族完成后检查：结果正确、完成与校验边界闭合、顺序合法、没有超时；预期的错峰、消息差异或重叠窗口确实出现在 trace。没有产生预期机制的输入应记录原因，必要时生成新版本重新 smoke，保留旧输入和结果。

先记录单次及整个家族耗时，评估后续总预算。不得静默扩展 job 数、消息大小、重复次数或实验时长。

### C. 探索：每种 10 次

对 smoke 通过且有明确研究问题的配置运行六场景各 10 次。全部 8 个配置都进入时为 480 次 replay；可按家族分批完成，不一次性启动全部矩阵。

配置的纳入标准必须是功能通过和机制可观测，不是某策略是否获胜。所有 smoke 配置都要记录最终纳入、调整或未继续的原因。

同一 repetition 内六场景用固定 seed 打乱，串行执行各 replay，避免实验之间争抢资源。保留相同 repetition 的配对差值和比值；配对不表示各次操作系统噪声相同。

### D. 有针对性的复核

先分析 10 次探索结果，再选择需要解释的场景补充轮询敏感性或新 seed 批次。只有需要判断趋势稳定性时才考虑 30–50 次重复，另行评估运行预算。不得事后只保留有利结果。

## 6. 指标与分析问题

每个场景至少报告：

- workload application makespan、每个 job 从统一释放起的完成时间。
- communication drain、validation 和 harness 时长，分别报告，不混入应用时间。
- ready-to-admit、应用等待绑定、底层 wait、collective 调用和调用到完成探测的区间。
- 本地队首空闲、待发射、容量占用及无法归因区间；协调耗时单独列出，说明是否与其他指标重叠，禁止直接相加成总耗时。
- 原始点、中位数、P10–P90、最小/最大值和样本数；成对 makespan/job 完成时间差值与比值。
- 各 rank 实际提交顺序与成员覆盖，观测通信区间重叠情况。区间重叠不独立证明设备实际并行。

按家族回答：

1. 通信规模增大后，ready-first 的额外成本占比是否下降？
2. 错峰场景的本地队首空档是否减少，减少量是否足以抵消协调成本？
3. 长短链中，整体 makespan 是否掩盖短 job 延迟？LTF 是否只提前长 job 而未改变最终结束时间？
4. 大消息位置对结果有多大影响？优势是否主要来自 FIFO 的固定 manifest 顺序？
5. consumer overlap 改变后，静态关键路径估计与实际 ready-to-job-end 的差距如何变化？

零假设或负结果也应报告。10 次样本用于探索，P10–P90 不是收益置信区间；CPU/Gloo 与 sleep 计算结果不外推为真实训练或 GPU/NCCL 性能。

## 7. 输出与验收

沿用 `benchmark/phase1.2/result/batches/<batch-id>/`：manifest、profiles、raw、summary、analysis 和 figures。

manifest 记录输入及 profile digest、执行命令、交错 seed、轮询间隔、运行顺序、完整性和文件哈希。当前工作区有未提交修改，除 git commit/dirty 标记外，应保存本次相关源码的哈希或可还原快照，确保同一 commit 下不同实现可区分。

验收要求：

- 所有正式样本通过 tensor 校验、成员覆盖、适用的 Plan/group 顺序和串行检查。
- 没有缺失或逆序的必填时间戳；completion observation 不晚于 drain，validation 不早于 drain。
- summary、图表和报告只使用对应 manifest 列出的样本，失败或不完整批次不作为正式比较。
- 等待状态覆盖相应时间范围且没有错误重叠；未知原因明确保留，不强行归入 scheduler delay。
- 代表 timeline 按预先定义的中位数邻近规则选择并记录路径，各 rank 分开绘制。
- 分析明确区分相同 Plan 的运行波动、不同 Plan 的系统差异，以及全局协调与本地串行的语义差别。
- 新增结果不覆盖历史原始数据；参数调整、统计勘误与未验证边界均有记录。

## 8. 非目标

本轮不实现 DAG runtime、不接入真实训练框架、不修改调度算法以迎合 workload、不开展 GPU/NCCL 或多机实验，也不把全部控制开销优化作为扩展实验的前置要求。先通过有限场景确定值得继续研究和优化的方向。
