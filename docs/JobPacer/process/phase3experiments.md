# JobPacer Phase 3.1 / 3.2 CPU 实验实施方案

日期：2026-09-22。状态：已完成最小实验支撑、CPU/Gloo pilot 及一次完成反馈延迟修正复测；正式多 seed 性能验收仍未完成。

本文件规定实验准备、执行、分析和产物要求。阶段目标沿用
[原始实验计划](../260914项目计划讨论/JobPacer实验计划.md)、
[Phase 3.1 计划](../plan/phase3.1.md) 和 [Phase 3.2 计划](../plan/phase3.2.md)，
不改变 runtime 的协议和阶段范围。已验证事实分别见
[Phase 2 结果](../result/phase2.md)、[Phase 3.1 结果](../result/phase3.1.md)、
[Phase 3.2 结果](../result/phase3.2.md) 和 [最近结构回归](../result/phase3.12fix.md)。
历史通过记录不代表本方案已执行，也不证明策略性能收益。

> 目录迁移说明：本文中的脚本名和命令记录实验当时的入口。当前 Phase 3 replay CLI 为
> `python -m examples.jobpacer.runtime.replay_launcher`，批次与 compact suite 分别为
> `python -m examples.jobpacer.experiments.gloo_phase3_batch` 和
> `python -m examples.jobpacer.experiments.compact_suite`；原 `scripts/run_phase3.py`、
> `scripts/run_experiments.py`、`scripts/run_compact_suite.py` 薄 wrapper 已于 2026-09-30 删除。
> 历史命令仅用于说明既有批次，不改写历史产物。

> 目录索引补充（2026-09-24）：旧 smoke 输入现位于
> `benchmark/phase3/experiments/dag-semantics/smoke/`；结果按场景写入
> `benchmark/phase3/results/<category>/<scenario>/<batch-id>/`，suite 汇总放在
> `results/suites/<suite-id>/`。具体映射见
> [`benchmark/phase3/results/migration-map.json`](../../../benchmark/phase3/results/migration-map.json)，当前命令见
> [`benchmark/phase3/experiments/suites/README.md`](../../../benchmark/phase3/experiments/suites/README.md)。

## 1. 研究问题与范围

本轮主要回答：

1. 在线 ready 时间偏离离线估计时，动态选择能否减少静态队首阻塞？
2. 多个通信同时 eligible 时，LTF 是否改善 workload makespan，代价是否体现为某些 job 的完成时间变差？
3. 有界主动等待在什么到达误差和通信尺度下有收益，什么时候不如立即服务？
4. 在分叉、汇合、多通信前沿的 DAG 中，上述结论是否仍成立？
5. 相对 Phase 1 裸发和 Phase 2 旧 scheduler，新 runtime 的整体收益能否覆盖控制与推进开销？

主实验固定：单机、2 rank、CPU/Gloo、float32 sum all-reduce、每 job 一个串行计算通道、
sleep 模拟计算。新 runtime 始终使用全局 `max_inflight=1`。
先做 2 job，选定场景再扩展到 4 job。不同实验进程批次串行运行。

本轮不扩展 GPU/NCCL、多 host、多资源、同 group 动态重排、多通信在途、真实训练框架。
CPU sleep 的作用是控制依赖和可重叠窗口；它不模拟实际 CPU 算力、内存带宽争抢。
Gloo 的性能结论限定于记录的机器和 backend，不能直接外推 PCIe/NCCL。

不预设动态优于静态，也不预设串行优于裸并发。性能持平、负收益和收益边界均是有效结果。

## 2. 比较组与归因边界

### 2.1 线性 workload 的共同对照组

| ID | 实验组 | 当前入口/配置 | 回答的问题 |
| --- | --- | --- | --- |
| B0 | Phase 1 bare | `run_phase1.py` | 无协调真实通信的整体基线 |
| B1 | Phase 2 静态 FIFO | `run_replay.py --mode scheduler --selection runtime_arrival --policy fifo --max-outstanding 1` | 旧静态系统表现 |
| B2 | Phase 2 静态 LTF | 同上，`--policy ltf` | 旧静态优先级表现 |
| S0 | 新 Static FIFO | `run_runtime_replay.py --policy static_fifo` | 同 runtime 的固定轮转基线 |
| S1 | 新 Static LTF | `--policy static_ltf` | 同 runtime 的静态优先级基线 |
| D0 | Dynamic FIFO | `--policy fifo` | 利用实际 eligible 到达次序 |
| D1 | Dynamic LTF | `--policy ltf` | 在 eligible 候选间按 `estimated_comm_s + remaining_tail_s` 选择 |
| D2 | Bounded Lookahead | `--policy lookahead` | 有界等待相对立即 LTF 的增益/损失 |

静态 FIFO 是预定轮转序列；Dynamic FIFO 是首次进入 eligible 的次序，名称相近但语义不同。
旧路径的 `ready_first` 是额外选择模式，不混入 B1/B2；如后续研究它，应增加独立组名。

比较规则：

- B0 对 B1/B2：报告串行约束、顺序和旧实现开销的综合变化。
- B1 对 S0、B2 对 S1：先检查实际静态任务序列相同，再报告迁移的综合变化。
- S0/S1 对 D0/D1/D2：作为在线策略收益的主要证据，协议、执行和完成反馈路径相同。
- D0 对 D1：解释优先级选择；D1 对 D2：解释主动等待。
- B0 对新 runtime：回答系统整体是否有用，不把差值全部归因于策略。

旧 scheduler 的 `max_outstanding=1` 是本地容量约束，不能仅凭该参数宣称与新 runtime
“全部成员物理完成后才释放容量”的全局语义相同。保存每 rank occupancy、launch 和完成记录，
披露语义差异；不为使数字相同而修改历史基线。策略因果结论以 S/D 组比较为准。

### 2.2 DAG 对照组

一般 DAG 主比较使用 S0、S1、D0、D1、D2，均运行同一 DAG runner 和同一图。
Phase 3.2 是输入与推进模型的扩展，不能把另一张线性图的耗时作为它的直接性能分母。

跨阶段有两个层次：

1. 线性共同子集运行全部 8 组，提供 Phase 1/2/3 的桥接结果。
2. 一般 DAG 运行 5 个新 runtime 策略，回答 DAG 上的策略问题。

若需宣称“DAG 调度优于无调度”，必须额外实现共享 DAG runner、相同完成和消费语义的 bare
入口，并独立验证跨成员 group 顺序。本文初次盘点时 CLI 尚无 DAG bare；后续已增加
`BareDagAdapter` 并在双卡 L1/D1/D3 上诊断，但 L1 观察到 rank 间全局 launch 序列分歧，当前仍不回答
bare 安全性/性能对照。不能把 DAG 展平成串行链、改变 group 或删掉依赖后充当 bare 基线；也不能将
raw-ordered 的受控全序参考称作原始 bare。最新执行结果见[七臂 preflight 报告](../result/phase3-gpu-seven-arm-preflight-execution-20260928.md)。

## 3. 当前能力与正式实验前的准备

以下基于当前代码检查，不表示已经实施本节改动。

| 项目 | 当前状态 | 本轮处理 |
| --- | --- | --- |
| 新 runtime 五种策略 | CLI 已支持 | 复用并保存实际选择序列 |
| 线性实际计算扰动 | 已支持按 seed/epoch/job/task/rank/segment 固定采样 | 结果中保存 producer/consumer 样本；保持策略估计冻结 |
| DAG 计算扰动 | `--compute-jitter`，按 seed/epoch/job/node/rank 固定样本 | 可复用；输出实际样本摘要 |
| 离线通信 profile | 旧、新 replay 均支持严格 `--comm-profile` | 已将同一 profile 注入线性新旧组，记录 digest、环境和来源 |
| DAG 静态顺序 | 自动 FIFO/LTF，或 `--static-order` 文件 | 保存生成结果，跨 seed 冻结 |
| DAG profile | 已支持与线性路径相同的严格签名覆盖 | 正式 DAG 输入不解释 manifest 占位估值 |
| Lookahead 等待预算 | 核心默认 0.02 s，replay 已暴露 `--wait-budget-s` | 首轮固定并记录；预算扫描仍未执行 |
| 计时口径 | 旧路径有 application/drain/validation 边界；新路径此前尚不等价 | 已补统一应用释放、drain、validation 边界；正式比较前仍需重跑共同矩阵 |
| 指标与 trace | 有 rank 事件、coordinator records、DAG 事件 | 已补串行批量汇总、原始结果和失败记录 |
| 一般 DAG bare | 未提供 | 可选，不阻塞 DAG 内部策略比较 |

### 3.1 最小准备改动

仅在 workload/adapter/harness/结果汇总层补实验能力，不重写 coordinator，不建立通用实验平台：

1. 分开保存名义计算估计与实际执行时长；线性旧/新路径读取同一份执行样本。
2. profile 以 op、bytes、dtype、group size、backend、device、reduction 匹配；
   对所有正式签名严格覆盖，缺项拒绝运行，不静默回退到手工估值。
3. 线性 LTF 新旧比较使用相同 tail 定义；DAG 使用当前关键路径定义，另存定义版本。
4. 统一计时、预热和 tensor 校验位置。新线性路径当前逐通信验证 tensor，会影响后续 ready，
   因此不能只在汇总时从总时间扣除验证耗时。应后置校验并保持所需 tensor 存活。
5. 批处理只负责生成配置、顺序运行现有入口、保存原始结果和汇总，不新增调度层。

新增功能必须有针对性检查：样本一致且不泄漏进 hint、计时边界、预期任务全集、
profile 缺项拒绝、静态顺序冻结、失败结果不计为成功性能样本。涉及实际 collective 的行为要做 Gloo 集成。

## 4. 环境、预热与尺度校准

### 4.1 环境固定

记录 CPU 型号、物理/逻辑核数、NUMA、内存、操作系统、Python/PyTorch 版本、Gloo 可用性、
线程环境变量、CPU affinity、代码快照及后台负载。不要擅自安装或升级软件。

同一批次使用相同 CPU 分配和线程设置；为 rank、控制线程和 Gloo 留足 CPU，
不要让一个策略独占更多核心。若设置 `OMP_NUM_THREADS`、`MKL_NUM_THREADS` 或 affinity，
所有组一致并写入 manifest。不能把整个多线程系统绑定到一个核后将拥塞解释为通信瓶颈。

### 4.2 通信校准

候选消息大小先取 4 KiB、1 MiB、16 MiB；必要时增加一个较大尺寸，但先检查内存预算。
每个签名建议预热 5 次、测量 30 次，记录各参与 rank 自己测得的 duration，再按轮取成员最大值。
保存 p10/p50/p90，策略估计使用冻结的 p50；重新校准必须生成新 profile ID。

另测两个独立 group 上相同通信集合的顺序与裸并发耗时，用于判断本机串行化代价。
控制 rendezvous 不得使用正被调度的 job collective。当前 profile 工具主要测无竞争签名，
并发校准需独立场景，不能把 profile p50 直接当作并发服务时间。

正式选出两个尺度：

- 小消息：暴露控制面、轮询和 Python 开销。
- 较长通信：实测通信耗时足以与控制延迟区分，用于调度机理比较。

不得通过增大 `estimated_comm_s` 伪造真实通信变长。记 `T` 为选定签名的实测 p50，
计算/通信比例采用 `C/T ∈ {0.25, 1, 4}`。先在一个尺度、一个比例上完成主矩阵，其他值为敏感性实验。

### 4.3 每次 replay 的预热

在应用计时前完成 group 创建、代表签名预热与 drain，并通过独立控制同步释放应用。
固定预热次数，warmup 任务不进入被测静态序列、任务全集或 JCT。
单独启动过一次程序不等于本次新建进程/group 已预热。

完成探测默认 1 ms，DAG 推进默认也为 1 ms。首轮固定默认值；再对代表场景扫描
0.2、1、5 ms，一次仅改变一个轮询参数，并记录 CPU 开销。旧/新相应完成探测配置应一致，
但相同间隔不代表相同探测实现。若延迟噪声淹没通信时间，报告该限制并调整消息尺度。

## 5. Phase 3.1 线性场景

所有主场景使用相同成员 `[0, 1]`，每 job 一个独立 ProcessGroup。先取每 job 8 项通信，
对称场景可缩为 4 项验证机制；正式时长由 pilot 决定，不把任务数当成固定验收条件。

| 场景 | 构造方式 | 主要比较与证据 |
| --- | --- | --- |
| L0 稳定均衡 | 相同消息、相近计算窗口、无随机扰动 | 全 8 组；动态开销，裸并发与串行差异 |
| L1 队首错位 | 静态首项 A 实际 producer 延后，B 已 ready；名义估计不变 | S0/S1 对 D0/D1；队首阻塞和 makespan |
| L2 tail 不对称 | 两 job 通信后有效剩余计算约为 `T` 与 `4T`；候选在容量占用期间积累 | D0 对 D1；候选集合、选择、每 job JCT |
| L3 可预测的未来重要任务 | 当前有短 tail 候选，长 tail 目标预计短时间后 ready | D1 对 D2；实际主动等待、到达、完成时间变化 |
| L4 预测失准 | 沿用 L3 估计，目标实际早到、按时或越过等待 deadline 才到 | D1 对 D2；回退、等待浪费和负收益 |
| L5 成员偏斜 | 某个 rank 的指定 producer 额外延迟，其他 rank 不变 | OFFER 到齐与 eligible 的差别，不能按 rank 0 ready 解释选择 |

L1 可先以延迟 `2T、5T` pilot；L3/L4 的预计到达间隔必须落在预算内，且满足实际 Lookahead
评分条件，不能仅凭“目标 tail 更长”认定应该等待。核心默认预算 20 ms 是等待上限，
不是每次固定等待 20 ms；应保存每轮实际 deadline 和目标。

L2 中两个候选必须在某次决策时同时 eligible。用前一项真实通信占用期间的计算窗口错位构造，
并从 coordinator trace 验证。若实际没有候选竞争，该运行仍保留，但标记为未触发优先级区分，
不能据此宣称 LTF 与 FIFO 等价。确定性策略分支另用事件驱动状态机测试，不靠反复跑直到出现想要顺序。

### 5.1 随机扰动

确定性机制确认后，在 L0/L1/L2 中选代表点加入：

`actual_duration = nominal_duration × (1 + a × u)`，其中 `u ∈ [-1, 1]`，`a ∈ {0, 0.1, 0.3, 0.5}`。

采样键固定为 `(seed, epoch, job_id, task_or_node_id, rank, compute_segment)`；
线性 producer 和 overlap compute 必须区分 segment。DAG 现有节点级采样可以继续使用，
跨表示比较时通过显式节点映射共享样本，不假设相同 seed 自动产生相同执行输入。

主扰动组可采用跨 rank 独立样本，额外用全成员相关延迟模拟共同 straggler；两者分开汇总。
实际样本只供执行器使用，policy 和静态顺序生成器只能访问冻结估计及合法在线状态。
同 seed 内所有策略共享样本；后续节点的绝对 ready 时刻由真实依赖推进产生。

## 6. Phase 3.2 DAG 场景

现有 `benchmark/phase3/{linear,diamond,multi-group}.json` 为机制样例，保留原文件，
正式参数化输入另存；不能把现有小样例的一次顺序或耗时当作稳定性能结论。

| 场景 | 图与控制条件 | 观察重点 |
| --- | --- | --- |
| G0 线性等价链 | compute → comm → compute → comm；先令旧模型 overlap compute 为 0 | 跨执行器桥接，节点推进与额外开销 |
| G1 diamond | producer 分叉为 comm 与独立 compute，join 等两者完成 | 通信延迟是否被计算隐藏、join 是否成为瓶颈 |
| G2 不对称多前沿 | 相同 job 的不同既定 group 同时产生候选，后继路径明显不同 | DAG LTF 是否服务有效关键分支 |
| G3 group 顺序受限 | 某分支后项先 ready，但受同 group 前序约束 | DAG-ready 与 eligible 差距、协议约束下的收益边界 |
| G4 多前驱预测 | 目标依赖已启动计算；另一组依赖尚未完成通信或未启动计算 | 安全预测前沿、等待/回退及不应预测的情况 |

G1 独立计算时长先取 `0.25T、T、4T`，展示通信位于可隐藏窗口内外的差别。
G2 长短分支先取约 4:1，再用近似对称分支作负对照。
G3 保持图与 group 联合约束无环；非法图单独作为启动前拒绝用例，不进入性能统计。

每 job 的计算节点仍按固定规则在一个计算通道执行，不把改变计算选择顺序算作通信策略收益。
LTF 的 DAG tail 为通信完成之后最长估计后继路径，排除当前通信自身时长；并行分支取 max。
该值不是精确剩余 JCT：汇合另一分支、计算排队和跨 job 竞争均可能改变实际关键路径。

线性 overlap 语义是 submit 返回后进行独立计算，再等待消费。迁移到 DAG 时应表达为分叉与汇合，
不能添加 `comm 完成 → independent compute` 边。即使分叉图正确，是否保证 submit-before-compute
也要单独核对。因此 G0 先用无 overlap 的共同子集；带 overlap 的迁移结果注明执行顺序差异。

## 7. 指标与时间语义

### 7.1 应用指标

每 rank 使用自己的单调时钟记录：应用释放 `release_r`、每 job 全部应用节点/消费完成
`done_jr`、通信 drain 完成、结果校验完成、harness 退出。启动、建组、预热、验证不计入应用时间。

- `JCT_j = max_r(done_jr - release_r)`，取参与 job 的成员；本轮所有 job 同时释放。
- `makespan = max_r(max_j(done_jr) - release_r)`，先在本地求差再跨 rank 取最大。
- 平均 JCT 与最慢 job JCT 均报告，避免只看总完成时间隐藏受损 job。
- `slowdown_j = JCT_j(shared) / JCT_j(isolated)`。isolated 使用同后端、同执行模式、同样本和计算语义；
  明确分母是模式匹配的单 job 结果，不能混用不同 runner 的 isolated 时间。
- 加速比定义为 `baseline_time / candidate_time`，大于 1 表示改善。

成员 release 通过独立控制同步尽量对齐；该 makespan 是 rank-local duration 最大值的操作性定义，
不宣称是通过严格时钟同步获得的全局物理跨度。记录释放机制及可测启动偏斜。

现有 `coordinator_epoch_duration_s` 包含中央协议活动（包括注册），仅用于协议诊断，
不得作为上述 makespan 的替代。job 自己的线程开始时间也不替代共同 release，否则会隐藏启动排队。

### 7.2 原因指标

| 指标 | 计时域 | 解释 |
| --- | --- | --- |
| eligible → grant | coordinator | 候选等待，包含容量和策略作用 |
| grant received → launch | rank-local | 本地控制/执行排队 |
| offer → grant received | rank-local | 含控制传输、其他成员等待及准入，不是纯 policy 延迟 |
| launch → completion observed | rank-local | backend 执行及探测延迟的合计 |
| consumer wait | rank-local | 未被独立计算隐藏的应用等待 |
| DAG ready → submit | rank-local | DAG 推进和绑定成本 |
| compute ready → started | rank-local | 计算通道排队 |
| predicted ready error | rank-local，同一预测锚点 | Lookahead 预测偏差 |
| 候选数、选择、目标与 deadline | coordinator | 检查机制是否真实触发 |

将中央空闲按无 eligible、静态队首阻塞、主动等待拆分；容量已占用单列。
若统计分解要求相加等于总窗口，应先定义互斥状态；不得将重叠原因重复累计。
通信占用比例只表示测得区间内有在途任务，不称为链路带宽利用率。
完成探测时刻是物理完成的观测上界；没有独立完成时间来源时，不单独声称测出了精确探测延迟。

DAG join 等待用最后一个前驱完成及后继启动解释。可画代表性 rank 的计算/通信时间线；
不同 rank 分面显示或用各自 release 作零点，不将未同步原始时间戳拼接后计算差值。

## 8. 运行批次与统计设计

### 8.1 分批推进

| 批次 | 内容 | 进入下一批条件 |
| --- | --- | --- |
| A | 环境记录、针对性测试、现有样例 smoke | 正确性和有界失败无未解释回归 |
| B | profile、裸并发/串行校准、单 job 基线 | 确定消息尺度和开销量级 |
| C | L0–L5 的确定性机制 pilot | 从 trace 确认队首阻塞、候选竞争、等待/回退等 |
| D | 选定线性场景，多 seed，全 8 组 | 统一输入与计时已完成，配对结果完整 |
| E | G0–G4，5 个新 runtime 策略 | 图语义、group 约束与预期机制可解释 |
| F | 少量敏感性点：job 数、消息、比例、轮询、预算 | 主结果足够稳定且仍有具体待回答问题 |

第一轮机制 pilot 每配置重复 5 次；随机 screening 先取 10 个 seed，每 seed 运行所有组。
例如选 4 个线性点与 3 个 DAG 点，各取 10 seed，基础为 `4×10×8 + 3×10×5 = 470` 次 replay，
尚不含 profile、isolated、重复和敏感性实验。先测一次 wall time 并估算总耗时，
不要在方案阶段直接启动这个批次，也不要展开所有参数的笛卡尔积。

正式主结果建议使用独立于 pilot 的 30 个预先固定 seed，每 seed 重复 3 次；
若预算有限，先缩减场景数量，保留配对结构，并披露样本限制。
无扰动场景的重复用于测系统噪声，不把不同无效 seed 当成新的 workload 样本。

### 8.2 配对与不确定性

同一 `(场景, 参数, seed, repeat)` 是一个 block；其中随机化策略执行顺序并保存随机顺序表。
所有运行串行执行；各组失败也记录，不能补跑直到出现成功且只保留成功样本。

先对同 seed 的重复取中位数，再计算策略配对差值/加速比；报告跨 seed 中位数、p10/p90、
胜/平/负比例，以及按 seed block 重采样的 95% bootstrap 区间。预先规定持平阈值，
例如结合 pilot 噪声设定，而非看完结果后调整。重复运行和同一 replay 中的任务不作为独立 seed。

超时、顺序/数值错误单列失败率。配对性能仅用双方有效的 block，同时报告缺失数量；
若某策略系统性失败，不给出忽略失败后的总体优势结论。环境故障保留原始记录并关联重跑原因。
小样本不报告稳定 P99；有限批作业吞吐若仅为 job 数/makespan，不作为独立研究发现。

## 9. 正确性与机制验收

每次成功性能样本必须同时满足：

1. 预期 job、task、DAG node 的集合和数量完整，无重复、遗漏或非预期项。
2. 每个参与成员的 all-reduce tensor 正确。
3. 新 runtime 的实际 launch 是共同 grant 在该 rank 的完整投影；每 group 成员序列匹配。
4. 新 runtime 容量从 grant 到全部成员 COMPLETED 释放，SUBMITTED 先于 COMPLETED。
5. DAG 完成依赖成立，join 不提前推进；同 group 后项不能越过前序。
6. 静态顺序没有跳过未 eligible 队首，实际序列与冻结文件/生成结果一致。
7. 结束消息与 drain 成功，无残留 worker，无未完成任务被当作成功。

bare 不要求不同独立 group 之间存在共同全局顺序；仍检查各 group 成员覆盖、序列及 tensor。
旧 scheduler 使用其自身顺序和容量验收，不伪装成已验证新 runtime 的全局完成协议。

性能矩阵之外单独检查 metadata mismatch、missing task、launch/probe failure、compute/binding failure
和非法 DAG。确认失败停发、唤醒等待者、有界退出。失败注入不参与正常 JCT 比较。
真实 Gloo socket 被沙箱拒绝时按权限流程处理，不改代码绕过、不记录为协议回归。

## 10. 当前可执行命令与待实现边界

以下命令从仓库根目录运行，作用是准备和 smoke；它们不是已完成的性能测量记录。
结果先写 `/tmp`，正式批次由批处理为每次运行分配唯一输出路径。

```bash
# 代码验收；记录实际通过、失败、跳过及耗时
PYTHONPATH=src pytest -q tests/unit/runtime
PYTHONPATH=src pytest -q
env PYTHONPATH=src RUN_JOBPACER_RUNTIME_REPLAY=1 \
  pytest -q tests/integration/test_runtime_replay.py

# 原路径通信 profile：当前 built-in balanced 主要用于流程检查
PYTHONPATH=src:. python -m examples.jobpacer.scripts.run_comm_profile \
  --workload balanced --backend gloo --world-size 2 \
  --warmup 5 --iterations 30 --timeout 60 \
  --output /tmp/jobpacer-phase3-profile.json

# Phase 1 / Phase 2 smoke
PYTHONPATH=src:. python -m examples.jobpacer.scripts.run_phase1 \
  --workload balanced --backend gloo --world-size 2 \
  --output /tmp/jobpacer-phase3-bare.json

PYTHONPATH=src:. python -m examples.jobpacer.scripts.run_phase2 \
  --mode scheduler --selection runtime_arrival --policy fifo \
  --workload balanced --backend gloo --world-size 2 --max-outstanding 1 \
  --completion-poll-interval-s 0.001 \
  --output /tmp/jobpacer-phase3-old-static.json

# 新线性路径；替换 policy 为 static_ltf/fifo/ltf/lookahead 覆盖其他组
PYTHONPATH=src:. python -m examples.jobpacer.runtime.replay_launcher \
  --policy static_fifo --workload balanced --backend gloo --world-size 2 \
  --poll-interval 0.001 --timeout 20 \
  --output /tmp/jobpacer-phase3-new-static.json

# DAG 路径及已有扰动功能
PYTHONPATH=src:. python -m examples.jobpacer.runtime.replay_launcher \
  --policy ltf --dag benchmark/phase3/multi-group.json \
  --backend gloo --world-size 2 --epoch 0 --compute-jitter 0.3 \
  --comm-profile /tmp/jobpacer-phase3-profile.json \
  --poll-interval 0.001 --dag-poll-interval 0.001 --timeout 20 \
  --output /tmp/jobpacer-phase3-dag-ltf.json

# 线性 8 组（含旧 bare/旧 FIFO/LTF 与新 runtime 五策略）
PYTHONPATH=src:. python -m examples.jobpacer.experiments.gloo_phase3_batch \
  --output-dir /tmp/jobpacer-phase3-batch-final-20260922 \
  --workload balanced --backend gloo --world-size 2 --seeds 0 --repeats 1 \
  --compute-jitter 0.3 --wait-budget-s 0.02 --comm-profile /tmp/jobpacer-phase3-profile.json \
  --include-old

git diff --check
```

DAG seed 来自 JSON；CLI 的 `epoch` 也参与现有采样，所有策略必须相同。批处理入口已自动
保存 profile/config、原始结果和汇总；它仍不提供一般 DAG bare，也不把单次运行升级为统计结论。

## 11. 产物组织与报告

建议新增正式输入目录 `benchmark/phase3/experiments/`，保存名义 workload、静态序列与参数说明；
保持现有三个机制样例不变。完整原始产物保存到容量允许的实验输出目录，
结果文档记录该目录及摘要；是否纳入版本控制按产物大小决定，不自动提交大量 trace。

每个 batch 至少包含：

- `manifest.json`：batch ID、环境、Git HEAD、dirty 状态、相关源码逐文件摘要、输入/profile/sample 摘要。
- `runs.jsonl`：每次完整命令、配置、seed/repeat、策略运行顺序、开始结束时间、wall time、退出码和结果路径。
- `inputs/`：冻结估计、实际样本、group、静态顺序和节点映射；策略可见部分与执行部分分开。
- `raw/`：原始 JSON、stdout/stderr、验证结论与失败原因。
- `summary.csv`：逐 run 指标；`jobs.csv`：逐 job JCT、isolated 分母和 slowdown。
- `paired-summary.csv` / `analysis.json`：seed 内 repeat 中位数后的配对差值、
  加速比、胜平负和 seed-block bootstrap 区间。
- `mechanisms.csv`：候选竞争、静态队首阻塞、Lookahead 等待/回退和成员 OFFER 偏斜证据。
- `figures/`：完成时间对照、扰动响应、阻塞/等待分解和代表性 trace。

最终报告建议至少给出：

1. 线性共同子集的 8 组 JCT/makespan 对照，以及旧/新协议差异说明。
2. 新 runtime 静态/动态对扰动幅度的响应曲线和配对不确定性。
3. FIFO/LTF/Lookahead 的代表 trace，包含负收益场景，说明收益来源。
4. 同 DAG 五策略对照，以及 group 约束和计算重叠导致的收益边界。
5. 控制/轮询开销的消息尺度敏感性，失败率和所有未验收范围。

实施变更记录追加在本文件或关联 process 文档；已执行命令、机器环境、实际结果与产物链接
写入新的 `docs/JobPacer/result/phase3experiments.md`。未执行时不创建“已验收”结论。
实验完成的标准是对上述研究问题给出有证据的回答，不要求任何策略取得正收益。

## 12. 本轮实施记录（2026-09-22）

已按本方案补齐并验证最小实验链路：

1. `examples/jobpacer/scripts/run_phase3.py` 新增严格通信 profile 注入、固定
   `--wait-budget-s` 和线性 `--compute-jitter`/`--epoch` 透传；旧 replay 使用同一线性样本键。
2. producer/consumer 实际时长按 `(workload seed, epoch, job, communication, rank, segment)`
   固定采样，hint 仍使用名义估计；结果保存实际样本、profile digest 和估值来源。
3. 新增 `examples/jobpacer/scripts/run_experiments.py`，串行保存 `manifest.json`、`runs.jsonl`、
   `raw/`、`summary.csv`；支持线性 8 组和同一现有 DAG 的五策略批次。
4. 针对性检查为 `41 passed`，最终全仓为 `170 passed, 35 skipped`，真实双 rank 集成为
   `30 passed`；真实 Gloo replay 产物见
   [Phase 3 实验结果](../result/phase3experiments.md)。

实际执行的 pilot 目录为 `/tmp/jobpacer-phase3-batch-final-20260922/` 以及三个
`/tmp/jobpacer-phase3-dag-*-20260922/`。本轮每组仅一个 epoch/repeat，未将单次排序写成
稳定收益；profile 仅覆盖 4 KiB，GPU/NCCL 和正式统计边界仍按第 11 节列为未验收。

## 13. 完成反馈延迟定位与修正（2026-09-22）

### 13.1 定位过程

对 `/tmp/jobpacer-phase3-batch-final-20260922/raw/0007-new-ltf-e0-r0.json` 的 rank-local runtime
事件和 coordinator 记录逐项对齐后，确认异常不在 Gloo collective 本身：`completion_observed` 到
`completed_sent` 约为 1.5--2.7 ms，但 coordinator 收到对应 `COMPLETED` 前还会等待约
40--48 ms；容量因此一直保持 `CAPACITY_FULL`，下一次 grant 被推迟。

将新 runtime 的逐通信 `torch.all(...).item()` 校验后置后，延迟仍复现；固定
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` 也没有改变形态。控制通道原先没有设置 `TCP_NODELAY`，
而所有控制事件都是独立的小 TCP/NDJSON 消息。该延迟的量级和对照结果与 Nagle/延迟 ACK 的
交互一致，不能把它解释成中心化 coordinator 的必然成本。

### 13.2 已实施修正

1. `src/runtime_comm_scheduler/runtime/transport.py` 在 client 和 accepted server socket 两端
   设置 `TCP_NODELAY`，减少 grant/progress 控制消息的等待。
2. `examples/jobpacer/runtime/runtime_worker.py` 在线性和 DAG runner 中保存 tensor 引用，统一在
   `runtime.finish_epoch()` 完成通信 drain 后执行 correctness scan；校验不再阻塞后续 ready、
   grant 或应用 makespan。
3. 新 runtime 增加 default process group 的共同 application release barrier，并输出
   `application_makespan_us`、`communication_drain_makespan_us`、`validation_total_us` 等边界。
   `run_runtime_replay.py` 的 `performance` 与旧 replay 的 `performance` 使用同一 rank-local
   duration 聚合语义，不跨 rank 直接相减；批处理优先读取该统一字段。

### 13.3 修正后验证

针对性单元检查：

```text
PYTHONPATH=src:. pytest -q tests/unit/runtime tests/unit/test_jobpacer_runtime_worker.py \
  tests/unit/test_jobpacer_runtime_results.py tests/unit/test_jobpacer_comm_profile.py
51 passed
```

真实双 rank 复测命令：

```text
env PYTHONPATH=src:. python -m examples.jobpacer.runtime.replay_launcher \
  --policy ltf --workload balanced --backend gloo --world-size 2 \
  --timeout 20 --compute-jitter 0.3 \
  --comm-profile /tmp/jobpacer-phase3-profile.json \
  --output /tmp/jobpacer-phase3-followup-ltf.json
```

结果为 `validation=ok`，两个 rank 的 all-reduce tensor 均正确；六项通信的 coordinator
`all_submitted_to_all_completed` 为约 1.31--2.05 ms，中位数约 1.56 ms，application
makespan 为 23.018 ms，communication drain makespan 为 23.943 ms。原始结果保存在
`/tmp/jobpacer-phase3-followup-ltf.json`。这是一项同配置的单次修正复测，用于确认异常消失，
不替代旧 pilot，也不构成策略性能收益或正式多 seed 结论。

后续正式实验必须在修正后的代码上重新生成 profile/批次，并完成正式规模的多 seed/repeat、isolated
基线和持久化原始产物；有限规模的 block 随机化和源码快照已在 13.4 完成。本节只验收完成反馈
异常已得到针对性修复。

### 13.4 修正后小批重复

为检查修正后的测量链路，运行了 5 个新 runtime 策略、2 个 epoch/seed、每个 2 次 repeat，
共 20 条真实 CPU/Gloo replay：

```text
env PYTHONPATH=src:. python -m examples.jobpacer.experiments.gloo_phase3_batch \
  --output-dir /tmp/jobpacer-phase3-followup-small-20260922 \
  --workload balanced --backend gloo --world-size 2 --seeds 0,1 --repeats 2 \
  --compute-jitter 0.3 --wait-budget-s 0.02 --poll-interval 0.001 --timeout 20 \
  --comm-profile /tmp/jobpacer-phase3-profile.json --order-seed 20260922
```

20/20 的 `status=ok` 且 validation 通过。批次按 seed/repeat block 随机化策略顺序，
`manifest.json` 为 schema 2，保存了 17 个相关源码文件的 SHA-256 摘要，source snapshot
digest 为 `489f2e6c5b33a88ce5d1281cbc3d3d4c010df2a6d3afdc2282a6b46560c345b0`。
各策略各有 4 个样本；makespan 中位数（秒）为 static FIFO `0.023292`、static LTF
`0.021082`、FIFO `0.020799`、LTF `0.020838`、Lookahead `0.021206`。这些样本用于确认
修正后链路和 block 产物完整，不足以证明策略排序或性能收益。

## 14. 正式实验输入与分析闭环实施（2026-09-22）

本轮只实施支撑与输入，未启动正式性能矩阵，也未向 result 文档追加性能结论。

1. 线性 workload 新增 execution-only 时长覆盖，支持按 job/task/rank/segment
   确定性指定实际 producer/consumer 时长。Phase 1/2/3 共用同一解析和采样函数，
   policy hint 仍只读名义时长。
2. 新旧 replay 在应用 release 前按 group 和实际消息签名执行可配置预热，
   预热 collective 不经调度 runtime，不进入任务集、静态序列和 JCT。
3. DAG 路径接通严格 communication profile，根据 op/bytes/dtype/group size/backend/
   device/reduction 覆盖所有 comm node，缺签名或环境不匹配时在启动前拒绝。
4. `benchmark/phase3/experiments/` 新增多尺度校准、L0–L5、G0–G4、L0 isolated
   分母和 pilot/screening 清单。输入 README 规定每个场景的 trace 验收条件；
   manifest 内通信估值是待 profile 覆盖的占位值，不作为校准结论。
5. 批处理拒绝复用非空输出目录，避免旧 JSON 使失败运行被误判成功；
   manifest 补充 CPU model、affinity、线程环境和 DAG/旧基线源码摘要。批次产物增加
   `jobs.csv`、`mechanisms.csv`、`paired-summary.csv` 和 `analysis.json`，并可用
   `--isolated-jobs` 按模式/策略/seed/repeat/job 严格连接分母计算 slowdown。

这些文件与接口只使正式 pilot/screening 可执行。机制是否真实触发、尺度选择、
多 seed 不确定性和性能收益仍必须由后续真实 CPU/Gloo 运行验收。

## 15. 校准与机制 pilot 执行记录（2026-09-22）

本节记录本轮实际启动的真实 CPU/Gloo 样本。它们仍是 pilot，不替代正式 screening
或 `30 seed × 3 repeat` 主矩阵。

### 15.1 多尺度校准

使用 `benchmark/phase3/experiments/calibration/multi-scale.json` 和
`run_comm_profile`，`warmup=5`、`iterations=30`、world size 2。profile 产物为
`/tmp/jobpacer-phase3-calibration-profile-20260922.json`，文件 SHA-256 为
`9fe7de4d1c710befcef2ddadd33ad1ef910b18beec9721f98f4d0e0c647aba3f`。

| bytes | p10 | p50（注入估计） | p90 |
| ---: | ---: | ---: | ---: |
| 4 KiB | 0.512 ms | 0.577 ms | 0.696 ms |
| 1 MiB | 2.419 ms | 3.268 ms | 7.302 ms |
| 16 MiB | 19.185 ms | 22.180 ms | 24.764 ms |

每个签名均有 30 个样本，环境为 CPU/Gloo、float32、sum、group `[0, 1]`。

`two-groups.json` 的校准也已执行：bare 结果为 `validation=ok`、workload makespan
26.567 ms、峰值在途 2；旧 scheduler `max_outstanding=1` 结果为
`validation=ok`、workload makespan 43.936 ms、严格串行 admission 验证通过。新 runtime
五策略校准批次 5/5 成功，产物为
`/tmp/jobpacer-phase3-calibration-new-20260922/`。这些是各一条配置观测，不能作为稳定
的 bare/旧 scheduler/新策略性能排序。

### 15.2 L0–L5 与 G0–G4 机制 pilot

pilot 使用 seed/epoch 0、每场景每策略 5 次 repeat、严格 profile、每次 replay
`warmup_iterations=1`，每个场景单独输出到
`/tmp/jobpacer-phase3-mechanism-pilot-20260922/`。除第一次 L3 的一次本地端口占用外，
重跑批次均 25/25 成功；L3 原批次保留在同名目录，重跑结果位于
`L3-lookahead-rerun/`。

| 场景 | 正确运行 | trace 机制证据 | 结果 |
| --- | ---: | --- | --- |
| L0 | 25/25 | validation-only | 可运行；不以此证明策略收益 |
| L1 | 25/25 | static FIFO/LTF 队首阻塞各 5/5 | 稳定触发静态队首阻塞 |
| L2 | 25/25 | `max_simultaneous_eligible >= 2` 为 25/25；gate-first 随策略为 0–5/5 | 候选竞争稳定，gate-first 非策略不变 |
| L3 | 25/25（重跑） | Lookahead wait 1/5 | 未形成稳定主动等待，不进入 screening |
| L4 | 25/25 | Lookahead wait + deadline fallback 1/5 | 未形成稳定失准回退，不进入 screening |
| L5 | 25/25 | OFFER spread ≥5 ms：各策略 4–5/5 | 成员偏斜证据基本稳定 |
| G0/G1/G3 | 各 25/25 | 节点/依赖/组内顺序校验通过 | 正确性样本成立 |
| G2 | 25/25 | `max_simultaneous_eligible >= 2` 为 25/25；gate-first 随策略为 0–5/5 | 多前沿竞争成立，gate-first 需按策略解释 |
| G4 | 25/25 | Lookahead wait + 无提前 unsafe 预测 1/5 | 当前输入未形成稳定等待机制 |

本轮还修正了 `run_experiments.py` 的 G4 机制判定：`unsafe` 只有在其预测时刻早于
同一 coordinator rank 观测到 `job-1/current` 完成时才算违规。此前只要 trace 中出现
过 `unsafe` anticipated 就标红，会把 current 已完成后的安全 frontier 声明误判为违规；
对应回归检查已加入 `tests/unit/test_jobpacer_phase3_experiments.py`。

L3 的初次失败原因为 rank 0 `OSError: [Errno 98] Address already in use`，rank 1 随后
报告 control connection closed；不是 collective、协议或 validation 错误。该环境故障未从
原始批次中删除，重跑原因和新目录均已保留。

### 15.3 L0 isolated 与 slowdown

`isolated/job-0.json`、`isolated/job-1.json` 各运行 5 个策略 × 5 repeat，均为 25/25
成功；随后 L0 shared batch 25/25 成功，并通过两个 `jobs.csv` 生成逐 job slowdown。
产物位于 `/tmp/jobpacer-phase3-isolated-pilot-20260922/`。

按策略聚合两 job、5 repeat 的 10 条 slowdown，median 为：static FIFO `0.557`、
static LTF `0.536`、FIFO `0.510`、LTF `0.571`、Lookahead `0.530`。这些分母已经匹配
runner、backend、profile、seed/repeat 和 warmup，但仍只有一个 seed，且产物当前位于
`/tmp`；不据此宣布 shared run 优于 isolated。

### 15.4 当前推进结论

多尺度 profile、裸/旧/新校准、L0–L5/G0–G4 pilot 和 L0 isolated 闭环已经实际跑通。
L1/L2/L5 以及 G0/G1/G2/G3 有足够的 pilot 机制/正确性证据进入下一轮筛选；L3/L4/G4
的等待/预测触发率不足，继续扩大 seed 前应先调整输入时间窗口或明确把它们作为负对照。
正式多 seed 性能矩阵、完整旧新共同对照、持久化原始 trace 归档和正式统计结论仍未完成。

## 16. Screening 执行记录（2026-09-22）

按照 `benchmark/phase3/experiments/suites/screening.json`，对 pilot 中机制证据较稳定的
L0/L1/L2 和 G0/G1/G2 执行了 `seeds=100..109`、每 seed 一次、五个新 runtime 策略的
配对 screening。共 300 次真实 CPU/Gloo replay，六个场景均为 50/50 `validation=ok`。

输出位于 `/tmp/jobpacer-phase3-screening-20260922/`，每个目录包含 raw JSON、逐 job 表、
机制表、配对汇总和 bootstrap 分析。`paired-summary.csv` 的 baseline 是 `new-static_fifo`，
speedup 大于 1 表示候选的 makespan 较小；先在每个 seed 内取 repeat median（本批每 seed
只有一次），再做 2000 次 seed-block bootstrap。

| 场景 | static LTF | FIFO | LTF | Lookahead |
| --- | ---: | ---: | ---: | ---: |
| L0 | 1.033 (0.980–1.184) | 1.067 (0.960–1.170) | 1.074 (0.944–1.234) | 1.104 (1.005–1.229) |
| L1 | 1.011 (0.910–1.055) | 1.082 (0.998–1.183) | 1.192 (1.100–1.265) | 1.086 (0.963–1.202) |
| L2 | 0.977 (0.863–1.034) | 1.051 (0.914–1.144) | 1.150 (1.059–1.257) | 1.071 (1.009–1.188) |
| G0 | 1.689 (1.624–1.818) | 1.723 (1.551–1.891) | 1.703 (1.616–1.779) | 1.360 (1.258–1.713) |
| G1 | 1.008 (0.925–1.104) | 1.010 (0.800–1.274) | 0.981 (0.873–1.127) | 0.932 (0.751–1.081) |
| G2 | 1.368 (1.296–1.498) | 0.983 (0.938–1.285) | 1.037 (0.948–1.337) | 0.951 (0.937–0.988) |

这些结果只说明 screening 的场景差异和统计产物链路已经可读：G0 的线性 bridge 与 G2
的静态 LTF 观测方向不同，G1 的区间跨过 1，不能合并成“动态策略普遍更好”的结论。
screening 没有包含旧 Phase 1/2 对照，也没有达到正式 `30 seed × 3 repeat` 规模。

本轮未将 L3/L4/G4 放入 screening：它们在 5-repeat pilot 中 Lookahead wait/fallback
只发生 1/5 次，输入尚未稳定触发预期机制。L5、G3 也尚未进入本轮配对 screening。

## 17. Screening 设计修正（2026-09-23；未运行新实验）

保留第 15–16 节原始 pilot 和 screening 数值，更新解释及后续输入：

1. 原 screening 六个场景的 `compute_jitter=0`；seed 100–109 没有抽取不同
   计算时长。`suites/screening.json` 标记为无扰动重复对照并显式固定 jitter、
   等待预算、轮询间隔、baseline 和顺序 seed。`perturbation-screening.json`
   选定 jitter 0.3，先覆盖 L0/L1/L5、G0/G1；其配对区间才可讨论所采样的
   计算扰动范围，仍需检查机制触发与运行环境。
2. L2/G2 原判定只检查 gate-first 和任意两个 eligible。现记录指定组内首项候选的
   全体成员是否在 gate 全员完成前 OFFER 到齐，以及首个研究候选决策时两者
   是否同时 eligible。容量占用期间 coordinator 不记录首次 eligible，前一条件
   由 OFFER 覆盖、节点依赖和组内首项约束重建。
   `priority-pilot.json` 把共同快照中的 FIFO/LTF 选择与自然到达的整体性能拆开。
   后者保留全部样本，按策略报告 gate-first 比例，不作事后成功样本筛选。
   第 15 节“候选竞争稳定”的描述只对应旧宽松指标，不能追认为严格触发率。
3. G0 新增 `linear/L0-no-overlap-bridge.json`，G0 compute 节点用相同 seed 和
   `execution.linear_sample_keys` 对应线性 producer 样本；两边 consumer overlap
   都为零。DAG 的 `G0-interleaved-order.json` 可注入 static LTF 组，同时保留
   原 static FIFO。比较 DAG/线性 runner 开销时用桥接输入；策略归因还要看
   FIFO→LTF、LTF→Lookahead 的配对结果，不能把相对逐 job Static FIFO 的
   约 1.7× 直接称作动态适应收益。
4. G1 保留单 job diamond 负对照：验 join、overlap 隐藏和无选择空间时的开销。
   将来研究 diamond 调度选择需另加竞争前沿。
5. 旧 isolated/shared 分批运行且线程环境未固定，约 0.5 的 slowdown 暂不解释。
   批处理现在校验两批的 profile、源码、环境、计算样本和预热配置；下一次
   小规模检查需随机交错 shared 与两个 isolated 输入并固定线程环境，检查
   tensor 构造、grant/launch/反馈分段和后台负载。配置校验本身不能替代交错。
6. L3/L4/G4 的名义目标到达差从约 5 ms 缩短到约 2.5 ms，并另列 4 ms
   等待预算 pilot。依据 1 MiB profile p50 约 3.268 ms，只有实际 trace 的
   `wait_score < dispatch_score`、目标到达与 deadline 证据满足条件后才扩批。

以上是输入、判定和分析入口调整；本节没有追加性能观测，也没有改变第 16 节的原始结果。

## 18. 精简正式实验实施（2026-09-23；未运行性能 replay）

原 `suites/formal.json` 的 6570 次计划保留作历史方案；后续执行清单为
`benchmark/phase3/experiments/suites/compact.json`。主实验只在 L0/L1 做
Phase 1/2/3 七组跨阶段对照；L2 两组、G2 四组、L5 两组。10 seed × 3 repeat、
jitter 0.3，共 660 次。机制层为固定输入零扰动、每配置 5 repeat，共 90 次；
isolated 诊断将 L0 shared 和两个单 job 输入随机交错，共 30 次。基础预算 780 次。
只有 L3 准时到达与 L4 超时回退通过 pilot 后，才分别追加两组 × 10 seed ×
3 repeat，共可选 120 次。G4 仍在机制层验证 DAG 安全预测，不先加入大矩阵。

批次实现 `examples.jobpacer.experiments.gloo_phase3_batch` 支持 `--arms`、`--preview`、`--resume`、`--history-summary`。
每个 `(seed, repeat)` block 随机化所选组顺序，结果按同 seed repeat 中位数和
seed-block bootstrap 分析。成功运行的同配置结果安全跳过；失败重试有独立 attempt
路径，旧记录留在 `runs.jsonl`。续跑先比对输入、profile、静态顺序、源码、git HEAD、
环境和批次参数摘要。套件入口 `examples.jobpacer.experiments.compact_suite` 默认预览；只有显式 `--execute`
才启动串行 replay。其主阶段在 L2/G2 的每策略 pilot 中要求至少 3/5 严格触发；
Lookahead 可选阶段要求 L3/L4 的 Lookahead pilot 各至少 4/5。门槛写入清单，
未达标则停止扩批。门槛用于决定是否扩大场景，不用于筛掉主矩阵的自然到达样本。

正式套件要求在固定 CPU affinity、线程变量下生成的新 profile；profile 文件需记录
两者，套件执行时逐项核对。旧 profile 缺这些字段，若无法证明一致，必须重新校准。
持平阈值由独立噪声 pilot 与实际意义事先确定，主阶段执行时显式传入，不沿用默认 1%。
已有完全同条件的结果可经摘要和原始记录核对后抵扣；工具不自动跨不同批次拼接样本。
历史失败、未触发、负收益照原样保留。命令与参数说明见 `suites/README.md`。

### 18.1 首次执行状态更新（2026-09-23）

精简 suite 已完成多尺度 profile、90 次机制验收、30 次 interleaved isolated、60 次
独立零扰动噪声 pilot，以及 L0/L1/L5 共 480 次主矩阵 replay。原始数据持久保存在
`benchmark/phase3/results/phase3-compact-runs-20260923/`。主矩阵采用噪声 pilot
测得的 11.96% P95 同配置差异作为 tie threshold。

L2/G2 的严格 gate-first 候选竞争均未达到每策略至少 3/5 的扩批门槛；L3 的
Lookahead 准时到达为 0/5，故 L2/G2 的 180 次主批次及 Lookahead 可选 120 次未启动。
机制 trace、配对估计、校准数值及解释边界详见
`docs/JobPacer/result/phase3experiments.md` 的“精简 suite 首次执行”节。此次执行
没有修改运行时、workload 或 suite 配置。

## 19. Runtime overhead 实验实施（2026-09-24）

按 `docs/JobPacer/process/phase3experiments.md` 前述运行时开销方案实施并完成 A–D；范围保持两 rank
CPU/Gloo、单全局在途 collective。没有扩至 GPU、多 inflight 或 DAG 性能矩阵。

实现方面，线性 Phase 3 replay 增加 `--binding-preparation precreate|on-ready`，默认对照使用
precreate：每 rank 在 ProcessGroup 建立后、预热和应用 release 前创建本 rank 独立 tensor/binding，
producer 完成后才 submit；on-ready 保留旧创建时机。两模式的准备失败都由全 rank 同步并有界退出。
DAG 路径不变。Phase 2 继续在 release 前创建 tensor，并记录既有准备成本。结果 JSON 记录准备区间、
每项创建时间、application release/end、通信 drain、validation 边界和准备模式。

新 runtime 补记 collective 调用开始/返回、SUBMITTED 发送起止、完成探测计数和应用 wait 返回；结果汇总
分别保留 rank-local 间隔与 coordinator 单时钟的 `grant→SUBMITTED 到齐`、
`SUBMITTED 到齐→COMPLETED 到齐`、`COMPLETED 到齐→下一 grant` 及间隔中合法候选是否存在。
不以提交回执替代 collective 返回，不将观测时刻称为精确物理完成，不推算未实测的 policy 调用耗时。

离线分析入口 `examples.jobpacer.analysis.visualize_phase3 batch` 为每个批次生成 makespan 图、独立
代表运行时间线、预先固定首个共同 seed/repeat 的 rank 0/1 配对时间线，以及 rank/coordinator 诊断、
逐 job 配对 JCT 和 `analysis.md`。分析代码 SHA-256 写入批次报告。运行时实验 raw 和 CSV 在被忽略的
`benchmark/phase3/results/` 本地产物树中，不会自动加入版本控制。

Stage 0 验证两种准备模式与 FIFO/LTF/static FIFO，覆盖 producer 先于 OFFER、grant/launch 投影、tensor
独立与数值校验，以及两种模式的 binding failure 有界退出。真实 Gloo 集成定向测试 8 passed、30
deselected；完整单元测试 209 passed、43 skipped；`git diff --check` 通过。

正式运行分成 A（90）、B（60）、C（90）和 D（180）次，共 420 次 replay，另有 C 阶段 6 次 smoke；
均串行执行。每个正式 replay 退出成功且 validation 为 ok。A/B 的 source snapshot digest 相同；
C/D digest 的差异仅为离线可视化/诊断代码变化，参与 collective 执行的源码文件摘要相同。批次均记录
dirty worktree、共同 profile SHA-256、CPU 型号/affinity、线程变量和原始 trace SHA-256。profile 使用
`phase3-compact-20260923/profile.json`，digest 为
`0283de537d286411b5014ca5436b6284c748fecc122360c8d1dfd8ab42573093`。详见
[`docs/JobPacer/result/phase3experiments.md`](../result/phase3experiments.md) 的 2026-09-24 runtime-overhead
结果节。

## 20. Runtime control-path 定位与单项候选（2026-09-24 UTC）

本轮继续使用 CPU/Gloo、两 rank、`max_inflight=1`、precreate、1 ms 完成轮询和固定线程环境。没有扩展策略矩阵。
诊断分为最小观测与紧凑诊断；两者都保留协议检查、任务/launch 顺序、tensor 校验和失败处理。`full` 保留原有
详细事件供兼容和深入诊断使用。每个热路径时间差只在所属 rank 或 coordinator 的本地 monotonic 时钟内计算。

最初两版诊断模式在 4 KiB × 32 的 E1 中分别增加约 49.991 ms 和 52.271 ms，故暂停 E2，缩减重复 coordinator
快照后再继续。最终紧凑诊断版本 E1 为 10/10 成功：最小观测 makespan 中位数 88.545 ms，诊断观测 99.764 ms，
差 11.219 ms；两 rank 汇总 process CPU 中位数分别 115.407/131.037 ms。诊断仍有可见扰动，结果未通过减去常数
方式修正。三次 E1 探测批次都留在结果目录，其中只有 compact-diagnostic 用作后续诊断配置。

E2 20/20 成功。makespan 从两路径共同的最大 rank application-release→application-end duration 计算，避免旧路径
与新路径混用指标别名。每配置为固定 epoch 6200 的五次 repeat，按 repeat 块交错运行，属于固定输入系统噪声观察，
并非五个独立 seed。结果如下：

| 输入 | old FIFO 中位 makespan | new Static FIFO 中位 makespan | 配对 new−old 中位差 | 每 collective 配对差 |
| --- | ---: | ---: | ---: | ---: |
| 4 KiB × 32 | 25.429 ms | 100.610 ms | +75.893 ms | +2.372 ms |
| 1 MiB × 32 | 57.292 ms | 131.244 ms | +73.952 ms | +2.311 ms |

新路径诊断中，4 KiB/1 MiB 的 rank-local submit→grant 中位分别约 1.211/1.243 ms，grant→collective start
约 0.178/0.184 ms，collective return→completion observation 约 1.086/2.045 ms，应用 wait 约
2.416/3.389 ms。首次 probe 通常在 collective return 后约 0.70/0.72 ms。Coordinator 单时钟上，event queue
等待中位约 0.12 ms、消息处理约 0.032 ms；容量释放到下一项首次 eligible 约 0.42 ms，候选 ready 后容量未继续
占用，policy 函数单独实测中位 0.003 ms（P90 0.005 ms），完整 decision processing 约 0.030 ms，grant writer
queue 到 sendall end 约 0.23–0.24 ms。它们是不同边界的
描述统计，不可相加为完整耗时分解；Phase 2 结果没有记录 CPU/context-switch 数据，故不做旧新 CPU 对照。

单项 E3 候选是让新 runtime 在 work 进入可探测集合时唤醒完成探测线程，同时保留周期轮询。默认行为没有改变，
`--wake-completion-on-submit` 只用于显式消融。E3 同样 20/20 成功：4 KiB 的首次 probe 中位从 0.673 ms 降到
0.152 ms，但 probe 数中位从 1 增到 2，completion observation 反而从 1.200 ms 到 1.312 ms；makespan 中位从
100.511 ms 升至 105.818 ms，五个配对 repeat 全部变慢。1 MiB 的首次 probe 也提前（0.694→0.153 ms），但
completion observation 从 2.008 ms 增至 2.388 ms；配对方向混杂，wakeup 只在 3/5 repeat 较快，arm 中位
makespan 从 132.953 ms 升至 137.899 ms。两个输入的汇总 rank process CPU 中位数均略增，额外 probe 没有转化为
稳定的应用收益。

因此 E3 未达到“预期区间改变且整体收益可重复”的门槛，E4（L0/L1 105 次回归）按计划不启动；没有将 wakeup
设为默认值，也没有另选第二项优化继续试探。当前证据排除了“中央 policy 计算本身占据约 2 ms”的解释，显示
准入/反馈和完成观测各有可测等待，但还不足以把全部新旧差值唯一归因到某一环节。结果、原始 JSON、paired
timeline 图及分析文件位于 `benchmark/phase3/results/runtime-overhead/`，结论见
[`phase3-runtime-overhead-20260924.md`](../result/phase3-runtime-overhead-20260924.md) 的 follow-up 节。

验证：针对性 unit tests 59 passed；真实双 rank Gloo 集成 40 passed，覆盖观测模式、grant/launch 投影、
SUBMITTED→COMPLETED 顺序、launch/probe failure 和有界关闭；E1/E2/E3 共 70 个固定输入正式对照 replay
全部 validation=ok。以上不代表 GPU/NCCL 或多机验收。

## 21. E2 已有 trace 的逐任务复核（2026-09-25）

按后续排查要求，先核对已有 E2 的 4 KiB × 32 与 1 MiB × 32 trace，没有重跑 E2，也没有改写其 manifest、
runs ledger、raw JSON 或既有图。原始 E2 的 20 次运行均通过校验。E2 缺少 DECLARE 调用起止边界，因此只补
了 rank-local DECLARE start/end、submit 开始，以及每条控制消息一次性的发送锁等待和 socket 写入/flush
区间；minimal 模式不产出这些详细事件。新增量不改变准备时机、轮询周期、消息顺序或完成协议。

对 coordinator 的 155 个任务转移/输入，以每个 run 内任务统计汇总后再比较五次 repeat。容量释放时已有至少
一个 OFFER 尚未进入中央输入队列的转移占 4 KiB 的 141/155、1 MiB 的 137/155；没有转移在容量仍占用时先
变为 eligible。容量释放到首次 eligible 的 run-level 中位数分别为 415 µs 和 420 µs；两条件满足后到
decision start 为 3–4 µs，decision processing 为 30–32 µs。更细分的 coordinator 与 rank-local 分段表见
[`follow-up analysis`](../../../benchmark/phase3/results/runtime-overhead/instrumentation/20260925-E1-declare-send-lock/followup-analysis.md)。

E2 的 rank trace 只能看到 DECLARE 返回点，于是另跑规定的 E1 补充批次：4 KiB × 32、new Static FIFO/precreate、
1 ms polling，minimal 与 diagnostic 各 5 次交错，共 10 次；全部 Gloo replay 验证通过。诊断观测 makespan
中位数 114.429 ms，minimal 为 102.038 ms，配对 repeat 差的中位数 +8.919 ms；CPU 时间也增加。该扰动有实质
影响，未通过减固定值修正性能。

新增 rank-local 结果显示，grant 收到到 launch worker dequeue 的中位数约 176–185 µs，dequeue 到 collective
调用约 11–12 µs；不是主要重复等待。DECLARE 调用中位数 rank 0/1 为 319/170 µs，其中发送锁等待中位数
69/60 µs，且常与前一项 COMPLETED 发送持锁区间重叠；OFFER 锁等待中位约 1 µs。较晚 OFFER 的 rank 会随
消息规模变化，不能归结为固定慢 rank。E2 的 collective 返回到完成观测仍约 1.08–1.17 ms（4 KiB）和
1.81–2.15 ms（1 MiB），此区间包含 backend 完成及轮询，不是精确物理完成延迟。

结论停在定位：下一次 grant 最大且反复出现的中央门控等待，是最后一个成员 OFFER 尚未到达 coordinator
输入队列；这可能来自成员应用推进/DECLARE/producer/submit 或接收线程排队，不能称作单向网络延迟。条件
满足后的中央处理、grant 发送排队和本地 launch worker 等待都更短。不同 rank 时钟未互相相减，各阶段区间
也未相加解释完整 2 ms。没有做 runtime 优化或启动额外策略矩阵。

验证：runtime unit tests 42 passed；全仓 218 passed、45 skipped；允许 TCP socket 的真实双 rank Gloo 集成
40 passed；git diff --check 通过。E1/E2 本地 raw、统计表与双 rank 配对时间线位于
`benchmark/phase3/results/runtime-overhead/instrumentation/20260925-E1-declare-send-lock/`。

## 22. DECLARE 消息消融 F0–F4（2026-09-25）

本轮只改变线性 workload 的 declaration 时机，不调整发送锁、线程结构或 completion polling。`before-producer`
仍为默认，按 DECLARE→producer→submit/OFFER 执行；`on-submit` 在 producer 完成后直接 submit，使用 OFFER
携带的完整 TaskSpec/TaskHint，不新增协议。两种模式共享任务、预创建 binding、profile、静态/策略入口和正确性检查；
Lookahead 与 DAG 拒绝 `on-submit`。改动限于 Phase 3 入口、rank harness、实验批处理、结果分析和测试。

F0 真实 CPU/Gloo 双 rank 定向集成覆盖两种模式的 TaskSpec/Hint、任务和 launch/grant 投影、producer-before-OFFER、
tensor 校验、元数据冲突、缺失任务、launch/探测失败有界退出及 Lookahead/DAG 拒绝：10 passed、37 deselected。
全仓测试为 221 passed、51 skipped。没有改 coordinator 协议。

F1/F2 使用 4 KiB×32、Static FIFO、precreate、1 ms poll、minimal/diagnostic 各 5 次；F3 在 1 MiB×32 上
minimal 复测 5 对；均串行随机化模式顺序。F1 与 F3 的五个固定输入配对均为 `on-submit` makespan 更低：
F1 中位差 −13.970 ms（范围 −45.260 至 −11.421 ms），F3 中位差 −12.871 ms（−41.512 至 −10.953 ms）。
它们证明零计算链上的差异可重复，但不是多 seed 鲁棒性结论。F2 为诊断观测，绝对 makespan 受额外 trace 影响，
只用于机制比较，不与 minimal 批次拼接或减去固定观测成本。

F2 先在每次 replay 内汇总 32 次通信，再比较配对 repeats。`on-submit` 的上项 wait 返回→下一次 submit 中位
减少约 540 µs（rank 0，5/5）及 286 µs（rank 1，5/5）；coordinator 的容量释放→最后 OFFER 入队减少
550 µs（范围 −690 至 −227 µs，5/5），容量释放→首次 eligible 减少 562 µs（−642 至 −171 µs，5/5）。
两条件满足后的 decision start 基本不变，decision processing 只有数微秒波动。与此同时 rank 0 OFFER 发送锁等待
中位增加约 147 µs（5/5 增加），rank 1 中位增加约 2 µs。证据支持 DECLARE 移除缩短应用推进和中央等成员的
空档，但更早 OFFER 部分转移了对控制发送锁的竞争；不能仅以 DECLARE 耗时消失判断收益。

门控通过后执行 F4：L0/L1、new FIFO、5 个执行 seed×3 个 repeat×2 种模式，共 60 次串行 CPU/Gloo replay。
使用 Phase A 同一 profile（digest `0283de537d286411b5014ca5436b6284c748fecc122360c8d1dfd8ab42573093`）、
0.3 execution jitter、1 ms poll 和 precreate。60/60 replay validation 成功，协议发送/回执计数符合各模式预期；
30/30 workload/seed/repeat 对的 producer/consumer 计算样本逐 rank、job 完全相同，且样本数量与输入一致。

F4 按 seed 内三次 repeat 中位数，再比较五个 seed block。`on-submit−before-producer` 的 makespan seed 中位差：
L0 为 −0.354 ms（五 seed 范围 −0.963 至 +2.223 ms，3/5 seed 更快）；L1 为 +0.406 ms（−0.316 至
+1.461 ms，1/5 更快）。15 个 repeat 配对的范围分别为 −4.748 至 +4.423 ms、−4.788 至 +4.018 ms，
明显宽于中心差值。两场景每 replay process CPU 时间中位减少约 2.4 ms，voluntary context switch 中位减少
45/47 次；但 L0/L1 应用 makespan 均未显示跨 seed 稳定改善。逐 job JCT 也有正负变化。

因此结论限于：省去独立 DECLARE 可减少控制消息和 CPU/线程调度负担，并在零计算链缩短路径；F4 没证明这些
收益能稳定转化为有计算/扰动应用的 makespan 改善。默认保持 `before-producer`，不扩 LTF、Lookahead、DAG 或
新性能矩阵。原始记录和表格位于本地忽略目录
`benchmark/phase3/results/runtime-overhead/control-path/20260925-F-declaration-mode/`；完整解释见
[`Phase F 结果记录`](../result/phase3-runtime-overhead-20260924.md) 的后续章节。F4 每 workload 的 makespan 图
分别位于该批次 `figures/L0-balanced/` 和 `figures/L1-head-misalignment/`。性能结果只适用于本机 CPU/Gloo，
不外推至 GPU/NCCL、多机或其他 workload。
