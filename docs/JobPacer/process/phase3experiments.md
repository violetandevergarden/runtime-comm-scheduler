# JobPacer Phase 3.1 / 3.2 CPU 实验实施方案

日期：2026-09-22。状态：已完成最小实验支撑、CPU/Gloo pilot 及一次完成反馈延迟修正复测；正式多 seed 性能验收仍未完成。

本文件规定实验准备、执行、分析和产物要求。阶段目标沿用
[原始实验计划](../260914项目计划讨论/JobPacer实验计划.md)、
[Phase 3.1 计划](../plan/phase3.1.md) 和 [Phase 3.2 计划](../plan/phase3.2.md)，
不改变 runtime 的协议和阶段范围。已验证事实分别见
[Phase 2 结果](../result/phase2.md)、[Phase 3.1 结果](../result/phase3.1.md)、
[Phase 3.2 结果](../result/phase3.2.md) 和 [最近结构回归](../result/phase3.12fix.md)。
历史通过记录不代表本方案已执行，也不证明策略性能收益。

> 目录迁移说明：当前实验入口位于 `examples/jobpacer/scripts/`，因此下面的
> `run_phase1.py`、`run_replay.py`、`run_runtime_replay.py`、`profile_communication.py`
> 和 `run_phase3_experiments.py` 分别对应 `run_phase1.py`、`run_phase2.py`、
> `run_phase3.py`、`run_comm_profile.py` 和 `run_experiments.py`；推荐以
> `PYTHONPATH=src:. python -m examples.jobpacer.scripts.<entry>` 启动。文中的历史命令仅用于
> 说明既有批次，不改写历史产物。

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
| D1 | Dynamic LTF | `--policy ltf` | 在 eligible 候选间按 tail 选择 |
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
入口，并独立验证跨成员 group 顺序。当前 CLI 没有 DAG bare；这项为可选扩展，未实现时明确
不回答该问题。不能把 DAG 展平成串行链、改变 group 或删掉依赖后充当 bare 基线。

## 3. 当前能力与正式实验前的准备

以下基于当前代码检查，不表示已经实施本节改动。

| 项目 | 当前状态 | 本轮处理 |
| --- | --- | --- |
| 新 runtime 五种策略 | CLI 已支持 | 复用并保存实际选择序列 |
| 线性实际计算扰动 | 已支持按 seed/epoch/job/task/rank/segment 固定采样 | 结果中保存 producer/consumer 样本；保持策略估计冻结 |
| DAG 计算扰动 | `--compute-jitter`，按 seed/epoch/job/node/rank 固定样本 | 可复用；输出实际样本摘要 |
| 离线通信 profile | 旧、新 replay 均支持严格 `--comm-profile` | 已将同一 profile 注入线性新旧组，记录 digest、环境和来源 |
| DAG 静态顺序 | 自动 FIFO/LTF，或 `--static-order` 文件 | 保存生成结果，跨 seed 冻结 |
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
PYTHONPATH=src:. python -m examples.jobpacer.scripts.run_phase3 \
  --policy static_fifo --workload balanced --backend gloo --world-size 2 \
  --poll-interval 0.001 --timeout 20 \
  --output /tmp/jobpacer-phase3-new-static.json

# DAG 路径及已有扰动功能
PYTHONPATH=src:. python -m examples.jobpacer.scripts.run_phase3 \
  --policy ltf --dag benchmark/phase3/multi-group.json \
  --backend gloo --world-size 2 --epoch 0 --compute-jitter 0.3 \
  --poll-interval 0.001 --dag-poll-interval 0.001 --timeout 20 \
  --output /tmp/jobpacer-phase3-dag-ltf.json

# 线性 8 组（含旧 bare/旧 FIFO/LTF 与新 runtime 五策略）
PYTHONPATH=src:. python -m examples.jobpacer.scripts.run_experiments \
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
- `summary.csv`：逐 run、逐 job 的标准化指标与配对标识。
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
env PYTHONPATH=src:. python -m examples.jobpacer.scripts.run_phase3 \
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
env PYTHONPATH=src:. python -m examples.jobpacer.scripts.run_experiments \
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
