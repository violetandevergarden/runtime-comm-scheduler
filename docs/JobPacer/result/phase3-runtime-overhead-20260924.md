# Phase 3 runtime overhead experiments — 2026-09-24

本轮完成预定 CPU/Gloo 诊断链：阶段 A 90 次、B 60 次、C 90 次、D 180 次，共 420 次正式 replay；另有
C 阶段 6 次 smoke。所有实验按序运行；420/420 正式结果及 6/6 smoke 的退出码成功、validation 为 `ok`，
对应 raw JSON SHA-256 与 `runs.jsonl` 一致。未扩到 GPU、多 inflight 或 DAG 性能矩阵。

环境为 Intel Core i7-14650HX、WSL2 Linux 6.18.33.2、Python 3.13.15、PyTorch 2.13.0+cu129、
CPU/Gloo、world size 2、affinity `0-23`，OMP/MKL/OpenBLAS/NumExpr 线程均为 1。所有批次使用同一已校准
profile，SHA-256 `0283de537d286411b5014ca5436b6284c748fecc122360c8d1dfd8ab42573093`。Manifest 记录
HEAD `322d8daf287f41a0a63030dd00c74e0fea54ff0d`、dirty worktree、源码/输入摘要、命令和环境。A/B 的源码
快照摘要为 `157c949d…34eb32`，C/D 为 `86bdfd37…cae4e9`；两者唯一文件差异是离线可视化/分析模块，
参与实验执行的源码文件摘要一致。C/D 新增的离线报告另记分析代码 SHA-256。

## A：binding 准备位置

每个场景为 5 seeds × 3 repeats，比较 old LTF、new LTF/on-ready、new LTF/precreate。先在同一 seed/repeat
内配对，再以 seed 为 block bootstrap。正的 speedup 表示 candidate 较快。

| 场景 | on-ready → precreate makespan 差 | speedup（95% seed-block bootstrap） | precreate preparation 中位数 | binding 创建总时长中位数 |
|---|---:|---:|---:|---:|
| L0 | −1.819 ms | 1.079（1.004–1.151） | 5.378 ms | 3.146 ms |
| L1 | −3.013 ms | 1.117（1.003–1.162） | 6.944 ms | 3.054 ms |

这里的负差表示 precreate 的应用 makespan 较短；它把创建工作移到了 release 前，但不能把 5.4/6.9 ms 直接当成
完整程序加速。与同批 old LTF 比，L0 的 new/precreate 仍慢约 6.396 ms（speedup 0.740，95% CI
0.644–0.907）；L1 差约 −0.412 ms（speedup 1.016，95% CI 0.934–1.045）。因此准备时机能解释一部分
应用关键路径，不足以单独解释所有新旧差异。

## B：完成轮询敏感性

每场景为 new LTF/precreate、5 seeds × 3 repeats。0.2 ms 对 1 ms 的差值为 candidate−baseline：

| 场景 | makespan 差 | speedup（95% CI） | 每 rank 进程 CPU 中位数（1 ms → 0.2 ms） | probes/rank-run | voluntary context switches/rank-run | 应用 wait 中位数 |
|---|---:|---:|---:|---:|---:|---:|
| L0 | −1.517 ms | 1.070（1.013–1.113） | 14.834 → 18.458 ms | 11 → 28.5 | 227 → 308.5 | 3.413 → 3.365 ms/task |
| L1 | −0.684 ms | 1.030（0.995–1.096） | 14.722 → 17.231 ms | 11 → 30 | 219 → 312 | 2.879 → 2.418 ms/task |

L0 的 makespan 改善在本批区间内可见；L1 区间跨 1。更短轮询增加进程 CPU、探测次数和主动上下文切换，故不将
0.2 ms 自动设为默认值。

rank-local 的 call-return→completion-observation 中位数约由 2.49/2.41 ms 降至 2.11/2.12 ms（L0/L1）；
它不是精确物理完成延迟。Coordinator 自身单调时钟上的三个区间如下：

| 场景/轮询 | grant→SUBMITTED 到齐 | SUBMITTED 到齐→COMPLETED 到齐 | COMPLETED 到齐→下一 grant | 间隔内存在合法候选 |
|---|---:|---:|---:|---:|
| L0 / 1 ms | 1.319 ms | 2.819 ms | 0.000 ms | 45/45 |
| L0 / 0.2 ms | 1.430 ms | 2.168 ms | 0.258 ms | 45/45 |
| L1 / 1 ms | 1.230 ms | 2.746 ms | 2.418 ms | 45/45 |
| L1 / 0.2 ms | 1.253 ms | 2.260 ms | 2.534 ms | 45/45 |

候选存在并不等于这些区间是纯 policy 计算时间；区间仍包含控制消息和事件处理，未单独计量 policy 函数耗时。

## C：通信链诊断

固定输入、零计算，old FIFO 对 new Static FIFO/precreate；每配置 1 seed × 5 repeats，下面为各 arm 的 makespan
中位数。该设计是固定输入重复，不作 seed-bootstrap 推断。

| 每项消息 | collective 数 | old FIFO | new Static FIFO | new−old |
|---:|---:|---:|---:|---:|
| 4 KiB | 1 | 0.689 ms | 2.922 ms | +2.233 ms |
| 4 KiB | 8 | 6.689 ms | 25.463 ms | +18.774 ms |
| 4 KiB | 32 | 25.963 ms | 96.544 ms | +70.581 ms |
| 1 MiB | 1 | 2.413 ms | 4.527 ms | +2.114 ms |
| 1 MiB | 8 | 16.038 ms | 32.349 ms | +16.311 ms |
| 1 MiB | 32 | 58.708 ms | 129.505 ms | +70.797 ms |
| 16 MiB | 1 | 16.396 ms | 18.254 ms | +1.858 ms |
| 16 MiB | 8 | 124.190 ms | 139.142 ms | +14.952 ms |
| 16 MiB | 32 | 494.906 ms | 559.867 ms | +64.961 ms |

差值随 collective 数近似线性增长：32 次配置每次约 2.03–2.28 ms；而 4 KiB、1 MiB、16 MiB 的 32 次总差
相近（约 65–71 ms）。在这组 CPU/Gloo、单在途设置下，证据更符合每轮固定运行时/准入/反馈成本累积，未见
差值随 payload 大小同比放大。该对照仍包含旧/新执行路径的所有差异，不能称为纯控制通道成本。

## D：策略净收益回归

使用 precreate、1 ms 轮询、jitter 0.3；每场景 5 seeds × 3 repeats。speedup 为表中 baseline / candidate，
大于 1 表示 candidate 较快。

| 场景 | 对照 | makespan 配对差（candidate−baseline） | speedup（95% CI） |
|---|---|---:|---:|
| L0 | old FIFO → new FIFO | +5.150 ms | 0.767（0.707–0.786） |
| L0 | new Static FIFO → new FIFO | −1.368 ms | 1.063（0.978–1.109） |
| L1 | old FIFO → new FIFO | +0.571 ms | 0.975（0.911–0.985） |
| L1 | old LTF → new LTF | +0.534 ms | 0.977（0.929–0.989） |
| L1 | new Static FIFO → new FIFO | −5.880 ms | 1.232（1.161–1.281） |
| L1 | new Static LTF → new LTF | −6.523 ms | 1.243（1.185–1.285） |
| L5 | old FIFO → new FIFO | −0.291 ms | 1.011（0.973–1.016） |
| L5 | new Static FIFO → new FIFO | −5.500 ms | 1.192（1.168–1.234） |

观测表明，L1/L5 上动态策略相对同 runtime 静态队列的改善超过了本轮测到的额外路径成本；L1 与 old 路径接近，
但本批 makespan 仍略慢。L5 与 old FIFO 接近。L0 的 new FIFO 没有显示对 new Static FIFO 的稳定改善，且仍明显
慢于 old FIFO。因此不能概括为 new 普遍更快或动态策略在所有场景都获益。

机制 trace 中，L1 的 new Static FIFO 和 new Static LTF 均在 15/15 次出现队首阻塞；指定的两个研究候选在
首次研究候选决策时同时 eligible 也是各 15/15。L5 的 new FIFO 与 new Static FIFO 成员偏斜条件均为 15/15。
这些比例说明对应场景在本批真实触发，但不替代 makespan/JCT 结果，也不证明其他 workload 有相同触发率。

逐 job JCT 也有取舍：L0 new FIFO 相对 old FIFO 的 job-0/job-1 配对中位差分别约 `+8.142/+4.374 ms`；
L1 new FIFO 相对 old FIFO 为 `+3.846/−12.438 ms`；L5 为 `+2.654/−13.716 ms`。所以有 job JCT 变差的场景，
整体 makespan 改善不能替代逐 job 公平性检查。完整 seed 配对逐 job 表保存在各批次 `job-paired-summary.csv`。

所有批次保留 `manifest.json`、`runs.jsonl`、输入快照、raw JSON、`summary.csv`、`jobs.csv`、
`task-timings.csv`、`paired-summary.csv`、`job-paired-summary.csv`、rank/coordinator 诊断、`analysis.md`
及图表。A/B、C smoke/main、D 的结果路径和图见 [`benchmark/phase3/results/README.md`](../../../benchmark/phase3/results/README.md)。
结果树被 `.gitignore` 忽略，已保存在当前工作区，不自动纳入 Git。

以上数字是本机单环境、有限 seed 的 CPU/Gloo 结果；不推及 GPU/NCCL、多机或多 inflight。性能时间区间有重叠，
不可相加为完整程序耗时。运行中的 control/policy 内部耗时尚未独立 instrument，因此剩余差值只能定位到逐轮路径，
不能进一步唯一归因。

## Follow-up：control-path E0–E3（2026-09-24 UTC）

本 follow-up 在原 A–D 测量之后先量化诊断扰动，再定位固定通信链的每轮额外时间，并只测试一项完成探测候选。
仍是 CPU/Gloo、双 rank、单在途、固定 affinity 和线程数；没有运行 L0/L1 E4。

### E0 与 E1：观测正确性和观测扰动

新增 `minimal` 与 `diagnostic` 观测级别。最小模式关闭热路径详细事件，但不关闭协议、任务全集、grant/launch
顺序、tensor 和失败校验；诊断模式记录本地 submit/OFFER、消息入队/出队/处理、grant 发布、launch handoff、
collective 返回、SUBMITTED、完成探测及应用 wait 边界。`full` 保留原有详细 trace。每次差值只在同一 rank 或
coordinator 时钟域内计算；sendall 返回不解释成远端收到，completion observation 不解释成物理完成时刻。

真实双 rank Gloo 集成 40 passed；针对性 unit tests 59 passed。E1 经两次降低 trace 负担后得到最终紧凑诊断批次，
10/10 成功。第一、二版详细诊断造成约 49.991/52.271 ms 的 makespan 增量；最终紧凑诊断为：

| 观测模式 | makespan 中位数 | 两 rank process CPU 中位数 | 合计 voluntary context switches 中位数 |
| --- | ---: | ---: | ---: |
| minimal | 88.545 ms | 115.407 ms | 2600 |
| diagnostic | 99.764 ms | 131.037 ms | 2685 |

诊断扰动为 +11.219 ms/32 次 collective，约 0.351 ms/项。它仍不可忽略，不能把诊断 trace “减去固定日志成本”；
E2/E3 在同一 diagnostic 级别内对照。三版 E1 原始结果均保留在
[`runtime-overhead/instrumentation/`](../../../benchmark/phase3/results/runtime-overhead/instrumentation/)。

### E2：旧/新逐轮路径分解

每个输入比较 old FIFO 与 new Static FIFO/precreate，各五个固定输入 repeats；每 repeat 块随机化顺序、串行执行。
两个路径统一使用最大 rank-local application-release→application-end makespan。20/20 replay 退出成功并通过 validation。

| workload | old FIFO | new Static FIFO | 配对 new−old 中位差 | 每项差值 |
| --- | ---: | ---: | ---: | ---: |
| 4 KiB × 32 | 25.429 ms | 100.610 ms | +75.893 ms | +2.372 ms |
| 1 MiB × 32 | 57.292 ms | 131.244 ms | +73.952 ms | +2.311 ms |

新路径每项的本地时间分布显示：

| new-path 区间 | 4 KiB | 1 MiB |
| --- | ---: | ---: |
| submit 调用→收到 grant | 1.211 ms | 1.243 ms |
| grant→collective 开始 | 0.178 ms | 0.184 ms |
| collective 返回→完成观测 | 1.086 ms | 2.045 ms |
| 应用 wait 总时长 | 2.416 ms | 3.389 ms |
| collective 返回→首次 completion probe | 0.696 ms | 0.722 ms |

Coordinator 单时钟上，事件队列等待中位约 0.12 ms，消息处理约 0.032 ms；容量释放到下一个候选首次 eligible
约 0.42 ms；候选 eligible 后没有继续占用已释放容量。policy 函数单独实测中位约 0.003 ms（P90 0.005 ms），
完整 decision processing 约 0.030 ms；decision end 到 grant writer queue 约 0.007 ms，writer queue 到
socket sendall end 约 0.23–0.24 ms。这个证据不支持把约 2.3 ms/项归因于 policy 函数计算；控制消息、成员到齐、
线程交接和完成反馈都占有路径，但区间有重叠，不能相加声称完成了精确分解。旧 Phase 2 输出未包含可比的进程
CPU/context-switch 计数。

### E3：提交时唤醒完成探测的候选

唯一测试的优化是：work 成为可探测状态时可唤醒 completion thread，周期探测仍保留。默认仍关闭该开关；新增 CLI
`--wake-completion-on-submit` 只用于消融。优化改变了预期区间：两消息大小下首次 probe 中位都提前约 0.52 ms；
但更早 probe 常在 backend work 完成前发生，增加一次 probe，随后仍要等待周期轮询。

| workload | poll makespan 中位 | wake makespan 中位 | 配对方向 | poll→wake 首次 probe | poll→wake completion observation | 汇总 CPU 中位 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 KiB × 32 | 100.511 ms | 105.818 ms | wake 5/5 更慢；配对差中位 +4.851 ms | 0.673→0.152 ms | 1.200→1.312 ms | 132.633→133.736 ms |
| 1 MiB × 32 | 132.953 ms | 137.899 ms | wake 3/5 较快，但变化混合 | 0.694→0.153 ms | 2.008→2.388 ms | 171.590→173.391 ms |

1 MiB 的五次配对里有较大的正负变化，不能据 3/5 单独判定收益；两个 arm 的 makespan 中位差仍是 wake 慢
4.946 ms。E3 不满足整体收益可重复条件，因此依 gate 不执行 E4 的 105 次 L0/L1 回归。未把该行为设为默认，
也未继续试第二项候选；E3 说明“首次探测更早”并不等于“完成反馈更快”。

E2/E3 每个 workload 都生成 makespan 比较、Phase 1.2 风格的 rank-0 独立代表时间线，以及固定首个共同
repeat 的 rank-0/rank-1 配对本地时间线：
[`E2 figures`](../../../benchmark/phase3/results/runtime-overhead/control-path/20260924-E2-main/figures/)、
[`E3 figures`](../../../benchmark/phase3/results/runtime-overhead/control-path/20260924-E3-completion-wakeup/figures/)。
成批 raw JSON、manifest、runs ledger 和表分别在同名 batch 目录下。E2/E3 的 `runs.jsonl` SHA-256 为
`07e0e16c…836aca` / `c86294dd…122b81f`；重建表格时验证 raw SHA，没有改写 manifest、runs ledger 或 raw。

结论仍是定位而非新性能验收：此环境下观察到的约 2.3 ms/collective 新旧差距并非中央 policy 计算单点导致；
完成探测唤醒候选提前了首次 probe，却没有缩短完成反馈或应用 makespan，因此保留原始周期轮询为默认。结果不
外推至 GPU/NCCL、多机、多 inflight，也不宣称新路径的完整额外成本已唯一分解。

### Follow-up：DECLARE 消息消融 F0–F4（2026-09-25）

本批只检验独立 DECLARE 对通信链关键路径和控制消息竞争的边际影响。`before-producer`（默认）执行
DECLARE→producer→submit/OFFER；`on-submit` 在 producer 完成后直接 submit，OFFER 仍携带完整 TaskSpec/TaskHint。
不改变任务、binding、预热、grant/全成员完成容量语义、消费或校验，不引入新协议；Lookahead/DAG 不允许该模式。

F0 真实双 rank CPU/Gloo 定向集成 10 passed、37 deselected，覆盖两模式元数据/顺序/数值、producer 未结束不 OFFER、
元数据冲突、缺失任务、launch 与 completion probe 失败的有界退出，以及 Lookahead/DAG 限制。全仓回归为
221 passed、51 skipped。F1、F2、F3、F4 共 90 次性能 replay 均正常结束并通过对应 validation。

F1 的 4 KiB×32 minimal 固定输入配对中，`on-submit−before-producer` makespan 差中位 −13.970 ms，五对范围
−45.260 至 −11.421 ms，5/5 更快。F2 是诊断观测，只用于机制；按每 run 先汇总再比较，wait-return→下一次 submit
中位减少约 540/286 µs（rank 0/1，均 5/5），coordinator 容量释放→最后 OFFER 入队减少 550 µs
（−690 至 −227 µs，5/5），容量释放→首次 eligible 减少 562 µs（−642 至 −171 µs，5/5）。但 rank 0 OFFER
发送锁等待中位增加约 147 µs（5/5），显示一部分竞争从独立 DECLARE 移到了更早到达的 OFFER；decision processing
没有同量级变化。诊断 trace 的绝对 makespan有明显观测扰动，未用于 F1/F3 性能比较，也未做固定成本扣除。

F3 的 1 MiB×32 minimal 配对中位差 −12.871 ms，范围 −41.512 至 −10.953 ms，5/5 更快。方向与 F1/F2 一致，
故按计划进入 F4；这些零计算链结果仍不构成多 seed 鲁棒性结论。

F4 使用 L0/L1、new FIFO、Phase A 相同 profile、0.3 execution jitter、1 ms completion poll 和 precreate；
五个 execution seed block，每 seed 三次 paired repeat，两模式顺序在 workload/seed/repeat 内随机化，合计 60 次串行
CPU/Gloo replay。60/60 validation 成功，任务、OFFER/SUBMITTED/COMPLETED 计数符合预期。30/30 配对块的 producer/
consumer execution samples 逐 rank/job 完全相同，且每块均与输入所需 16 个样本相符。

主要结果按 seed 内三次 repeat 的中位数后配对：

| 场景 | on-submit−before-producer makespan 差的 seed-block 中位数 | 五个 seed-block 差值范围 | on-submit 更快 |
| --- | ---: | ---: | ---: |
| L0-balanced | −0.354 ms | −0.963 至 +2.223 ms | 3/5 |
| L1-head-misalignment | +0.406 ms | −0.316 至 +1.461 ms | 1/5 |

五个 seed-block 中位差完整值（同为 `on-submit−before-producer`）：L0 `+2.223, −0.901, +1.705, −0.963,
−0.354 ms`；L1 `+0.044, +0.406, +1.080, +1.461, −0.316 ms`。这避免只凭总体中位数掩盖方向不一致。

每个场景 15 个 repeat 配对的范围分别为 −4.748 至 +4.423 ms、−4.788 至 +4.018 ms；差值方向不稳定且中心效应
小于运行波动。process CPU time 中位配对差约 −2.4 ms（两 workload 均 13/15 对 on-submit 较低）；voluntary
context switch 中位减少 45/47 次。逐 job JCT 有正有负，不能把资源开销下降解释为应用收益。

结论：DECLARE 的边际成本在零计算通信链上可测，直接提交减少控制消息与 CPU/线程调度活动；F4 没有证明其能在
L0/L1 上稳定降低应用 makespan。维持 `before-producer` 默认，不据此扩展策略矩阵或改默认。结果限于固定 L0/L1
输入及本机 CPU/Gloo，不推及其他 workload、GPU/NCCL 或多机。

F4 原始结果、manifest、运行 ledger、逐任务表、seed 配对 makespan/JCT、资源配对统计和样本一致性记录保存在本地
忽略的 [`F4 批次目录`](../../../benchmark/phase3/results/runtime-overhead/control-path/20260925-F-declaration-mode/application-L0-L1/)。
每个 workload 的图：[L0 makespan](../../../benchmark/phase3/results/runtime-overhead/control-path/20260925-F-declaration-mode/application-L0-L1/figures/L0-balanced/makespan-comparison.svg)、
[L1 makespan](../../../benchmark/phase3/results/runtime-overhead/control-path/20260925-F-declaration-mode/application-L0-L1/figures/L1-head-misalignment/makespan-comparison.svg)。

### 2026-09-25：E2 trace 复核与 DECLARE 补充测量

E2 的 20 条已有 raw trace 全部校验通过，本次没有重跑 E2。统计先在每个 run 内汇总 31 个任务转移，再比较
五个 run 的中位数和范围；表中 Task P90 是五个 run 内 P90 的中位数，括号为最大 repeat P90。时钟均为
coordinator 本地时钟，单位 µs。

| Coordinator 区间 | 4 KiB × 32 | 1 MiB × 32 |
| --- | ---: | ---: |
| 上项最后 COMPLETED 入队 → 容量释放 | 147 [143–160]；P90 273 (331) | 149 [141–155]；P90 286 (319) |
| 容量释放 → 最后成员 OFFER 入队 | 302 [248–413]；P90 625 (652) | 296 [280–401]；P90 665 (780) |
| 最后 OFFER 入队 → 首次 eligible | 104 [101–127]；P90 186 (232) | 101 [97–135]；P90 178 (207) |
| 容量释放 → 首次 eligible | 415 [351–546]；P90 740 (751) | 420 [395–503]；P90 731 (866) |
| 两条件满足 → decision 开始 | 3 [3–4]；P90 4 (4) | 4 [3–4]；P90 4 (5) |
| decision processing | 30 [30–31]；P90 37 (39) | 32 [29–32]；P90 40 (41) |
| decision 结束 → writer queue 入队 | 7 [7–7]；P90 9 (9) | 7 [7–8]；P90 8 (9) |
| writer queue 入队完成 → sendall 开始 | 93 [89–99]；P90 150 (180) | 95 [92–101]；P90 171 (175) |
| sendall 开始 → 结束 | 80 [64–116]；P90 174 (198) | 110 [85–112]；P90 166 (212) |

容量释放时至少一个 OFFER 尚未入队的转移分别为 141/155 和 137/155；两条 OFFER 都已入队但接收事件循环
尚未处理齐的情况只有 14/155 和 18/155。两种输入均没有出现候选先于容量释放 eligible（0/155）。因此
主要反复门控是等最后一个成员 OFFER 到达 coordinator 输入队列，其次是入队后接收循环的少量积压。最后
OFFER 入队时间不是网络单向延迟，它还受成员何时推进到 DECLARE/producer/submit 及中央接收线程调度影响。

E2 原 rank trace 没有 DECLARE 调用边界，故以单独的 E1 补充批次测量：4 KiB × 32、new Static FIFO/precreate、
1 ms polling、minimal/diagnostic 各 5 次交错。10/10 Gloo 运行成功并验证。diagnostic 相比 minimal 的
makespan 中位数高 12.391 ms，配对 repeat 差中位数 +8.919 ms（4/5 更慢）；两 rank process CPU 中位数
分别 164.807/145.679 ms。新增诊断有实质观测扰动，不将其作为无扰动性能基线，也不减去固定“日志成本”。

rank-local 配对统计同样先逐 run 汇总。rank 0/1 的 DECLARE 调用中位数为 319/170 µs，发送锁等待为
69/60 µs；等待与前一项 COMPLETED 发送持锁区间经常重叠。OFFER 发送锁等待中位约 1 µs。OFFER 发送结束
到 grant 接收为 826/1396 µs，grant 接收到 launch worker dequeue 为 176/185 µs，dequeue 到 collective
调用为 12/11 µs。E2 的最后到达 OFFER rank 会随消息大小变化，不能判定某 rank 恒慢。E2 完成观测区间约
1.08–1.17 ms（4 KiB）及 1.81–2.15 ms（1 MiB），包含 backend 等待与轮询，不等于精确物理完成延迟。

双 rank 时间线按 rank 分面、各自以本地 application release 为原点，没有跨 rank 相减：
[`配对时间线 SVG`](../../../benchmark/phase3/results/runtime-overhead/instrumentation/20260925-E1-declare-send-lock/figures/paired-rank-timeline-repeat0.svg)。
完整分段表、运行级范围和解释见
[`follow-up analysis`](../../../benchmark/phase3/results/runtime-overhead/instrumentation/20260925-E1-declare-send-lock/followup-analysis.md)。
停止条件已达到：最大重复等待位于成员 OFFER 到达/应用推进一侧；条件满足后的 policy、中央处理和本地 launch
等待较短。本次未实施优化，也未启动额外 E3/E4 或策略矩阵。
