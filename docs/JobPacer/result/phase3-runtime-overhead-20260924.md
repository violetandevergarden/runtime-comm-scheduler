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
