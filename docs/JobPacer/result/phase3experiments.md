# JobPacer Phase 3.1 / 3.2 CPU 实验结果

日期：2026-09-22。本文只记录本次实际执行的 CPU/Gloo pilot，不把单次运行写成稳定性能结论。

> 目录迁移说明（2026-09-24）：本报告中的旧路径、命令和实验事实保留不变。compact 结果现按语义类别保存，suite
> 索引见 [`benchmark/phase3/results/suites/20260923-compact/`](../../../benchmark/phase3/results/suites/20260923-compact/)，
> 旧新路径及迁移前摘要见 [`migration-map.json`](../../../benchmark/phase3/results/migration-map.json)。

## 执行环境与输入

- CPU：Intel Core i7-14650HX，12 physical cores / 24 logical CPUs，1 NUMA node。
- OS：WSL2，Linux 6.18.33.2；Python 3.13.15；PyTorch 2.13.0+cu129；Gloo 可用。
- world size：2；backend：Gloo；dtype：float32；reduction：sum；新 runtime `max_inflight=1`。
- profile：`/tmp/jobpacer-phase3-profile.json`，4 KiB、warmup 5、measurement 30，group `[0, 1]`。
  实测 p10/p50/p90 为 `0.0005807/0.0006600/0.0008042 s`；策略使用冻结 p50。
- 线性运行固定 `epoch=0`、`compute_jitter=0.3`、`wait_budget_s=0.02 s`、completion poll `1 ms`，
  旧路径和新路径使用同一 workload、profile 与确定性 producer/consumer 样本。
- DAG 运行使用现有 `benchmark/phase3/{linear,diamond,multi-group}.json`，同样使用
  `compute_jitter=0.3`、`epoch=0`。没有把这些机制样例扩写成 G0–G4 的正式参数化矩阵。

新增批处理入口为 `examples/jobpacer/run_phase3_experiments.py`，每次运行保存 command、退出码、
stdout/stderr、原始 JSON，并生成 `manifest.json`、`runs.jsonl` 和 `summary.csv`。本次产物位于：

- `/tmp/jobpacer-phase3-batch-final-20260922/`：线性 8 组；
- `/tmp/jobpacer-phase3-dag-linear-20260922/`、`/tmp/jobpacer-phase3-dag-diamond-20260922/`、
  `/tmp/jobpacer-phase3-dag-multigroup-20260922/`：每个 DAG 五策略。

## 线性共同子集

下表为 `summary.csv` 中的 rank-local job duration 最大值定义的 makespan；每个配置只有一个
seed/repeat，单位秒。新 runtime 的“平均/最慢 job JCT”列实际来自 rank-local job duration，
不是跨 rank 同步 release 后的严格 JCT，不能据此声称全局时钟跨度。所有 8 条运行 `status=ok`，
tensor 校验与 runtime validation 均通过。

| 组 | policy | makespan | 平均 job JCT | 最慢 job JCT |
| --- | --- | ---: | ---: | ---: |
| B0 old bare | fifo label | 0.013782 | 0.012656 | 0.013638 |
| B1 old scheduler | fifo | 0.016161 | 0.015114 | 0.015989 |
| B2 old scheduler | ltf | 0.013796 | 0.013095 | 0.013672 |
| S0 new static | static_fifo | 0.244383 | 0.220251 | 0.244383 |
| S1 new static | static_ltf | 0.241855 | 0.219806 | 0.241855 |
| D0 new dynamic | fifo | 0.249703 | 0.225205 | 0.249703 |
| D1 new dynamic | ltf | 0.240584 | 0.219778 | 0.240584 |
| D2 new dynamic | lookahead | 0.241002 | 0.219916 | 0.241002 |

该表只能说明本次机器和小消息 pilot 的一次观测。新旧路径的完成探测、控制协议和容量实现不同，
因此 B0/B1/B2 与 S/D 的差值是整体系统差异，不能归因成 policy 收益；S/D 之间也没有足够
重复次数支持显著性或稳定排序。新路径本次明显慢于旧路径，记录为控制/完成路径开销信号，后续需
用独立 profile、重复和更长通信重新确认。

## DAG 五策略 pilot

下表为三个现有 DAG 样例的一次五策略运行，列出 makespan 秒；每行 5/5 成功。

| DAG | static_fifo | static_ltf | fifo | ltf | lookahead |
| --- | ---: | ---: | ---: | ---: | ---: |
| linear | 0.183963 | 0.181333 | 0.190094 | 0.177208 | 0.140602 |
| diamond | 0.093985 | 0.093039 | 0.094469 | 0.099905 | 0.089654 |
| multi-group | 0.280708 | 0.285452 | 0.267401 | 0.289098 | 0.233908 |

这些数字不是跨图比较的性能结论。机制 trace 提供了以下可复核事实：

- `multi-group/static_fifo` 首次运行记录 `STATIC_HEAD_BLOCKED ≈ 0.05305 s`，实际 dispatch
  序列保持静态队首；
- `multi-group/static_ltf` 首项为 `job-1/comm-c0`，没有跳过 group 内顺序；
- `multi-group/lookahead` 记录 `ACTIVE_LOOKAHEAD ≈ 0.000220 s`，并保留容量满区间；
- 三种 DAG 的每个策略均检查了预期节点/任务集合、每个 rank 的 grant/launch 投影和 all-reduce
  结果，未发现顺序或数值错误。

## 验证命令与结果

实验支撑改动后的针对性检查：

```text
PYTHONPATH=src pytest -q tests/unit/runtime tests/unit/test_jobpacer_comm_profile.py
41 passed
```

最终工作树全仓检查：

```text
PYTHONPATH=src pytest -q
170 passed, 35 skipped

env PYTHONPATH=src RUN_JOBPACER_RUNTIME_REPLAY=1 \
  pytest -q tests/integration/test_runtime_replay.py
30 passed in 85.46s
```

其中 35 个 skipped 包括默认关闭的本地 TCP replay 参数和不可用硬件路径；开启 opt-in 后的
真实双 rank 集成以及故障注入均通过。实验批次完成线性 8 条和 DAG 15 条真实 Gloo replay，
均为退出码 0、validation ok。

## 已完成的实验支撑改动

- 新 runtime 支持严格匹配的 `--comm-profile`，并在结果中记录 profile digest、环境和估值来源。
- 新旧线性 replay 都支持固定 `--compute-jitter` 与 `--epoch`；样本键包含 workload seed、epoch、
  job、communication、rank 和 producer/consumer segment，实际样本不会写入 policy hint。
- 新 runtime 暴露并记录 `--wait-budget-s`。
- 批处理入口顺序运行新旧线性组或同一 DAG 的五策略，失败运行保留原始输出，不从统计中静默删除。
- 新增确定性样本键回归检查；没有引入额外调度抽象或依赖。

## 未验收范围

本次没有完成：4 KiB/1 MiB/16 MiB 多尺度 profile、单 job isolated 分母、L0–L5 正式多 seed
矩阵、30 seed × 3 repeat 主结果、bootstrap 区间、rank-specific L5 straggler 输入、G0–G4
正式 DAG 参数化输入、一般 DAG bare、GPU/NCCL、跨 host、多 inflight、多资源和长期性能研究。
因此本文不回答“动态策略稳定优于静态”或“新 runtime 整体优于旧系统”。

## 完成反馈延迟修正复测（2026-09-22）

旧 pilot 原始 trace 显示，`completion_observed`/`completed_sent` 到 coordinator 收齐
`COMPLETED` 之间存在约 40--48 ms 的空档；本地 collective launch 到完成观测只有约 1.5--2.7 ms。
将逐通信 tensor 校验后置后该空档仍存在，线程数对照也未改变结果。检查控制通道后发现 client
和 server socket 均未设置 `TCP_NODELAY`；修正后同时保留 tensor 引用，将 correctness scan
统一移动到 `runtime.finish_epoch()` 之后，并增加 application release、drain、validation
边界和统一 `performance` 汇总字段。

修正后的单次 CPU/Gloo 复测：

- 命令：`run_runtime_replay.py --policy ltf --workload balanced --backend gloo --world-size 2 --compute-jitter 0.3 --comm-profile /tmp/jobpacer-phase3-profile.json`。
- 产物：`/tmp/jobpacer-phase3-followup-ltf.json`。
- `validation=ok`，两 rank 的 all-reduce 结果正确。
- 六项通信的 coordinator `all_submitted_to_all_completed` 为约 1.31--2.05 ms，中位数约 1.56 ms。
- application makespan 为 23.018 ms，communication drain makespan 为 23.943 ms；时间均按各 rank 本地单调时钟先求差，再取成员最大值。

该复测只证明本次完成反馈异常在同配置下消失，不替代上面的单次 pilot，也不支持动态策略性能收益结论。
正式 30 seed × 3 repeat、多尺度 profile、isolated 基线和持久化正式批次仍未完成；后续性能结果
必须基于修正后的代码重新生成。修正后的有限 block 重复见下节，不替代正式统计设计。

## 修正后小批重复

为检查修正后的测量链路，运行了 5 个新 runtime 策略、`seeds=0,1`、每个 seed 两次 repeat，
共 20 条真实 CPU/Gloo replay。产物目录为
`/tmp/jobpacer-phase3-followup-small-20260922/`；20/20 退出成功且 validation 为 `ok`。
该批次使用 `--order-seed 20260922` 按 `(seed, repeat)` 随机化策略 block 顺序，manifest schema
为 2，并保存 17 个相关源码文件的 SHA-256 摘要；source snapshot digest 为
`489f2e6c5b33a88ce5d1281cbc3d3d4c010df2a6d3afdc2282a6b46560c345b0`。

每个策略有 4 个样本，统一 application makespan 的中位数（秒）为：

| policy | 样本数 | makespan 中位数 |
| --- | ---: | ---: |
| static_fifo | 4 | 0.023292 |
| static_ltf | 4 | 0.021082 |
| fifo | 4 | 0.020799 |
| ltf | 4 | 0.020838 |
| lookahead | 4 | 0.021206 |

该小批只验证修正后的反馈、计时和 block 产物链路，不足以证明策略排序、统计显著性或新 runtime
相对旧路径的收益；正式矩阵、isolated 分母、多尺度 profile 和 DAG 参数化实验仍未完成。

## 多尺度校准与正式输入 pilot（2026-09-22）

本节追加本轮真实 CPU/Gloo 执行结果。所有数字均为 pilot 观测，不构成动态策略收益结论。

环境仍为 Intel Core i7-14650HX、WSL2 Linux 6.18.33.2、Python 3.13.15、PyTorch
2.13.0+cu129、world size 2、Gloo、float32 sum；CPU affinity 为 0–23，线程环境变量未固定。

### Profile 与校准

命令使用 `run_comm_profile` 对
`benchmark/phase3/experiments/calibration/multi-scale.json` 执行 warmup 5、测量 30。
profile 产物为 `/tmp/jobpacer-phase3-calibration-profile-20260922.json`，SHA-256 为
`9fe7de4d1c710befcef2ddadd33ad1ef910b18beec9721f98f4d0e0c647aba3f`。

| 消息大小 | p10 | p50 | p90 |
| ---: | ---: | ---: | ---: |
| 4 KiB | 0.512 ms | 0.577 ms | 0.696 ms |
| 1 MiB | 2.419 ms | 3.268 ms | 7.302 ms |
| 16 MiB | 19.185 ms | 22.180 ms | 24.764 ms |

每个签名 30 个样本。`two-groups.json` 校准中，bare workload makespan 为 26.567 ms，
旧 scheduler 串行 workload makespan 为 43.936 ms，二者 validation 均为 `ok`；新 runtime
五策略校准也为 5/5 成功。由于每项只有一次配置观测，这里只记录口径和量级。

### 机制 pilot

使用 seed 0、每策略 5 repeats、严格 profile 和 replay warmup。L0–L5 与 G0–G4 的结果目录
为 `/tmp/jobpacer-phase3-mechanism-pilot-20260922/`；每个场景为 25 次新 runtime replay，
L3 另有一次重跑目录 `L3-lookahead-rerun/`。

| 场景 | 成功 | 关键证据 |
| --- | ---: | --- |
| L0 | 25/25 | 正确性通过；不是机制收益样本 |
| L1 | 25/25 | static FIFO/LTF 均 5/5 观察到静态队首阻塞 |
| L2 | 25/25 | 25/25 存在至少两个 simultaneously eligible 候选；gate-first 依策略为 0–5/5 |
| L3 | 25/25（重跑） | Lookahead wait 1/5，触发不稳定 |
| L4 | 25/25 | Lookahead wait + deadline fallback 1/5，触发不稳定 |
| L5 | 25/25 | 各策略 OFFER spread ≥5 ms 的次数为 4–5/5 |
| G0/G1/G3 | 各 25/25 | DAG 节点集、join/顺序和 tensor 校验通过 |
| G2 | 25/25 | 25/25 存在至少两个 simultaneously eligible 候选；gate-first 依策略为 0–5/5 |
| G4 | 25/25 | Lookahead wait 且无提前 unsafe 预测 1/5 |

L3 初始批次另有 1 次环境失败：rank 0 报 `Address already in use`，rank 1 控制连接关闭；
原始记录保留，重跑成功。没有发现 collective、协议或数值错误。

G4 的机制摘要在本轮修正：只有 `unsafe` 预测早于同一 coordinator rank 观测到
`job-1/current` 完成才记作违规。此前的简单字符串检查会把完成后的安全 frontier 误报为
违规；对应回归测试已加入，代码修改不改变 runtime 调度语义。

### Isolated pilot

`job-0`、`job-1` isolated 各 25/25 成功，L0 shared batch 25/25 成功；产物位于
`/tmp/jobpacer-phase3-isolated-pilot-20260922/`，`jobs.csv` 已填充逐 job denominator 和
slowdown。两 job、5 repeat 合并后的 slowdown median 为：static FIFO `0.557`、static LTF
`0.536`、FIFO `0.510`、LTF `0.571`、Lookahead `0.530`。该结果只有一个 seed，且仍是
小消息/单机 pilot，不解释为整体收益。

### 当前状态

本轮已完成：多尺度 profile、bare/旧 scheduler/新 runtime 校准、L0–L5/G0–G4 机制 pilot、
L0 isolated 分母和逐 job slowdown。L1/L2/L5、G0/G1/G2/G3 可进入下一轮 screening；
L3/L4/G4 的等待/预测触发率不足，需先调整场景窗口或作为负对照处理。正式多 seed/repeat
矩阵、跨阶段共同对照、持久化原始 trace 归档和性能收益结论仍未完成。

## Screening（2026-09-22）

按照 `benchmark/phase3/experiments/suites/screening.json`，对 L0/L1/L2/G0/G1/G2 执行
`seeds=100..109`、每 seed 一次、五个新 runtime 策略的配对 screening。总计 300 次真实
CPU/Gloo replay，六个场景均为 50/50 `validation=ok`。原始产物和汇总位于
`/tmp/jobpacer-phase3-screening-20260922/`。

以下为相对 `new-static_fifo` 的 makespan median speedup；括号为 2000 次 seed-block
bootstrap 95% 区间，>1 表示候选较快：

| 场景 | static LTF | FIFO | LTF | Lookahead |
| --- | ---: | ---: | ---: | ---: |
| L0 | 1.033 (0.980–1.184) | 1.067 (0.960–1.170) | 1.074 (0.944–1.234) | 1.104 (1.005–1.229) |
| L1 | 1.011 (0.910–1.055) | 1.082 (0.998–1.183) | 1.192 (1.100–1.265) | 1.086 (0.963–1.202) |
| L2 | 0.977 (0.863–1.034) | 1.051 (0.914–1.144) | 1.150 (1.059–1.257) | 1.071 (1.009–1.188) |
| G0 | 1.689 (1.624–1.818) | 1.723 (1.551–1.891) | 1.703 (1.616–1.779) | 1.360 (1.258–1.713) |
| G1 | 1.008 (0.925–1.104) | 1.010 (0.800–1.274) | 0.981 (0.873–1.127) | 0.932 (0.751–1.081) |
| G2 | 1.368 (1.296–1.498) | 0.983 (0.938–1.285) | 1.037 (0.948–1.337) | 0.951 (0.937–0.988) |

这是新 runtime 内部 screening，不包含 Phase 1/2；不同场景方向不一致，不能据此声称动态
策略普遍优于静态。G1 区间跨 1，G2 中 Lookahead 的 screening 结果低于 1，而 G0 的
相对差异较大，均需在统一旧新口径、更大 repeat 和持久化原始 trace 后再解释。

L3/L4/G4 未进入本轮 screening，因为 5-repeat pilot 中 Lookahead wait/fallback 仅 1/5
次，尚未稳定触发预期机制；L5/G3 也尚未做配对 screening。正式 `30 seed × 3 repeat`
主结果和完整旧新共同对照仍未完成。

### 2026-09-23 解释勘误（无新运行）

上述六场景 screening 的 `compute_jitter=0`，seed block 没有产生不同的计算时长
样本。配对 bootstrap 区间描述固定输入下的重复运行差异，不能解释为对计算扰动
的鲁棒收益。

L2/G2 的 gate-first 不是各策略共同前提：screening 中 Static LTF 为 0/10，
FIFO 分别为 6/10、7/10，LTF 分别为 5/10、6/10（L2、G2 顺序）。
旧的 `max_simultaneous_eligible >= 2` 没核对两个指定研究候选是否在 gate 完成前
eligible。因此这些 makespan 差异包含 gate 调度和后续候选选择；旧批次保留
全部运行，不从中事后筛 gate-first 子集估计收益。

G0 原输入缺少与 L0 相同的 consumer overlap 和 jitter 采样映射；其 Static FIFO
实际按 job 顺序执行，约 1.7× 的相对差异包含冻结顺序质量。G1 仅一 job、join
之后才有第二项通信，作为负对照解读。此前 isolated 分母与 shared 分批执行，
shared JCT 约 24–27 ms、isolated 约 40–51 ms 的反向差异尚未查明；
既有约 0.5 的 slowdown 不作为资源竞争效应或正式结论。L3/L4/G4 原等待仅 1/5
也归因于待验证的预测窗口设计，而非单纯重复不足。相应输入和检查已修改，
新结果需重新运行才能评价。

## 精简 suite 首次执行（2026-09-23）

按 `benchmark/phase3/experiments/suites/compact.json` 运行。持久化产物位于
`benchmark/phase3/results/phase3-compact-runs-20260923/`；校准 profile 位于
`benchmark/phase3/results/phase3-compact-20260923/calibration/profile.json`，SHA-256
为 `0283de537d286411b5014ca5436b6284c748fecc122360c8d1dfd8ab42573093`。

环境：Intel Core i7-14650HX、WSL2 Linux 6.18.33.2、Python 3.13.15、PyTorch
2.13.0+cu129、CPU/Gloo、world size 2；CPU affinity `0-23`，OMP/MKL/OpenBLAS/NumExpr
线程均为 1。profile warmup 5、每个签名采样 30 次：4 KiB、1 MiB、16 MiB 的
p10/p50/p90 分别为 `0.705/0.902/1.236 ms`、`1.868/2.552/3.654 ms`、
`18.097/19.566/21.658 ms`。suite source snapshot digest 为
`2a206ce4bfda9f7665e62a938b7279a986edf0b0db07ca48e1083ec3c0c5e884`。

共检查 660 次真实 replay，所有结果 JSON 的 SHA-256 与运行记录一致，且
validation 均为 `ok`：机制验收 90 次、主矩阵 480 次、isolated 诊断 30 次、
独立噪声 pilot 60 次。compact 基础计划的 780 次中，L2/G2 主矩阵共 180 次未扩批：
L2 和 G2 的 FIFO/LTF 各自严格 gate-first 竞争均为 `0/5`；指定研究候选虽在
gate 完成前 OFFER 到齐，但首次研究候选决策时没有形成共同 eligible 竞争。L3
准时 Lookahead 为 `0/5`，因此可选 120 次性能批次未运行；L4 与 G4 的 deadline
fallback 各为 `5/5`。L5 成员 OFFER spread 机制判定为 `10/10`；G0 线性与 DAG
桥接各 `5/5` 成功，计算样本/launch 顺序桥接检查为 `ok`。未达到门槛的结果保留，
没有从任何性能汇总中筛除运行。

噪声 pilot 使用 L0、零计算扰动、单个 `new-fifo` arm；10 个 seed、每 seed 3 次，
执行两份 A/B 并按 seed 奇偶交替先后顺序。每份内部先对 repeat 取中位数，再比较
同 seed 的 A/B，10 个 seed-block 的绝对相对差中位数为 `5.99%`，P95 为 `11.96%`。
该 P95 作为主矩阵显式 `tie_threshold=0.1196095405`。噪声幅度较大，胜/平/负标签
应谨慎解读；此阈值不是性能显著性检验。

主矩阵按 `seeds=4000..4009`、每 seed 3 repeats、jitter `0.3`、wait budget
`20 ms` 执行。L0 与 L1 各 210 次，L5 为 60 次，全部成功。主要配对结果为
baseline/candidate makespan 比值；大于 1 表示 candidate 较快，区间为 2000 次
seed-block bootstrap 95% 区间：

| 场景 | baseline → candidate | speedup（95% bootstrap CI） |
| --- | --- | ---: |
| L0 | old bare → old FIFO | 0.872 (0.846–0.917) |
| L0 | old FIFO → new static FIFO | 0.662 (0.613–0.687) |
| L0 | old LTF → new static LTF | 0.670 (0.641–0.704) |
| L0 | new static FIFO → new FIFO | 1.045 (0.999–1.100) |
| L0 | new FIFO → new LTF | 1.003 (0.951–1.046) |
| L1 | old bare → old FIFO | 0.811 (0.783–0.863) |
| L1 | old FIFO → new static FIFO | 0.750 (0.725–0.799) |
| L1 | old LTF → new static LTF | 0.775 (0.746–0.788) |
| L1 | new static FIFO → new FIFO | 1.224 (1.198–1.283) |
| L1 | new FIFO → new LTF | 1.000 (0.986–1.028) |
| L5 | new static FIFO → new FIFO | 1.200 (1.154–1.269) |

这是通过机制筛选后的部分矩阵，不包含 L2/G2 主性能批次或 Lookahead 扩批；结合
较高的零扰动噪声，不据此概括动态策略的普遍收益。跨 Phase 1/2/3 的 makespan
均取结果 JSON 的 `performance.workload_makespan_us`，rank-local duration 先在
本地求差再取 rank 最大值；它比较的是整条执行路径，不单独归因于调度策略。

interleaved isolated 的 5-repeat slowdown 中位数（shared job JCT / 对应 isolated
job JCT）为：FIFO 的 job-0 `1.366`、job-1 `1.095`；LTF 的 job-0 `1.300`、job-1
`1.151`。该诊断只有一个 seed，不解释为稳定资源竞争效应。原始 JSON、manifest、
逐 job/机制记录、配对 CSV 和噪声 pilot 分批摘要均留在上述结果目录。
