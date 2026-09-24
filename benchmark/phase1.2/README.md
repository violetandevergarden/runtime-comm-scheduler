# JobPacer Phase 1/2 benchmark

本目录保存 JobPacer Phase 1/2 的实验输入、原始 trace、汇总数据和图表；执行入口在
`examples/jobpacer/scripts/run_phase1_2_experiments.py`。
当前正式实验运行在两 rank CPU/Gloo 环境，使用真实异步 `all_reduce`；结论不能直接外推到
GPU/NCCL 或真实训练任务。

## 1. 先看这里：最终实验

`results/` 按实验主题分类，batch 目录名是时间戳加随机哈希。最终结果只有下面三组，建议按表中顺序阅读。
每个 batch 的 `manifest.json` 是唯一的样本索引，不要通过扫描目录自行收集
`run-*.json`。

| 顺序 | 最终实验 | Batch | 规模 | 首先看什么 |
| ---: | --- | --- | ---: | --- |
| 1 | **最终容量探索** | [`20260919T143653Z-9536db`](results/capacity/20260919T143653Z-9536db/) | 720/720 | [容量分析](results/capacity/20260919T143653Z-9536db/analysis.md)、[`capacity-makespan.svg`](results/capacity/20260919T143653Z-9536db/figures/capacity-makespan.svg)、[`capacity-observed-concurrency.svg`](results/capacity/20260919T143653Z-9536db/figures/capacity-observed-concurrency.svg) |
| 2 | **最终 0.1 ms 灵敏度复核** | [`20260920T015224Z-ec2fad-poll-sensitivity`](results/polling/20260920T015224Z-ec2fad-poll-sensitivity/) | 80/80 | [灵敏度分析](results/polling/20260920T015224Z-ec2fad-poll-sensitivity/analysis.md)、[`summary.svg`](results/polling/20260920T015224Z-ec2fad-poll-sensitivity/figures/summary.svg) |
| 3 | **最终 SRJF 策略实验** | [`20260920T021723Z-6f899d`](results/priority/20260920T021723Z-6f899d/) | 1040/1040 | [SRJF 分析](results/priority/20260920T021723Z-6f899d/analysis.md)、[`policy-makespan-by-capacity.svg`](results/priority/20260920T021723Z-6f899d/figures/policy-makespan-by-capacity.svg)、[`policy-mean-job-completion.svg`](results/priority/20260920T021723Z-6f899d/figures/policy-mean-job-completion.svg)、[`policy-paired-differences.svg`](results/priority/20260920T021723Z-6f899d/figures/policy-paired-differences.svg) |

三组实验的关系是：先用容量探索观察 FIFO/LTF 在 k1、k2、k3、unbounded 下的变化；再用
0.1 ms 批次复核 `overlap-window-long` 中接近探测分辨率的容量排序；最后在冻结容量
`[1,2,3,unbounded]` 下重新运行 bare、FIFO、LTF 和 SRJF。

### 1.1 Analysis 报告勘误

阅读现有 `analysis.md` 时必须使用以下口径：

1. **容量报告的 `Capacity interpretation` 数字被误标为 paired median。** 这段文字实际是
   “两个场景各自中位数之差”，不是“每个 repetition 先相减后所得差值的中位数”。例如
   comm-heavy 的 FIFO k2−k1，正确配对中位数是 `-70.719 ms`，该段文字写成
   `-72.357 ms`。同一 workload 前面的 `Paired differences` 表和
   `summary.json -> paired_comparisons` 是正确口径，应以它们为准。
2. **`Workload replay window` 不是单次 workload makespan。** 它是该 workload 全部场景 run
   从首个开始到最后一个结束的墙钟跨度。性能时间看 `workload_makespan_us`。
3. **表中的 `peak admission / launched` 是 rank/run 样本分布的中位数。** 它不是所有 rank
   的全局峰值，也不是链路带宽利用率。
4. **SRJF 报告的 `Static SRJF interpretation` 使用各场景独立中位数作描述。** 判断
   SRJF−FIFO 或 SRJF−LTF 的效果，仍应读取它前面的同 repetition `Paired differences`。
5. P10–P90 是 10 次运行的样本分布，不是策略收益的置信区间；completion timestamp 是
   probe observation，不是精确设备完成时刻。

修正报告生成器并重新 finalize 前，机器可读的 `summary.json -> paired_comparisons` 是正式的
比较数据源。

### 1.2 Smoke 和历史数据

Smoke batch 用于正确性和结束路径检查，不用于性能结论：

| 实验 | Batch | 规模 |
| --- | --- | ---: |
| 单 workload 容量 smoke | `20260919T131832Z-000d71` | 18/18 |
| 完整容量 smoke | `20260919T134222Z-afde4d` | 144/144 |
| 完整 SRJF smoke | `20260920T015844Z-fa107f` | 208/208 |

`results/baseline/historical/20260919T083615Z-10a786` 是容量扫描前的 8-workload、6-scenario
历史基线，`results/polling/20260919T091635Z-ceff56` 是它的 0.1 ms 探测灵敏度批次。更早的 batch 用于
测量语义和 runner 演进；其中 `20260919T030103Z-bc722c`、
`20260919T032406Z-ac8211`、`20260919T075437Z-511186` 的 manifest 为
`complete=false`，不得进入统计。

batch 机制启用前的三 workload 结果（raw、summary、analysis、figures、profiles）整体保存在
`results/baseline/legacy-three-workloads/`。完整旧新路径及迁移摘要见
[`results/migration-map.json`](results/migration-map.json)。

## 2. 目录结构

```text
benchmark/phase1.2/
  README.md
  experiments/
    baseline/ capacity/ polling/ priority/ # 分类说明；参数矩阵仍由 runner 管理
    shared/
      dag/                     # 早期 chain-DAG 输入
      workloads/               # JobPacer workload manifests
  results/
    README.md migration-map.json
    baseline/ capacity/ polling/ priority/
    baseline/legacy-three-workloads/ # 旧版、非 batch 结果
    <category>/<batch-id>/
        manifest.json          # 样本集合、场景、顺序、命令和哈希
        plans.json             # 静态 Plan、digest 和评分 diagnostics
        capacity-selection.json # 容量冻结记录；仅相关 batch 存在
        profiles/              # 本 batch 使用的通信 profile
        raw/<workload>/<scenario>/run-XX.json
        summary.json           # 从 manifest 样本派生的机器可读统计
        analysis.md            # 人类可读报告
        figures/               # 从同一 manifest 生成的 SVG
```

批次按完整目录迁移，不改写 batch 内文件。manifest 中的 batch 内相对路径继续有效；历史命令和旧路径由
[`results/migration-map.json`](results/migration-map.json) 追溯。移动后仍应先核对 manifest 哈希，再执行 finalize。

## 3. Benchmark 如何构造

### 3.1 Workload schema

`experiments/shared/workloads/*.json` 中每个 job 是一条线性通信链，每项 communication 主要包含：

| 字段 | 含义 |
| --- | --- |
| `id` | job 内从 0 连续递增的通信 ordinal |
| `num_bytes` | 每个 rank 上 collective tensor 的实际字节数 |
| `op` | 当前为 `all_reduce` |
| `producer_compute_s` | communication ready 前的应用计算 |
| `consumer_compute_s` | submit 返回后、应用 wait 前允许与通信重叠的计算 |
| `estimated_comm_s` | 静态 Plan 构造使用的离线通信估计，不控制真实通信时长 |
| `ranks` | job 的 ProcessGroup 成员；`null` 表示全部 rank |

零估计时长不表示删除通信；只要 communication 节点存在且 `num_bytes` 合法，就会执行真实
collective。

早期三个 chain-DAG 可以重新转换：

```bash
python examples/jobpacer/workload_builder.py \
  benchmark/phase1.2/experiments/shared/dag \
  benchmark/phase1.2/experiments/shared/workloads
```

转换器只接受互不相交的线性 chain，保留相邻和零时长 communication，并要求通信显式提供
`num_bytes`。

### 3.2 当前 8 个 workload

| Workload | 构造 | 主要观察问题 |
| --- | --- | --- |
| `comm-heavy` | 3 jobs × 4 次 16 MiB；每次前置计算 0.5 ms | 通信密集下容量增加是否带来并发收益 |
| `staggered-compute` | 3 jobs × 4 次 1 MiB；producer 窗口在 1/2/6/12 ms 间错峰 | 静态 Plan 队首未 ready 和评分假设偏差 |
| `long-short-chains` | 1 个 8-task 长 job + 3 个 2-task 短 job；均为 1 MiB | 长/短 job 完成时间、LTF/SRJF 取舍 |
| `mixed-message-sizes-large-first` | job-0 为 4 × 16 MiB，job-1/2 为 4 × 1 MiB | 大消息 job 的优先级及短 job 延迟 |
| `mixed-message-sizes-large-last` | job-0/1 为 4 × 1 MiB，job-2 为 4 × 16 MiB | job ID/Plan 位置变化是否改变策略结果 |
| `overlap-window-short` | 2 jobs × 4 次 1 MiB；consumer window = 0 | 无 consumer overlap |
| `overlap-window-medium` | 同上；consumer window = T | overlap 约等于 profile 的 1 MiB p50 |
| `overlap-window-long` | 同上；consumer window = 3T | 较长 overlap 是否掩盖容量限制 |

三个 overlap workload 共享 batch 内新测得的 `overlap-window-gloo.json`；workload 中的
consumer window 固定为记录源 profile 的 `0/T/3T`，不会根据本轮 profile 动态重写 workload。

### 3.3 通信 profile

每个新 batch 在测量窗口外为所需通信签名运行：

```text
3 次 warmup + 10 次正式测量
```

profile 从 collective API 调用开始，到同步确认完成结束；记录 p10、p50、p90、均值、标准差、
API call 时长和 return-to-sync 时长。Plan 使用匹配签名的 p50 作为
`estimated_comm_s`。profile 环境、文件哈希和规范化 JSON digest 均写入 manifest。

### 3.4 场景和策略

共同名称含义：

- `phase1_bare`：由 `examples/jobpacer/scripts/run_phase1.py` 裸发；不经过 scheduler、不排序、不限制
  outstanding。它是独立执行路径，不是 scheduler 的 unbounded 容量点。
- FIFO：按 communication ordinal 对 job 做确定性轮转，形成严格静态 Plan。
- LTF：每步从各 job 的下一项中选择 estimated remaining path 最大者。
- SRJF：使用同一分数选择最小者；静态、非抢占，且不能越过尚未 ready 的固定 Plan 头。
- `k1/k2/k3`：rank-local 最大 admission occupancy；`unbounded` 在 CLI 中表示为
  `max_outstanding=0`。

#### 最终容量探索：9 个场景

Batch：`20260919T143653Z-9536db`。

| 场景 | 执行路径 | 静态顺序 | 配置容量 | 主要比较目的 |
| --- | --- | --- | ---: | --- |
| `phase1_bare` | Phase 1 raw collective | runtime arrival | 不限制 | 不经过 scheduler 的端到端参考 |
| `fifo_k1` | scheduler | FIFO | 1 | FIFO 单在途基线 |
| `fifo_k2` | scheduler | 与 `fifo_k1` 相同 FIFO Plan | 2 | FIFO 从 1 增至 2 的并发收益 |
| `fifo_k3` | scheduler | 与 `fifo_k1` 相同 FIFO Plan | 3 | FIFO 从 2 增至 3 的边际收益 |
| `fifo_unbounded` | scheduler | 与 `fifo_k1` 相同 FIFO Plan | 0/unbounded | FIFO 不主动限制容量的参考 |
| `ltf_k1` | scheduler | LTF | 1 | LTF 单在途基线及同容量 FIFO 对照 |
| `ltf_k2` | scheduler | 与 `ltf_k1` 相同 LTF Plan | 2 | LTF 从 1 增至 2 的并发收益 |
| `ltf_k3` | scheduler | 与 `ltf_k1` 相同 LTF Plan | 3 | LTF 从 2 增至 3 的边际收益 |
| `ltf_unbounded` | scheduler | 与 `ltf_k1` 相同 LTF Plan | 0/unbounded | LTF 不主动限制容量的参考 |

同一策略的四个容量场景共享相同 Plan；因此容量曲线只改变配置容量和由此产生的实际并发。
FIFO 与 LTF 的同容量比较改变静态排序，不能再解释成纯容量效果。

#### 最终 0.1 ms 灵敏度复核：8 个场景

Batch：`20260920T015224Z-ec2fad-poll-sensitivity`。只运行
`overlap-window-long`，场景为：

```text
fifo_k1, fifo_k2, fifo_k3, fifo_unbounded
ltf_k1,  ltf_k2,  ltf_k3,  ltf_unbounded
```

这些场景与容量探索中的定义相同，但 completion poll 从 1 ms 统一改为 0.1 ms，并使用新的
seed 重新运行。该批次没有 `phase1_bare`、没有 SRJF，也不能与 1 ms 样本混合统计。

#### 最终 SRJF 策略实验：13 个场景

Batch：`20260920T021723Z-6f899d`。它读取冻结容量
`[1,2,3,unbounded]`，场景如下：

| 容量 | FIFO 场景 | LTF 场景 | SRJF 场景 | 比较方法 |
| --- | --- | --- | --- | --- |
| 1 | `fifo_k1` | `ltf_k1` | `srjf_k1` | 同为单在途，比较静态排序 |
| 2 | `fifo_k2` | `ltf_k2` | `srjf_k2` | 同配置 k2，比较策略及实际并发 |
| 3 | `fifo_k3` | `ltf_k3` | `srjf_k3` | 同配置 k3，比较策略及实际并发 |
| unbounded | `fifo_unbounded` | `ltf_unbounded` | `srjf_unbounded` | 都不主动限制容量，但实际峰值可能不同 |

另有共同参考 `phase1_bare`，合计 `1 + 4 × 3 = 13` 个场景。FIFO、LTF、SRJF 和 bare 都在
这个 batch 内重新运行；不能把容量 batch 中的旧样本与它配对。

“同配置容量”不表示“同实际并发”。严格静态 Plan 可能被尚未 ready 的队首阻塞，因此比较
SRJF/FIFO/LTF 时必须同时查看 `peak_admission_occupancy`、`peak_launched_inflight` 和
ready→admit。

#### 历史 6-scenario 基线

Batch `20260919T083615Z-10a786` 使用旧矩阵：

| 场景 | 含义 |
| --- | --- |
| `phase1_bare` | Phase 1 裸发 |
| `ready_first_unbounded` | scheduler 路径、全局 ready-first、不限制容量 |
| `ready_first_serial` | ready-first、容量 1，并带全局 completion coordination |
| `fifo_unbounded` | 静态 FIFO、不限制容量 |
| `fifo_serial` | 静态 FIFO、容量 1 |
| `ltf_serial` | 静态 LTF、容量 1 |

它用于早期拆分 scheduler 路径、单在途和静态队首效应，不是本轮最终容量/SRJF 数据。

### 3.5 LTF/SRJF 评分

LTF/SRJF 的共同分数为：

```text
remaining(i) = max(estimated_comm_i, consumer_compute_i)
             + sum(
                   producer_compute_j
                   + max(estimated_comm_j, consumer_compute_j)
                   for j > i
               )
```

当前候选的 producer 不计入分数，表示“候选已经 ready”的离线假设。每一步候选分数、选择、
tie-break 和最终 Plan 都保存在 `plans.json`。

### 3.6 重复与运行顺序

每个 repetition 内，全部场景按 `(seed, workload, repetition)` 做确定性 shuffle，随后逐个
replay 串行执行。不同 replay 不并发运行，避免人为制造额外 CPU/Gloo 竞争。manifest 保存
`round_orders`，每条 run record 也保存 `repetition` 和 `order_index`。

正式批次使用 10 次重复，属于探索性样本。P10–P90 是观察到的运行分布，不是策略收益的
置信区间。

## 4. 实验复现

以下命令从仓库根目录执行。

### 4.1 环境

运行实验需要项目本身和 PyTorch distributed；生成 SVG 还需要可选 Matplotlib 依赖：

```bash
python -m pip install -e '.[visualization]'
```

当前正式数据的环境、Python/PyTorch 版本、backend、world size、git commit、dirty 状态和相关
源码 SHA-256 记录在各 batch 的 `manifest.json -> metadata`。要做可比较的复现，应先核对这些
字段，而不是只复用相同命令。

### 4.2 先运行 smoke

容量 smoke：

```bash
PYTHONPATH=src:. python -m examples.jobpacer.scripts.run_phase1_2_experiments \
  --experiment capacity-scan \
  --repeats 2 \
  --experiment-seed 120914
```

runner 默认执行全部 8 个 workload。可以重复 `--workload` 做更小的检查：

```bash
PYTHONPATH=src:. python -m examples.jobpacer.scripts.run_phase1_2_experiments \
  --experiment capacity-scan \
  --repeats 2 \
  --workload comm-heavy \
  --workload staggered-compute
```

每次命令都会在 `results/baseline/<batch-id>/` 创建新 batch，不会覆盖已有结果。

### 4.3 复现容量正式实验

```bash
PYTHONPATH=src:. python -m examples.jobpacer.scripts.run_phase1_2_experiments \
  --experiment capacity-scan \
  --repeats 10 \
  --experiment-seed 120914
```

预期矩阵为 `8 × 9 × 10 = 720` 次 replay。主实验强制所有场景使用 1 ms completion poll。

### 4.4 冻结容量并复现 SRJF

本轮冻结记录位于：

```text
results/capacity/20260919T143653Z-9536db/capacity-selection.json
```

它绑定容量 batch ID 和 `summary.json` SHA-256，冻结容量为 `[1,2,3,0]`，其中 `0` 表示
unbounded。

```bash
PYTHONPATH=src:. python -m examples.jobpacer.scripts.run_phase1_2_experiments \
  --experiment srjf \
  --capacity-selection \
    benchmark/phase1.2/results/capacity/20260919T143653Z-9536db/capacity-selection.json \
  --repeats 10 \
  --experiment-seed 120915
```

预期矩阵为 `8 × 13 × 10 = 1040` 次 replay。FIFO、LTF、SRJF 和 bare 都在这个新 batch 中
重新运行；不能从容量 batch 复制样本做配对。

如果从一个新容量 batch 重新开始，应先分析完整容量曲线，再创建属于该 batch 的
`capacity-selection.json`。不要把旧选择文件改写为指向新 summary，也不要为不同策略各选
不同容量后声称是纯排序比较。

### 4.5 复现 0.1 ms 灵敏度实验

```bash
PYTHONPATH=src:. python -m examples.jobpacer.scripts.run_phase1_2_experiments \
  --poll-sensitivity-source-batch \
    benchmark/phase1.2/results/capacity/20260919T143653Z-9536db \
  --poll-sensitivity-workload overlap-window-long \
  --repeats 10 \
  --experiment-seed 143654
```

不指定 `--poll-sensitivity-pair` 时，runner 使用该实验预设的 FIFO/LTF 容量比较，共涉及
8 个场景和 80 次 replay。也可以重复传入显式配对：

```bash
--poll-sensitivity-pair fifo_k2:fifo_k1 \
--poll-sensitivity-pair ltf_k2:ltf_k1
```

灵敏度 batch 中所有被比较场景统一使用 0.1 ms，不与 1 ms 主批次合并统计。

### 4.6 核验并重新生成派生产物

已有容量/SRJF batch 可以从 manifest 所列 raw trace 重新核验并生成 summary、analysis 和图：

```bash
PYTHONPATH=src:. python -m examples.jobpacer.scripts.run_phase1_2_experiments \
  --finalize-batch \
  benchmark/phase1.2/results/<category>/<batch-id>
```

该命令会验证 workload/profile/Plan/trace 哈希、场景笛卡尔积、执行顺序、trace 时间边界、
tensor validation 和有限容量约束，然后重写派生文件及其 artifact 哈希。它不会重新执行 raw
replay。不要对不完整或打算原样封存的 batch 随意运行 finalize。

单条 replay 的完整命令保存在：

```text
manifest.json -> runs[] -> command
```

这是复现某个异常样本时最准确的入口。

## 5. 数据文件和字段含义

### 5.1 `manifest.json`

| 字段 | 含义 |
| --- | --- |
| `complete` | 只有 run 笛卡尔积完整且 finalize 成功后才为 true |
| `experiment` | `capacity-scan`、`srjf` 或 `poll-sensitivity` |
| `scenario_order` / `scenarios` | 场景显示顺序及 mode、policy、capacity、Plan digest |
| `round_orders` | 每个 workload/repetition 的实际随机交错顺序 |
| `runs` | 唯一原始样本索引；包含命令、路径、状态、输入 digest 和 trace SHA-256 |
| `profiles` / `workload_profiles` | profile 文件、环境匹配和哈希 |
| `artifacts` | summary、analysis、plans 和 figures 的 SHA-256 |
| `representative_traces` | timeline 为每个场景自动选择的原始 trace |
| `metadata` | 环境、poll interval、git 和源码/workload 哈希 |

`complete=true` 和哈希匹配证明样本集合与文件完整，并不自动证明统计结论具有普适性。

### 5.2 `raw/.../run-XX.json`

每个文件是一轮完整两-rank replay，主要分为：

- `config`：mode、policy、scenario、capacity、profile 和 poll interval；
- `ranks[]`：每个 rank 的 run 边界、job/task trace、launch sequence、control telemetry；
- `performance`：workload/job makespan、run 内平均 job completion、容量观测；
- `validation`：tensor 正确性、Plan/group 顺序、容量和时间边界检查。

重要 run 级指标：

| 指标 | 定义 |
| --- | --- |
| `workload_makespan_us` | 各 rank 从统一 application release 到 application end 的最大值 |
| `job_makespans[].makespan_us` | 该 job 所有参与 rank 的 application completion 最大值 |
| `mean_job_completion_us` | 先在本次 run 内对全部 job completion 求算术平均 |
| `communication_drain_makespan_us` | 从 release 到全部通信完成被 observer 观察到 |
| `validation_total_us` | drain 后 tensor 正确性校验时间，不进入应用 makespan |
| `harness_total_us` | 测试 harness 的完整耗时 |

关键 task 时间戳：

```text
producer_compute_start_ts -> ready_record_ts
submit_api_start_ts       -> submit_api_return_ts
admit_ts
collective_call_start_ts  -> collective_call_return_ts
completion_observed_ts
consumer_compute_start_ts -> consumer_compute_end_ts
application_wait_start_ts -> underlying_wait_start_ts -> wait_return_ts
application_task_end_ts
validation_start_ts       -> validation_end_ts
```

`completion_observed_ts` 是 completion probe 首次观察到完成的时间，不是精确设备完成时间。
主批次默认轮询间隔为 1 ms，因此接近该尺度的差值需要单独做一致的 0.1 ms 复核。

### 5.3 容量观测

每个 rank 使用半开区间重建：

```text
admission occupancy: admit_ts <= t < completion_observed_ts
launched inflight:   collective_call_start_ts <= t < completion_observed_ts
pending launch:      admit_ts <= t < collective_call_start_ts
```

- `peak_admission_occupancy`：scheduler 容量真正约束的峰值，包含已准入未发射任务；
- `peak_launched_inflight`：已调用底层 collective、尚未被 probe 观察完成的峰值；
- `*_by_count_us`：观察窗口内 count=0/1/2/... 各持续多久；
- `*_time_us`：`count × duration` 的 time-weighted task-time，不等于墙钟 makespan；
- `finite_capacity_ok`：有限容量下是否满足 rank-local peak ≤ configured capacity。

这些都是 rank-local 观测，不代表全局 completion barrier。

### 5.4 `summary.json`

summary 只读取 manifest 列出的成功样本。每个分布保存：

```text
samples, n, median_us, p10_us, p90_us, min_us, max_us
```

`paired_comparisons` 按相同 workload 和 repetition 配对。命名：

```text
<current>_minus_<baseline>
```

差值定义为 `current - baseline`：负值表示 current 更快，正值表示 current 更慢。配对差值的
中位数不是“两个场景中位数之差”；正式判断优先读取 `paired_comparisons`。

### 5.5 `plans.json`、`analysis.md` 和 `capacity-selection.json`

- `plans.json`：每个 workload/policy 的 key 顺序、Plan digest、same-plan 标记、候选评分和
  tie-break diagnostics；
- `analysis.md`：summary 的可读表格和方法限制；原始统计以 `summary.json` 为准；
- `capacity-selection.json`：第一阶段分析后冻结的第二阶段容量及理由，并绑定 source summary
  哈希。

报告口径勘误集中列在本文 [1.1 节](#11-analysis-报告勘误)。容量正式 batch 的配对表读取的是
正确配对差值；在修正并重新 finalize 前，以配对表和
`summary.json -> paired_comparisons` 为准。

## 6. 图表阅读方法

所有正式 SVG 都从 batch manifest 自动生成。蓝色原始点表示每次 run；红色短线/方形为
中位数；竖线或横线为 P10–P90。P10–P90 是样本分布范围，不是置信区间。

### 6.1 Timeline

文件：`figures/<workload>-timeline-rank0.svg`。

- 每个纵向 panel 是一个 scenario；横轴从该 rank 的 `application_release_ts` 开始，单位 ms；
- 每个 job 分为 compute、admission、communication、application、backend、validation 六行；
- 灰色：producer compute；
- 绿色：consumer overlap compute；
- 橙色：ready → admit；
- 蓝色：collective call start → completion observed；
- 浅红：application wait → underlying Work 绑定；
- 红色：underlying wait → wait return；
- 紫色：drain 后 harness validation；
- 黑三角：ready；黑竖线：admit、collective submit 或 completion observation；
- `control coordination`：ready-first 控制面 collective；
- `scheduler state`：按下表重建的 rank-local 状态。

Scheduler-state 颜色含义：

| 状态 | 含义 |
| --- | --- |
| `no ready work` | 当前没有 ready task |
| `capacity busy` | admission occupancy 已达到有限容量 |
| `scheduler HOL` | Plan 头未 ready，但后续存在 ready task，且当时没有更高优先级状态 |
| `scheduler delay` | Plan 头已 ready，但尚未准入 |
| `data collective in flight` | 至少一个 collective 已调用且未观察完成 |
| `work-conserving idle` | 有可运行工作却没有发射/在途通信 |
| `admission pending launch` | 已准入但底层 collective 尚未调用 |
| `coordination wait` | ready-first 控制面等待 |
| `unattributed wait` | 无法由现有事件唯一归因 |

状态使用优先级切分，因此不能把所有看似相关的时间重复相加。特别是已有 collective 在途时，
后续 ready task 被静态队首挡住的时间可能显示为 `inflight`，HOL 行只表示当前归因规则下的
队首空档。

每个场景独立选择 workload makespan 最接近该场景中位数的 trace；不同 panel 可能不是同一
repetition。选择结果保存在 `manifest.json -> representative_traces`。图一次只画一个 rank，
不同 rank 的单调时钟绝对值不能直接合并。

手工重画或放大：

```bash
PYTHONPATH=src:. python -m examples.jobpacer.analysis.visualize timeline \
  --result-dir benchmark/phase1.2/results/<category>/<batch-id> \
  --workload staggered-compute \
  --rank 0 \
  --x-min-ms 0 --x-max-ms 120 \
  --output /tmp/staggered-rank0.svg
```

### 6.2 通用 `summary.svg`

每个 workload 独立分面，依次显示 workload makespan、run 内平均 job completion 和每个 job
completion。最后一行是 workload makespan 的配对差值：黑色 0 线右侧表示 current 更慢，
左侧表示 current 更快。

### 6.3 容量扫描图

| 图 | 阅读方法 |
| --- | --- |
| `capacity-makespan.svg` | x 轴为 k1/k2/k3/unbounded；FIFO 蓝色、LTF 橙色；bare 单独作为参考，不与容量连线 |
| `capacity-job-completion.svg` | 各颜色是 job，黑色是每次 run 内 job completion 平均；用于区分总 makespan 与短 job 延迟 |
| `capacity-observed-concurrency.svg` | 每个点是一个 rank/run 的 peak admission 或 launched；短横线是配置的有限容量 |
| `capacity-occupancy-time.svg` | 每个 scenario 三根相邻柱依次为 admission、launched、pending-launch；堆叠颜色表示 count=0/1/2/... 的平均 rank 时间 |

配置容量没有达到不等于 scheduler 失效，可能是 job 数、producer readiness 或 job 内依赖限制了
实际并发。逻辑字节数除以应用耗时只能称为逻辑吞吐，不能解释为物理链路利用率。

### 6.4 SRJF 策略图

| 图 | 阅读方法 |
| --- | --- |
| `policy-makespan-by-capacity.svg` | 相同容量下比较 FIFO/LTF/SRJF 的 workload makespan；bare 是独立参考 |
| `policy-mean-job-completion.svg` | 比较每次 run 内的平均 job completion；不能用各 job 中位数的平均代替 |
| `policy-per-job-completion.svg` | 查看短 job 改善是否由长 job 延后换取 |
| `policy-paired-differences.svg` | 同 repetition、同容量配对；current−baseline，正值更慢 |
| `policy-ready-wait.svg` | 每次 run 中所有 rank/task 的最长 ready→admit |
| `policy-scheduler-state.svg` | 各状态的 mean rank time 堆叠；控制面 coordination 单独记录，可能与数据面状态重叠 |

比较策略时同时看配置容量和 `capacity-observed-concurrency`。相同 `max_outstanding` 不保证不同
静态 Plan 达到相同实际并发；策略可能因固定队首而主动压低可利用容量。

### 6.5 灵敏度图

灵敏度 batch 的 `summary.svg` 与 timeline 只描述 0.1 ms 新样本，不与 1 ms batch 池化。
如果差值接近 poll interval、P10–P90 跨 0 或随 poll interval 改变符号，应标记为探测分辨率
敏感，而不是选择更有利的批次。

## 7. 结果使用约束

- 所有性能比较优先使用同一 batch、同一 repetition 的 paired difference；
- 不把独立场景中位数之差称为配对中位数；
- 不将 P10–P90 称为置信区间；
- 不将 completion observation 称为精确物理完成；
- 不将 rank-local capacity validation 称为全局完成屏障；
- 不将 bare 当成 scheduler unbounded 容量点；
- 不依据同一批数据事后为每个策略挑选不同最佳容量；
- `git_worktree_dirty=true` 的 batch 虽有源码哈希，但哈希不能恢复未提交内容；长期归档应同时
  保存 commit、diff 或源码快照；
- 当前 10 次重复用于功能、机制和明显趋势分析；需要正式平台结论时，应在目标 GPU/NCCL
  环境重新 profile，并增加交错重复次数。
