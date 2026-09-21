# JobPacer Phase 1/2：容量扫描与 SRJF 详细实施计划

日期：2026-09-19。

状态：已完成（2026-09-20）。下方清单和执行记录对应代码、测试、实验、作图及验收结果；
实验结论仅适用于本文注明的 CPU/Gloo、两 rank 和线性 sleep workload 范围。

## 1. 目标、边界和执行顺序

本轮只回答两个问题：

1. 在相同静态 FIFO/LTF Plan 下，`max_outstanding=1、2、3、unbounded` 如何影响应用
   makespan、job 完成时间和实际并发；
2. 容量集合冻结后，静态非抢占 SRJF 相对同容量 FIFO/LTF 如何重新分配 job 完成时间，
   是否改善平均 job 完成时间，以及它对长 job 和整体 makespan 的代价。

必须严格分两阶段执行：

```text
实现公共实验基础
  -> FIFO/LTF 容量扫描 smoke
  -> FIFO/LTF 容量扫描正式探索批次
  -> 分析并冻结第二阶段容量集合
  -> 实现/验收 SRJF
  -> FIFO/LTF/SRJF 新批次 smoke
  -> FIFO/LTF/SRJF 正式探索批次
  -> 绘图与分阶段结论
```

第二阶段不得先运行 SRJF 再依据其结果为不同策略各选一个“最佳容量”。容量选择只读取第一阶段
的结果，并写入独立的选择记录后冻结。

本轮保持以下边界不变：

- 两 rank、CPU/Gloo、真实 `all_reduce`；
- 已有 8 个线性 workload；
- 每个 job 内通信依赖顺序不变；
- completion poll 主批次固定为 1 ms；
- 不实现 DAG、GPU/NCCL、自适应容量、抢占、aging 或新的在线调度协议；
- bare 继续由 `examples/jobpacer/run_phase1.py` 执行，不经过 scheduler；
- 数值校验继续在应用结束和通信排空后执行，不重新进入 task 推进关键路径。

## 2. 预计修改范围

| 文件 | 修改内容 |
| --- | --- |
| `examples/jobpacer/plan_builder.py` | 抽取共同 remaining score，增加静态 `srjf` 排序和选择诊断 |
| `examples/jobpacer/run_replay.py` | 接受 `srjf`；把通用有限容量合法性写入验证结果 |
| `examples/jobpacer/replay_worker.py` | 输出容量观测所需边界和 rank-local 占用统计，保持执行语义不变 |
| `benchmark/phase1.2/run_experiments.py` | 支持 capacity/SRJF 两套显式矩阵、manifest 驱动的 finalize、配对统计和报告 |
| `examples/jobpacer/visualize.py` | 从 manifest 读取场景；增加容量曲线、实际并发和策略比较图 |
| `tests/unit/test_jobpacer_plan.py` | SRJF 顺序、同分、job 内顺序和 overlap score 回归 |
| `tests/unit/test_jobpacer_measurement.py` | 占用重建、峰值、time-weighted 分布和图表数据检查 |
| `tests/unit/test_scheduler.py` | `max_outstanding>1` 的容量、完成释放、错误唤醒和排空检查 |
| `tests/integration/` | 两 rank Gloo 下 k=2/3、计划一致性、容量上界和正常退出 smoke |

不新建策略插件框架。现有 `register_policy()` 已足够；不为一次实验引入新的配置依赖或统计
依赖。

## 3. 第零阶段：公共实验基础改造

### 3.1 让场景集合以 manifest 为唯一事实来源

当前 runner 和 visualizer 各自依赖固定 `SCENARIOS`，不适合 9 场景容量扫描和冻结后才确定
容量数的 SRJF 实验。改造后由 runner 构造有序场景映射并原样写入 manifest：

```json
{
  "experiment": "capacity_scan",
  "scenarios": {
    "phase1_bare": {
      "mode": "bare",
      "policy": null,
      "selection": "runtime_arrival",
      "max_outstanding": 0,
      "capacity_label": "unbounded"
    },
    "fifo_k2": {
      "mode": "scheduler",
      "policy": "fifo",
      "selection": "runtime_arrival",
      "max_outstanding": 2,
      "capacity_label": "k2"
    }
  }
}
```

约束如下：

- JSON 对象插入顺序就是作图和报告默认顺序；另外保存显式 `scenario_order`，避免工具依赖
  JSON 实现细节；
- runner 执行、expected run 数、finalize、summary、配对统计、代表 trace 选择和绘图只遍历
  manifest 中的场景；
- finalize 不再用当前源码中的默认矩阵推测旧 batch；
- 每条 run record 保存 `scenario`、`policy`、`max_outstanding`、`repetition`、`order_index`、
  命令、起止时间、输出路径和 SHA-256；
- batch 开始后不允许修改场景矩阵；恢复或 finalize 时校验 run key 恰好为
  `(workload, repetition, scenario)` 的笛卡尔积，且无重复；
- visualizer 不扫描目录中的额外 `run-*.json`，只读取 manifest 列出的、哈希匹配的文件。

建议为 runner 增加最小的实验选择参数：

```text
--experiment capacity-scan
--experiment srjf
```

`capacity-scan` 固定生成 bare 加 FIFO/LTF 的 1、2、3、0；`srjf` 要求显式传入第一阶段冻结
的容量选择文件。不要增加任意策略表达式解析器。

### 3.2 两套场景矩阵

第一阶段有序矩阵固定为：

```text
phase1_bare
fifo_k1, fifo_k2, fifo_k3, fifo_unbounded
ltf_k1,  ltf_k2,  ltf_k3,  ltf_unbounded
```

其中 CLI 仍以 `0` 表示不限容量，展示层统一写作 `unbounded`。`phase1_bare` 不得画成
“scheduler 的 unbounded 容量点”，因为它还改变了执行路径。

第二阶段场景由冻结容量集合 `K` 生成：

```text
phase1_bare
for k in K:
    fifo_<k>, ltf_<k>, srjf_<k>
```

`K` 必须包含 `k1` 和 `unbounded`；只有第一阶段显示证据时才增加 `k2` 和/或 `k3`。三种
策略必须使用完全相同的 `K`。

### 3.3 固定、记录并校验 Plan

每个 workload/profile 组合只为每个策略构造一次 Plan。容量不是 Plan 输入：

- `fifo_k1/k2/k3/unbounded` 引用同一个 FIFO Plan digest；
- `ltf_k1/k2/k3/unbounded` 引用同一个 LTF Plan digest；
- 第二阶段同理，并增加 SRJF Plan；
- `plans.json` 保存 Plan key 序列、digest、score definition 和逐步 diagnostics；
- manifest 的每个 scheduler 场景记录其 `plan_digest`；
- finalization 检查同一 workload/policy 的各容量场景 digest 一致；
- 每个正式 trace 仍检查实际提交投影和各 ProcessGroup 内顺序与共同 Plan 相符。

若 FIFO 与 LTF/SRJF 在某个同构 workload 上得到相同 Plan，报告显式标记
`identical_plan=true`，该组差异只反映运行波动，不能解释为排序收益。

### 3.4 容量观测的两个口径

“配置容量”不能代替实际并发。对每个 rank 使用已有 task 时间边界重建两个阶梯函数：

```text
admission occupancy:
    admit_ts <= t < completion_observed_ts

launched inflight:
    collective_call_start_ts <= t < completion_observed_ts

pending launch:
    admit_ts <= t < collective_call_start_ts
```

其中 admission occupancy 是 scheduler 限制的容量口径；它包含已准入但尚未调用 collective
的任务。launched inflight 才表示调用后的观测在途数量。`completion_observed_ts` 是探测到完成
的时刻，不表述为精确物理完成时刻。

每个 rank、每次 run 输出或在 summary 中派生：

- `configured_max_outstanding`；
- `peak_admission_occupancy`；
- `peak_launched_inflight`；
- `admission_occupancy_time_us`：按 count 记录时间加权时长；
- `launched_inflight_time_us`：按 count 记录时间加权时长；
- `pending_launch_time_us`；
- 可选比例只用相同闭合观察窗口作分母，并同时保留原始微秒数。

重建时以所有 admit、call start 和 completion observation 为切分点；同一时间戳采用半开区间
语义，并按“完成释放、调用开始、准入占用”的确定性顺序处理，避免瞬时伪峰值。观察窗口使用
`application_release_ts -> communication_drain_end_ts`。

有限容量验收条件为每个 rank：

```text
peak_admission_occupancy <= configured_max_outstanding
```

这是 rank-local 容量验证，不得改写为全局完成屏障。unbounded 场景只记录观测峰值，不设置
人为上界。

### 3.5 新增 run 级完成指标

每次 run 先计算，再跨 repetition 汇总：

- `workload_makespan_us`：各 rank application makespan 的最大值；
- `job_completion_us[job_id]`：该 job 所有参与 rank 从统一 release 到 application end 的最大值；
- `mean_job_completion_us`：本次 run 内所有 `job_completion_us` 的算术平均；
- 每个 job 的 makespan/完成时间原始值；
- 最长和分位数 ready-to-admit；
- scheduler state 时间与两种 occupancy 分布。

不得先对每个 job 跨运行取中位数，再平均这些中位数来冒充平均 job 完成时间。

### 3.6 配对比较定义

配对只按同一 workload、同一 repetition 匹配，统一报告 `current - baseline`：

第一阶段至少产生：

- 同策略：`k2-k1`、`k3-k1`、`unbounded-k1`；
- 同策略：`k1-unbounded`、`k2-unbounded`、`k3-unbounded`；
- 同容量：`ltf-fifo`；
- 每个 scheduler 场景：`scenario-phase1_bare`。

第二阶段至少产生：

- 同容量：`srjf-fifo`；
- 同容量：`srjf-ltf`；
- 同容量：`ltf-fifo`；
- 每个 scheduler 场景：`scenario-phase1_bare`。

每项分别覆盖 workload makespan、`mean_job_completion_us` 和各 job 完成时间，保留原始差值、
样本数、中位数、P10–P90。P10–P90 只称为运行结果分布，不称为置信区间。

## 4. 第一阶段：FIFO/LTF 容量扫描实现

### 4.1 Runner 行为

容量扫描命令的预期形式为：

```bash
python benchmark/phase1.2/run_experiments.py \
  --experiment capacity-scan \
  --repeats 2 \
  --experiment-seed 120914
```

runner 对每个 workload：

1. 创建或复用该 batch 内的通信 profile；overlap-window 家族继续共享内容一致的 profile，
   但每条 run 保存 profile digest；
2. 每个 repetition 对全部 9 个场景做确定性 shuffle；
3. 逐个 replay 串行运行，不并发启动不同场景；
4. bare 调用 `run_phase1.py`，其它场景调用 `run_replay.py`；
5. 每次成功后立即校验 trace、写 SHA-256 并原子更新 manifest；
6. 任一 run 失败时保留 `complete=false`、失败命令和错误状态，不生成“完整批次”结论；
7. 全部通过后才生成 `plans.json`、`summary.json`、`analysis.md` 和 figures，最后将
   `complete` 置为 true。

随机顺序由 `(experiment_seed, workload, repetition)` 决定。若所有 workload 使用相同场景
排列，也必须在 manifest 明确记录；优先加入 workload 名，避免每个 workload 都在同一时间
位置运行同一策略。

### 4.2 容量专项正确性检查

在正式 smoke 前完成：

- core scheduler 在 `k=2` 时可准入两项，第三项等待，任一已准入任务被 completion probe
  观察完成后释放一个槽位；
- `k=3` 和 `0` 不错误退化为串行；
- 已准入未发射任务也计入容量；
- consumer 未调用 `wait()` 时，completion observer 仍能释放容量；
- executor/probe 抛错时，所有待等待的 `ScheduledWork` 被唤醒并得到同一失败，而不是挂起；
- finish/drain 在多项在途时能够有界结束；
- 跨多个 ProcessGroup 时，各 group 的 Plan 投影顺序合法；
- 两 rank Gloo smoke 中各 rank 的峰值 admission occupancy 不超过 k，tensor 校验通过；
- `k>1` 至少有一个人工可并发 fixture 实际观察到峰值大于 1，证明测试没有只覆盖名义容量。

不得为了让检查通过而给有限容量场景增加跨 rank completion barrier；扫描保持当前 rank-local
scheduler 语义。

### 4.3 Smoke 和正式探索批次

按 workload 家族分批检查，建议顺序：

1. `comm-heavy`；
2. `mixed-message-sizes-large-first/large-last`；
3. `long-short-chains`；
4. `staggered-compute`；
5. `overlap-window-short/medium/long`。

每个场景先运行 2 次 smoke：

```text
8 workloads × 9 scenarios × 2 repetitions = 144 replay runs
```

每完成一个家族，记录 wall-clock、失败数和大致资源消耗。完整 smoke 只有在以下条件全部成立
时才能升级：

- 144/144 run（或所选家族的完整笛卡尔积）均有明确状态；
- 所有成功 run tensor 正确、成员和 task 覆盖完整；
- scheduler 场景全部匹配共同 Plan 和 group 投影；
- 有限容量全部通过 rank-local 上界验证；
- 所有 task completion observation 不晚于 communication drain；
- summary 的样本路径、数量和哈希与 manifest 一致；
- k>1 未出现线程挂起、错误未唤醒或 drain 超时。

之后以新 batch、同一 1 ms poll interval 运行 10 次探索：

```text
8 workloads × 9 scenarios × 10 repetitions = 720 replay runs
```

旧 batch 的 bare 不复用为配对基线；本轮 bare 必须与其余 8 个场景交错重跑。

### 4.4 第一阶段图表

至少生成以下图：

1. `capacity-makespan.svg`：按 workload 分面，x 轴为 `k1/k2/k3/unbounded`，FIFO 和 LTF
   分线；显示原始点、median、P10–P90。bare 用独立水平参考线/点，不连接进容量曲线；
2. `capacity-job-completion.svg`：每个 workload 展示各 job 完成时间及
   `mean_job_completion_us`，避免只看最后结束的 job；
3. `capacity-observed-concurrency.svg`：同时画配置容量、`peak_admission_occupancy` 和
   `peak_launched_inflight` 的原始分布；
4. `capacity-occupancy-time.svg`：按场景展示 time-weighted occupancy count 分布，区分
   admission occupancy 与 launched inflight；
5. 每个 workload 至少一张可追溯 timeline，优先选择 k1、出现平台/反转的有限 k、
   unbounded 和 bare。9 场景全画导致不可读时可拆页，但不能手工挑最好一次。

代表 trace 仍按“该场景 workload makespan 最接近其中位数”自动选择，并在 manifest 保存路径
和哈希。不同场景的代表 trace 可能来自不同 repetition，图注必须说明。

### 4.5 第一阶段分析和容量冻结

第一阶段 `analysis.md` 对每个 workload 都报告完整容量曲线，不只列事后最快点，并回答：

- k 从 1 到 2/3 是否改善，何处趋平；
- 实际峰值是否达到配置值；若没有，是否受 job 数、ready 或依赖限制；
- 有限 k 是否优于 unbounded，配对差值是否大到超过 completion probe 分辨率疑虑；
- 容量增加后静态队首阻塞是否仍在；
- LTF/FIFO 相同容量差异是否主要表现为 job 完成次序重分配；
- 与 bare 的差距是否来自 scheduler 路径、静态顺序和容量的合并效应。

随后新增第一阶段 batch 内的 `capacity-selection.json`：

```json
{
  "source_batch_id": "...",
  "required_capacities": [1, 0],
  "selected_optional_capacities": [2],
  "second_stage_capacities": [1, 2, 0],
  "rationale": ["..."],
  "source_summary_sha256": "..."
}
```

选择规则：

- 永远保留 1 和 0；
- k2/k3 只有在至少一个重点 workload 出现值得复核的配对改善，或能改变排序效应时才纳入；
- 若 k2 和 k3 的实际峰值及结果没有有效区别，只保留信息量更高的一个；
- 若 unbounded 最好，不强行选择有限“最优容量”；
- 选择理由引用 summary 字段和图，不根据 SRJF 结果回改。

该文件写完并经人工阅读后才开始第二阶段。

## 5. 第二阶段：SRJF 最小实现

### 5.1 共同评分函数

将当前 `ltf_score()` 的实现抽取/重命名为策略无关的 remaining score；保留兼容别名可减少
无关改动。对 job 当前候选通信 `i`：

```text
remaining(i) = max(estimated_comm_i, consumer_compute_i)
             + sum(
                   producer_compute_j
                   + max(estimated_comm_j, consumer_compute_j)
                   for j > i
               )
```

语义必须写入 diagnostics：

```text
zero-admission-delay estimated remaining critical path
```

候选当前 producer 不计入分数，因为评分假设候选已经 ready；后续 task producer 各计一次。
consumer compute 与该段通信取 `max`，表示现有 replay 中允许重叠，而不是把 consumer compute
错误相加为通信后的 tail。

### 5.2 SRJF 选择算法

SRJF 与 LTF 共用候选生成和分数：

```text
positions[job_id] = 0
while 仍有未排入 Plan 的任务:
    candidates = 每个 job 的下一项
    LTF  选择 (-remaining, job_id, ordinal) 最小项
    SRJF 选择 ( remaining, job_id, ordinal) 最小项
    将选中项加入 Plan
    positions[selected.job_id] += 1
```

稳定 tie-break 固定为 `job_id`、`ordinal`；不得使用线程到达顺序、对象地址或运行时结果。
`policy_names()` 注册顺序预期变为 `("fifo", "ltf", "srjf")`。

`policy_diagnostics()` 对 LTF 和 SRJF 都保存每一步：

- 所有候选 key；
- 每个候选 remaining score；
- 被选 key 和 score；
- 排序方向 `max`/`min`；
- tie-break 值；
- score definition。

静态构建时下一候选未必在 replay 中已经 ready，因此 SRJF 仍可能等待固定队首。文档、CLI
帮助和分析只称其为“静态、非抢占的 shortest remaining job first”，不得称为在线
work-conserving SJF。

### 5.3 SRJF 单元与集成验收

增加确定性 fixture 覆盖：

- `long-short-chains` 型 workload 中，SRJF 先选短 remaining job，LTF 先选长 remaining job；
- `estimated_comm > consumer_compute` 和反向情况都按 `max` 计分；
- 后续 producer compute 只计一次，当前候选 producer 不计；
- 分数完全相同时按 job ID、ordinal 稳定选择，重复构造 Plan digest 相同；
- 每个 job 的 ordinal 严格递增，Plan 包含全部 task 且无重复；
- workload job 输入顺序变化时，只要 job ID 和内容相同，同分结果仍符合已声明 tie-break；
- `policy_diagnostics()` 的候选分数、selected score 和最终 Plan 一致；
- 两 rank 从相同 workload/profile 构造相同 SRJF Plan；
- k1 和 unbounded 下均能完成、校验正确并匹配 Plan；
- 失败或性能较差不修改评分公式，也不从正式样本中删除。

## 6. 第二阶段：策略比较实验

### 6.1 冻结输入与新 batch

第二阶段 runner 读取 `capacity-selection.json`，验证：

- `source_batch_id` 指向 complete 的第一阶段 batch；
- `source_summary_sha256` 匹配；
- 容量集合包含 1 和 0，且只含 0/1/2/3；
- 场景恰好为 bare 加每个容量下的 FIFO/LTF/SRJF；
- 所有策略共享 workload、profile、backend、poll interval 和执行路径参数。

预期命令形式：

```bash
python benchmark/phase1.2/run_experiments.py \
  --experiment srjf \
  --capacity-selection benchmark/phase1.2/result/batches/<capacity-batch>/capacity-selection.json \
  --repeats 2 \
  --experiment-seed 120915
```

第二阶段必须创建新 batch，并在新 batch 内重跑 FIFO、LTF 和 bare。不得从第一阶段复制性能
样本；第一阶段只提供容量选择证据。

若冻结容量数为 `C`：

```text
每个 workload 每轮场景数 = 3C + 1
smoke runs = 8 × (3C + 1) × 2
exploration runs = 8 × (3C + 1) × 10
```

同一 repetition 内对全部 `3C+1` 场景确定性随机交错，replay 继续逐个运行。

### 6.2 第二阶段统计与图表

重点图表：

1. `policy-makespan-by-capacity.svg`：每个容量下 FIFO/LTF/SRJF 的 workload makespan；
2. `policy-mean-job-completion.svg`：每次 run 内平均 job 完成时间；
3. `policy-per-job-completion.svg`：各 job 原始点、中位数和 P10–P90，突出 long/short job
   的得失；
4. `policy-paired-differences.svg`：同容量 SRJF−FIFO、SRJF−LTF、LTF−FIFO，0 线右侧表示
   current 更慢；
5. `policy-ready-wait.svg`：最长 ready-to-admit 及 scheduler state 归因；
6. 重点 workload timeline：`long-short-chains`、两个 mixed-message-sizes 和
   `staggered-compute`，展示静态顺序、队首等待和实际重叠。

图中必须标注 n、单位、poll interval 和“P10–P90 非置信区间”。容量轴使用分类值，不能把
unbounded 当作数值 0 连线。

### 6.3 第二阶段必须回答的问题

对每个容量分别回答：

1. SRJF 是否降低 `mean_job_completion_us` 和短 job 完成时间；
2. 长 job 延后多少，整体 makespan 是否增加；
3. SRJF/LTF 是否主要改变完成先后而没有改善总吞吐；
4. 容量增加后策略差异减弱、消失还是仍受静态队首阻塞支配；
5. 同 Plan workload 上观察到的差异是否落在环境波动范围；
6. 相对 bare 的端到端差距是否仍存在。

不得由有限、一次性到达的 workload 声称 SRJF 不会饥饿，也不得把 CPU/Gloo sleep workload
外推为 GPU/NCCL 或真实训练性能。

## 7. Completion poll 敏感性复核

两个主批次统一使用 1 ms。若某项关键结论的配对差值接近 1 ms、不同容量排序被 1 ms 尺度
覆盖，或 P10–P90 跨过 0 且结论依赖亚毫秒差异，则单独创建 0.1 ms sensitivity batch：

- 只选触发疑问的 workload 和被比较场景；
- 比较双方全部改为 0.1 ms，不把 0.1 ms 和 1 ms 样本合并；
- 使用新 seed 和相同配对设计；
- 单独输出 manifest、summary、图和 analysis；
- 若结论随 poll interval 变化，正式报告标为 resolution-sensitive，而不是选择更有利批次。

## 8. 产物目录和可复现性

两个阶段各自产生独立目录：

```text
benchmark/phase1.2/result/batches/<batch-id>/
  manifest.json
  plans.json
  capacity-selection.json       # 仅第一阶段分析后产生
  profiles/
  raw/<workload>/<scenario>/run-XX.json
  summary.json
  analysis.md
  figures/
    ...
```

manifest 至少保存：

- experiment 类型、batch ID、complete 状态；
- workload 清单及内容 digest；
- 有序场景矩阵和容量标签；
- repetitions、seed、每轮实际执行顺序；
- backend、world size、poll interval、Python/Torch/平台信息；
- runner、replay worker、Plan builder、visualizer、scheduler 的源码 SHA-256；
- profile 路径、规范化内容 digest 和文件 SHA-256；
- 每个 trace 的路径、状态和 SHA-256；
- summary、analysis、plans、figures 的路径和 SHA-256；
- 代表 timeline trace 的路径。

finalize 必须在任何派生产物生成前验证上述输入。重画图只读取 batch manifest，不读取目录中
未列出的旧文件。派生产物更新后同步更新其哈希，历史 complete batch 不被新实验覆盖。

## 9. 分阶段验收门槛

### 9.1 代码验收

- `policy_names()` 包含 `srjf`，FIFO/LTF 既有顺序回归不变；
- SRJF 算法、score 和 diagnostics 的单元测试通过；
- k=2/3/unbounded 的 scheduler 容量、释放、错误和 drain 测试通过；
- occupancy 重建测试覆盖 pending launch 和同 timestamp 边界；
- runner/finalize/visualizer 均以 manifest 场景为准；
- 全量现有 unit/integration 测试无回归。

### 9.2 单个 run 验收

- tensor 校验成功；
- rank、成员、job 和 task 覆盖完整；
- scheduler 实际提交序列及 group 投影符合共同 Plan；
- 有限容量每个 rank 的 admission occupancy 峰值不超过 k；
- application、drain、validation、harness 边界闭合；
- 所有 task 的 completion observation 已在 drain 前记录；
- 必填时间戳单调，缺失或逆序直接失败，不在绘图时静默省略；
- 命令、输入 digest 和输出哈希已记录。

### 9.3 Batch 验收

- run key 等于预期笛卡尔积，无缺失和重复；
- 每轮每个场景恰好一次，执行顺序有记录；
- summary 的 n 与 manifest 一致；
- 配对比较不存在缺失 repetition；
- Plan digest 在同 workload/policy 的容量之间一致；
- 图表所有原始点均能追溯到 manifest run；
- `analysis.md` 明确失败、跳过、未验证边界和实验总耗时；
- 只有全部通过后 manifest 才设 `complete=true`。

## 10. 实施清单

按以下顺序提交和验证，避免把基础设施、算法和性能结果混在一个不可定位的改动中：

- [x] A1. runner 增加 experiment 类型和有序场景构造；
- [x] A2. finalize、summary、visualizer 改为完全 manifest 驱动；
- [x] A3. 增加 admission/launched 两套 occupancy 重建、统计和有限容量验证；
- [x] A4. 增加 run 内平均 job 完成时间及容量配对比较；
- [x] A5. 完成公共单元与两 rank k>1 集成测试；
- [x] B1. 执行 FIFO/LTF 容量 smoke，检查 144 个 run 或所选家族完整矩阵；
- [x] B2. 执行 10 次容量探索批次并生成容量图；
- [x] B3. 写第一阶段分析和 `capacity-selection.json`，冻结容量集合；
- [x] C1. 抽取共同 remaining score，实现 SRJF 和 diagnostics；
- [x] C2. 完成 SRJF 单元与两 rank 集成检查；
- [x] D1. 按冻结容量执行第二阶段 2 次 smoke；
- [x] D2. 执行第二阶段 10 次探索批次；
- [x] D3. 生成策略图、timeline、summary 和独立 analysis；
- [x] E1. 对任何接近探测分辨率的关键差异运行独立 0.1 ms sensitivity batch；
- [x] E2. 最终复核报告只区分“容量效果”和“同容量排序效果”，不声称未验证的因果关系。

## 11. 完成定义

只有同时满足以下条件，本任务才算完成：

1. 第一阶段 9 场景容量扫描和第二阶段冻结容量策略比较均有独立、完整、可哈希验证的 batch；
2. SRJF 的定义、实现、diagnostics 和测试完全一致；
3. 报告同时展示整体 makespan、每个 job、run 内平均 job 完成时间和实际并发，而非只展示
   最快配置；
4. 所有正式图由 manifest 样本自动生成并可追溯；
5. 结果无论是“有限容量无收益”“SRJF 只改善短 job 但拖慢整体”还是“未超过 bare”，都原样
   保留和解释；
6. 分析明确当前结论只适用于本轮 CPU/Gloo、两 rank、线性 workload 和所用探测分辨率。

## 12. 执行记录与结果概述

完整 manifest、原始 trace、summary、analysis 和图均保存在各自 batch 目录中；不要将不同
batch 的 trace 合并计算。

| 阶段 | Batch | 结果 |
| --- | --- | --- |
| 容量 smoke | `20260919T134222Z-afde4d` | 144/144 成功 |
| FIFO/LTF 容量探索 | `20260919T143653Z-9536db` | 720/720 成功；8 workloads × 9 场景 × 10 次 |
| SRJF smoke | `20260920T015844Z-fa107f` | 208/208 成功；8 workloads × 13 场景 × 2 次 |
| FIFO/LTF/SRJF 探索 | `20260920T021723Z-6f899d` | 1040/1040 成功；8 workloads × 13 场景 × 10 次 |
| poll 灵敏度复核 | `20260920T015224Z-ec2fad-poll-sensitivity` | 80/80 成功；overlap-window-long 的 8 个 FIFO/LTF 场景 × 10 次，poll 为 0.1 ms |

第一阶段选择文件为
`benchmark/phase1.2/result/batches/20260919T143653Z-9536db/capacity-selection.json`，冻结
集合为 `[1, 2, 3, 0]`（展示为 k1/k2/k3/unbounded）。k2 在多个重点 workload 上明显优于 k1，
k3 在 comm-heavy、long-short-chains 和 mixed-message-sizes-large-last 上又提供额外信息，且
rank-local 实测峰值到达 3，因此两者都保留；这不是依据第二阶段 SRJF 性能挑选的“最佳容量”。

0.1 ms 复核中，overlap-window-long 的容量配对差值仍大多有跨 0 的 P10–P90，且部分中位数
相对 1 ms 批次变号。该 workload 的 k2/k3/unbounded 排名因此标为 resolution-sensitive；不据此
主张其中某个容量更快，也不把两种 poll interval 的样本合并。

第二阶段的配对结果体现的是同容量排序效果，不是容量效果：

- SRJF 在两个 mixed-message-sizes workload 中降低 run 内平均 job 完成时间，并明显提前
  部分短 job；以 k2 对 FIFO 为例，large-first 平均 job 完成时间差中位数为 −19.55 ms，
  makespan 为 +17.90 ms；其中 16 MiB 的 job-0 延后 +17.79 ms，两个 1 MiB job 分别提前
  42.55/35.17 ms。large-last 平均 job 完成时间差为 −11.40 ms、makespan 为 +15.94 ms；两个
  1 MiB job-0/job-1 分别提前 29.39/20.89 ms，16 MiB 的 job-2 延后 +15.94 ms。对应的 paired
  P10–P90 同样显示这些 job 取舍方向稳定，完整差值留在 summary 中。
- 这种收益并非普遍存在。comm-heavy 在 k2 对 FIFO 的 makespan 中位数差为 +49.39 ms，平均
  job 完成时间差为 +5.06 ms；staggered-compute 各容量下 SRJF 相对 FIFO/LTF 的 makespan 和
  平均 job 完成时间都更差。
- long-short-chains 中，SRJF 相对 LTF 的平均 job 完成时间下降，但 makespan 上升；相对 FIFO
  则并未普遍改善平均值。容量增加后，SRJF 的长队首任务和静态非抢占顺序仍能限制并发；例如
  comm-heavy、mixed-message-sizes-large-first/last 的 SRJF k3 中位 rank-local admission 峰值为
  2，而 FIFO/LTF k3 达到 3。配置容量不能代替实测并发指标。
- FIFO 与 LTF 在部分 workload 上 Plan 相同；其差值只能视作运行波动，不能归因于排序收益。
  各 workload/policy 的 Plan digest、paired raw differences、每个 job 的结果及 P10–P90 均保存在
  `summary.json`；P10–P90 是观测分布，不是置信区间。

测试验收：`pytest -q tests/unit tests/integration` 为 95 passed；两 rank Gloo 容量/策略集成验证
通过。两个正式 batch 的每个有限容量 run 均通过 rank-local admission 上界检查，计划 digest 在
同策略不同容量间一致；summary、analysis 和 figures 均通过 manifest 哈希核验。上述结论不外推
至 GPU/NCCL、真实训练、在线到达、抢占或饥饿特性。
