# JobPacer Phase 1/2 测量语义与实验对照修正计划

## 1. 背景

Phase 1/2 当前已经具备以下能力：

- Phase 1 通过 `run_phase1.py` 裸发不同 job 的异步 collective；
- Phase 2 通过 `AdmissionScheduler` 执行静态 FIFO/LTF Plan；
- bare 和 scheduler 路径都使用 completion probe 记录物理完成观测时间；
- benchmark runner 能重复执行实验并汇总 makespan；
- `visualize.py` 能画 timeline 和多次运行分布。

现有结果足以验证功能，但还不能把性能差异唯一归因于调度策略。当前测量同时混入逐 tensor
正确性校验、不同含义的 wait 时间戳、tail 估计与实际重叠语义偏差、并发度变化、scheduler
执行路径变化、固定批次运行的时间漂移，以及不闭合的 makespan 边界。

本计划先修正时间与执行语义，再建立能够拆分退化来源的对照组，最后重新生成 trace、统计、
图表和分析。直接增加重复次数不能替代这些修正。

## 2. 修正原则

1. **应用工作、通信完成观测、测试校验分层。** 数值正确性校验不能继续隐式进入应用
   makespan。
2. **事件名称必须对应真实边界。** `Work.wait()` 被调用、底层 Work 被绑定、底层 wait
   被调用和 completion probe 观察完成是不同事件。
3. **一次只改变一个实验因素。** 排序、最大在途数量和 scheduler 执行路径必须有独立对照。
4. **所有性能比较来自同一实验批次。** runner、summary 和 visualizer 共同读取 batch
   manifest，不再各自扫描目录猜测样本集合。
5. **只陈述当前证据支持的结论。** P10–P90 是样本分布范围，不是收益置信区间；rank-local
   completion observation 不是全局完成屏障。

## 3. 统一事件与时间模型

### 3.1 Task 事件

每个 task 至少记录以下字段，旧字段在一个兼容周期内保留但不再用于新分析：

| 新字段 | 含义 |
| --- | --- |
| `producer_compute_start_ts` | producer compute 开始 |
| `ready_record_ts` | intent 对应用层 ready |
| `submit_api_start_ts` | 应用进入裸 collective 或 `scheduler.submit()` |
| `submit_api_return_ts` | 上述 API 返回 |
| `admit_ts` | scheduler 准入；bare 等于 `collective_call_start_ts` |
| `collective_call_start_ts` | 开始调用底层 `dist.all_reduce` |
| `collective_call_return_ts` | 异步 collective 调用返回并取得 Work |
| `completion_observed_ts` | completion probe 首次观察到物理完成 |
| `consumer_compute_start_ts` | 可重叠 consumer compute 开始 |
| `consumer_compute_end_ts` | 可重叠 consumer compute 结束 |
| `application_wait_start_ts` | 应用开始调用 `work.wait()`，在等待绑定之前记录 |
| `underlying_wait_start_ts` | 已取得底层 Work，准备调用其 `wait()` |
| `wait_return_ts` | Work wait 返回 |
| `application_task_end_ts` | 应用完成本 task，默认等于结果消费边界，不含测试校验 |
| `validation_start_ts` | 测试 harness 开始校验该 tensor |
| `validation_end_ts` | 测试 harness 完成校验该 tensor |

`complete_ts` 改为兼容别名，值等于 `completion_observed_ts`；输出中增加
`trace_schema_version`，新分析只读取新字段。`first_wait_ts` 不再作为含义明确的统计字段：

- bare 旧值接近 `application_wait_start_ts`；
- scheduler 旧值实际是绑定后的 `underlying_wait_start_ts`；
- 两者不能继续直接比较。

### 3.2 派生耗时

trace 和 summary 分别报告：

```text
submit_api_duration_us
  = submit_api_return_ts - submit_api_start_ts

collective_call_duration_us
  = collective_call_return_ts - collective_call_start_ts

post_return_completion_observation_us
  = completion_observed_ts - collective_call_return_ts

call_to_completion_observation_us
  = completion_observed_ts - collective_call_start_ts

deferred_binding_wait_us
  = underlying_wait_start_ts - application_wait_start_ts

underlying_wait_duration_us
  = wait_return_ts - underlying_wait_start_ts

validation_duration_us
  = validation_end_ts - validation_start_ts
```

`actual_duration_us` 改名为 `call_to_completion_observation_us`。旧名只作为兼容字段，不再称为
“真实通信耗时”，因为 completion timestamp 含轮询探测延迟。

### 3.3 Run 级边界

每个 rank 明确记录四层边界：

| 边界 | 定义 |
| --- | --- |
| `application_release_ts` | 所有本地 job 线程到达启动门后统一释放 |
| `application_end_ts` | 所有 job 完成 workload 语义，不含测试校验 |
| `communication_drain_end_ts` | scheduler/observer 已观察全部通信完成 |
| `validation_end_ts` | 所有 tensor 正确性校验结束 |
| `harness_end_ts` | trace 整理与必要收尾完成 |

对应输出：

- `application_makespan_us`；
- `communication_drain_makespan_us`；
- `validation_total_us`；
- `harness_total_us`。

旧 `replay_makespan_us` 在兼容周期内等于 `application_makespan_us`，但 summary 和图表使用
新名称。任何 task 的 `completion_observed_ts` 晚于 `communication_drain_end_ts` 都是错误；
它可以晚于 `application_end_ts`，因为应用 wait 对 NCCL 不一定表示物理完成。

## 4. P1 修正

### 4.1 将数值校验移出逐任务关键路径

当前每项 `work.wait()` 后立即执行 `torch.all(tensor == expected).item()`，校验耗时进入 job
makespan，并决定下一 task 何时开始。修正为：

1. 每个 task 使用独立、预分配 tensor，job 线程不复用尚未校验的 tensor；
2. job 线程在应用消费边界后立即推进下一 task，不执行全 tensor 扫描；
3. 所有 job 结束且 completion observer/scheduler 排空后，由主线程统一校验；
4. 每个 tensor 记录 `validation_start_ts`、`validation_end_ts` 和 `correct`；
5. 校验失败仍使本次 run 失败，但不回写应用 makespan；
6. 输出 run 级 `validation_total_us`，可视化使用独立 harness lane 展示，不与 compute 或
   communication 混为一条轨道。

对于 NCCL，必须先完成物理 completion drain，再访问校验 tensor。不能用校验产生的隐式
同步替代 completion probe。

验收：关闭或扩大校验 tensor 扫描成本时，`application_makespan_us` 不变；
`validation_total_us` 随扫描成本变化。

### 4.2 拆分应用 wait 与底层 wait

修改 `ScheduledWork.wait()`：

1. 方法入口立即、且只记录一次 `application_wait_start_ts`；
2. 等待 `_underlying` 绑定的时间计入 `deferred_binding_wait_us`；
3. 绑定完成后、调用底层 `Work.wait()` 前记录 `underlying_wait_start_ts`；
4. replay worker 不再用 scheduler timing 覆盖应用侧 wait 时间戳；
5. bare 路径同样记录两个字段，通常二者接近，但仍保持统一 schema。

可视化将两段等待分开：

- ready/admission 轨道展示 `ready_record_ts -> admit_ts`；
- application wait 轨道展示 `application_wait_start_ts -> underlying_wait_start_ts`；
- backend wait 轨道展示 `underlying_wait_start_ts -> wait_return_ts`。

验收：delayed 场景中，应用在底层 Work 绑定前阻塞的约 10–100 ms 不再从图和统计中消失。

### 4.3 修正 LTF 的估计语义

当前实现把 `consumer_compute_s` 全部当作通信后的剩余工作，但 replay 允许它从
`scheduler.submit()` 返回后立即执行，并与 admission wait 和通信重叠。因此不再将当前
score 描述为“通信完成后的 tail”。

第一步将静态策略明确定义为 **estimated remaining critical path first**，保留 CLI 名称
`ltf` 以兼容现有实验，但在 trace 中记录：

```text
policy = "ltf"
score_definition = "zero-admission-delay estimated remaining critical path"
```

在假设 admission delay 为零时，任务段估计为：

```text
segment_i = producer_compute_i + max(estimated_comm_i, consumer_compute_i)
```

候选 task 已经 ready，因此当前 task 不再计 producer：

```text
score(i) = max(estimated_comm_i, consumer_compute_i)
         + sum(
             producer_compute_j
             + max(estimated_comm_j, consumer_compute_j)
             for j > i
           )
```

Plan trace 保存每一步所有候选的 score、被选 key 和稳定 tie-break。该 score 只表示静态、
零排队假设下的估计关键路径，不宣称是实际 post-completion tail。

同时报告两类偏差：

- `ready_to_job_end_observed_us` 与静态 remaining critical-path estimate；
- `completion_to_job_end_observed_us`，作为观测到的 post-completion tail，但不把它反向用于
  同一次实验的静态策略。

验收：单元测试覆盖 `consumer_compute_s < estimated_comm_s`、大于它、以及 admission delay
远大于二者的场景；文档和图不再把 12 ms consumer compute 直接称为通信后的 12 ms tail。

### 4.4 增加能够拆分因素的对照组

新的最小实验矩阵为：

| 场景 | 执行路径 | 选择方式 | max outstanding | 用途 |
| --- | --- | --- | ---: | --- |
| `phase1_bare` | raw collective | runtime arrival | unlimited | 原始并发 baseline |
| `ready_first_unbounded` | scheduler/executor/probe | globally ready first | unlimited | 估计 scheduler/协调路径开销 |
| `ready_first_serial` | scheduler/executor/probe | globally ready first | 1 | 相对上一项估计串行化代价 |
| `fifo_unbounded` | scheduler/executor/probe | static FIFO | unlimited | 静态顺序且保留并发 |
| `fifo_serial` | scheduler/executor/probe | static FIFO | 1 | 相对上一项估计单在途代价 |
| `ltf_serial` | scheduler/executor/probe | static corrected LTF | 1 | 与 FIFO 同并发度比较排序 |

由以下成对差异回答不同问题：

- `ready_first_unbounded - phase1_bare`：scheduler、控制协调和执行路径的合并开销；
- `ready_first_serial - ready_first_unbounded`：相同动态选择路径下的单在途代价；
- `fifo_serial - ready_first_serial`：同为单在途时静态 FIFO 队首阻塞/顺序的代价；
- `ltf_serial - fifo_serial`：相同执行路径和并发限制下 LTF 与 FIFO 的顺序差异；
- `fifo_serial - fifo_unbounded`：同一静态 Plan 下限制 outstanding 的代价。

这些差异仍是系统级合并效应，不能写成严格因果分解；尤其 globally-ready-first 包含控制面
协调开销。analysis 必须注明这一点。

### 4.5 Globally-ready-first 对照

不能让各 rank 独立选择本地 ready task，否则 rank 0 可能等待 communicator A、rank 1 等待
communicator B，在 `max_outstanding=1` 下死锁。ready-first 必须跨 rank 协调。

在 JobPacer 层实现最小控制协议，不改变核心静态 Plan 的正确性语义：

1. 每个 rank 的 job 线程仍提交 `CommIntent`；
2. 每个 rank-local controller 汇总本地 ready key；
3. 专用 control ProcessGroup 按固定 round 执行 `all_gather_object`；
4. 候选集合为所有成员 rank ready 集合的交集；
5. 从候选中按“首次进入全局 ready round，再按稳定 TaskKey”选择；
6. rank 0 广播选择结果，所有 rank 发射同一 key；
7. serial 模式下，各 rank 观察本地物理完成后再执行一次 control completion barrier，随后
   进入下一选择 round，因而提供明确的全局单在途对照；
8. unbounded 模式允许继续选择其它 globally-ready task，但仍保证每个 ProcessGroup 内顺序。

控制 collective 的次数、耗时和字节数单独记录，不能隐入数据面通信耗时。若 control
ProcessGroup 对 Gloo 数据面产生明显干扰，结果只作为 head-of-line 功能对照；目标 NCCL
环境应考虑 TCPStore 或独立 CPU/Gloo 控制面。

### 4.6 交错运行顺序

runner 不再按场景各跑完 10 次。改为按 repetition round 运行：

```text
for repetition in 0..N-1:
    order = deterministic_shuffle(all_scenarios, experiment_seed, repetition)
    for scenario in order:
        run scenario
```

每轮每个场景恰好出现一次，使用固定 seed 的确定性 shuffle。manifest 记录：

- repetition；
- 本轮执行顺序；
- scenario 的 order index；
- 开始/结束 wall-clock；
- 命令行和退出状态。

summary 除各场景分布外，增加同一 repetition 内的 paired difference。10 次样本仍只用于探索
和发现明显问题；正式 GPU/NCCL 结论使用至少 30–50 个交错 repetition。

## 5. P2 修正

### 5.1 区分 profile service time 与 replay observation time

通信 profile 的边界保持“collective 调用开始到同步确认完成”，重命名为：

```text
profile_service_time_s
```

profile 同时记录：

- collective API call duration；
- call return 到同步完成的时长；
- 总 service time。

replay 不再把 `completion_observed_ts - collective_call_return_ts` 与 profile p50 直接写成
“预测与真实通信耗时”。replay 报告三个组成部分：

- API call duration；
- return-to-observed-completion；
- call-to-observed-completion。

completion 字段明确包含 poll delay。trace 记录 `completion_poll_interval_s`，图例写成
“completion observed”，不能写成精确硬件完成。默认 1 ms 对亚毫秒通信误差过大；Gloo
小消息实验增加 0.1 ms 配置进行敏感性检查，但不默认用高频轮询替换所有环境。结果差异小于
轮询分辨率时不作性能解释。

### 5.2 统一启动门并闭合 makespan

所有本地 job 线程先完成创建并等待 `threading.Barrier`。每个 rank 在默认 control group 上
执行一次启动前 barrier，然后：

1. 记录 `application_release_ts`；
2. 释放本地线程启动门；
3. job 的 makespan 统一从 release 计算，不再从各线程实际获得 CPU 的时刻分别起算；
4. 另记每个线程的 `thread_first_run_ts`，用于观察 host scheduling jitter；
5. join 完成后记录 `application_end_ts`；
6. scheduler/observer 排空后记录 `communication_drain_end_ts`；
7. deferred validation 后记录 `validation_end_ts`。

workload 性能主指标采用各 rank 自己计算的 `application_makespan_us` 后取成员 rank 最大值。
同时报告 drain 和 harness 指标，禁止跨 rank 直接相减绝对 timestamp。

### 5.3 收紧“严格串行”表述

现有检查：

```text
next.admit_ts >= previous.completion_observed_ts
```

只在同一 rank 的单调时钟内成立。验证字段改名为：

```text
rank_local_serial_admission_verified
```

文字统一写成“各 rank 本地串行 admission 验证通过”。静态 FIFO/LTF 不宣称下一任务准入前
上一任务已在全部 rank 完成。

只有 `ready_first_serial` 的控制协议在所有 rank 报告本地完成后执行 completion barrier，才
额外报告：

```text
global_completion_barrier_before_next_selection = true
```

这仍表示实验控制协议提供的全局边界，不扩展成未来中央 runtime 的容量保证。

### 5.4 严格 timeline 数据校验

`visualize.py` 不再静默省略缺失或逆序区间。读取 trace 时按 schema version 校验：

- 必填 timestamp 缺失立即失败并指出文件、rank、job、ordinal 和字段；
- 每个区间必须 `end >= start`；
- producer start 不晚于 ready；
- collective call start 不晚于 call return 和 completion observation；
- application wait start 不晚于 underlying wait start 和 wait return；
- validation 只能发生在 communication drain 之后；
- rank-local timestamp 只能相对本 rank origin 使用。

timeline 使用独立轨道，避免后画区间覆盖前画区间：

```text
job-N compute
job-N admission/control wait
job-N communication
job-N application/binding wait
job-N backend wait
harness validation
scheduler state
```

通信条使用半透明填充，wait 条使用轮廓或独立轨道；任何重叠都应同时可见。

### 5.5 Admission 等待来源和 work-conserving idle

不能把整个 `ready -> admit` 都解释为静态队首阻塞。根据 Plan、ready、admit、outstanding
和 completion 事件，在单 rank timeline 离线重建 scheduler state：

| 状态 | 判定 |
| --- | --- |
| `no_ready_work` | 没有 pending task ready，资源空闲 |
| `capacity_busy` | 已达到 max outstanding，不能继续 admit |
| `head_not_ready_with_later_ready` | 静态 Plan 队首未 ready，但后续 task 已 ready |
| `ready_head_scheduler_delay` | 队首 ready、容量可用，但尚未 admit |
| `inflight` | 至少一个数据面 collective outstanding |

汇总新增：

- `work_conserving_idle_us`：无 data collective in flight 且至少一个 task ready；
- `head_of_line_idle_us`：上述时间中由静态队首未 ready 导致的部分；
- `capacity_wait_us`；
- `scheduler_delay_us`。

重建逻辑只使用同一 rank 的时间戳。若事件不足以唯一分类则标为 `unattributed_wait_us`，禁止
强行归因。ready-first 控制面等待另列 `coordination_wait_us`。

### 5.6 批次目录与 run manifest

每次实验创建不可覆盖的独立 batch：

```text
benchmark/phase1.2/result/batches/<batch-id>/
  manifest.json
  profiles/
  raw/<workload>/<scenario>/run-XX.json
  summary.json
  analysis.md
  figures/
```

`batch-id` 使用 UTC 时间和短随机后缀，不能复用固定目录。manifest 至少包含：

- trace schema version；
- git commit 与 worktree dirty 状态；
- 完整命令行和实验 seed；
- backend、world size、PyTorch/CUDA/NCCL、设备和主机信息；
- workload/profile digest；
- scenario 定义和 `max_outstanding`；
- 预期 repetition 数；
- 每个 run 的相对路径、执行顺序、状态和文件 SHA-256；
- batch 是否完整。

runner 先写 `complete=false`，全部运行、校验和 summary 成功后原子更新为 `complete=true`。
失败批次保留用于诊断，但 visualizer 默认拒绝。

summary 和 visualizer 只读取 manifest 列出的文件，不再 `glob("run-*.json")`。因此从 10 次改
为 5 次不会读到旧样本。`result/latest` 可以是文本指针或软链接，但不是数据目录。

## 6. 统计与结论表述

### 6.1 统计输出

每个场景继续报告：

- 所有原始点；
- 中位数；
- P10/P90 样本分位数；
- 最小值和最大值；
- 样本数。

交错运行后，对预先定义的场景对报告同 repetition 的 paired difference 和 paired ratio。
P10–P90 明确标注为“运行结果分布”，不能标为置信区间。若后续需要置信区间，应另加成对
bootstrap，并在方法中记录 resample seed 和次数。

### 6.2 修正分析语言

在完成新实验前，旧 analysis 中过强的结论改为：

> 当前静态 LTF 在该 workload 上表现更差；trace 显示连续选择同一 job 暴露了推进空档，
> 但结果受到校验开销、tail 估计语义偏差、强制串行和运行时间漂移共同影响，不能据此判断
> longest-tail 思想本身无效。

其它统一措辞：

- “严格串行”改为“各 rank 本地串行 admission”；
- “通信完成时间”改为“completion probe 观察时间”；
- “LTF 稳定提升/退化”改为“本批次中位数差异”，除非成对重复和不确定性分析支持稳定性；
- “主要代价是失去并发”改为待对照矩阵验证的假设；
- delayed 的橙色区间不能直接等同于链路空等，必须结合 scheduler-state lane 和
  `work_conserving_idle_us`。

## 7. 代码改动范围

预计修改：

- `src/runtime_comm_scheduler/telemetry.py`
  - 增加 application/underlying wait 与 collective call 边界字段；
- `src/runtime_comm_scheduler/work.py`
  - 在等待绑定前记录 application wait，在调用底层 wait 前记录 underlying wait；
- `src/runtime_comm_scheduler/scheduler.py`
  - 暴露统一的 call-start/call-return telemetry；保留 rank-local completion probe；
- `examples/jobpacer/replay_worker.py`
  - 移出逐 task 校验、加入启动门、记录多层 run 边界、统一 bare/scheduler schema；
- `examples/jobpacer/plan_builder.py`
  - 修正 LTF score 并输出候选 score diagnostics；
- `examples/jobpacer/run_replay.py`
  - 更新 validation 名称、run 级性能汇总和 schema 校验；
- `examples/jobpacer/visualize.py`
  - 严格 trace 校验、独立轨道、scheduler state 和 manifest 输入；
- `benchmark/phase1.2/run_experiments.py`
  - batch manifest、交错顺序、新对照矩阵和成对统计；
- `benchmark/phase1.2/result/analysis.md`
  - 旧结果降级为历史记录或由新批次分析替换；
- 对应 unit/integration tests 和 JobPacer README。

不为这些改动引入数据库、DataFrame 框架、分布式 tracing 系统或交互式 dashboard。统计与
manifest 继续使用 Python 标准库；绘图沿用可选 Matplotlib。

## 8. 测试计划

### 8.1 单元测试

1. 校验移出后，增加校验延迟只改变 validation/harness duration，不改变 application
   makespan；
2. `ScheduledWork.wait()` 在未绑定时正确记录 application wait，绑定后再记录 underlying
   wait；多次 wait 不覆盖首次时间；
3. bare 和 scheduler 的 call/start/return/observed 字段含义一致且单调；
4. completion poll interval 被写入 trace，observed duration 名称不再冒充精确通信时长；
5. corrected LTF 对 overlap 小于/大于通信的场景产生预期 score，并输出稳定 tie-break；
6. start barrier 让所有 job 使用相同 release origin，同时保留 thread-first-run jitter；
7. timeline 对缺失字段、逆序字段和跨 boundary validation 明确失败；
8. scheduler-state 重建覆盖 no-ready、capacity、HOL、scheduler-delay 和 unattributed；
9. manifest 只列本批次文件，旧 run 文件不会进入 summary 或作图；
10. deterministic shuffle 在相同 seed 下复现，在每轮包含每个场景恰好一次。

### 8.2 两 rank Gloo 集成测试

1. Phase 1 仍不使用 Plan admission，launch sequence 可随 ready 改变；
2. deferred validation 后 collective 数值仍正确；
3. static FIFO/LTF 保持每个 ProcessGroup 内顺序一致；
4. `fifo_serial`/`ltf_serial` 通过 rank-local serial admission 检查；
5. globally-ready-first 在两 rank 选择相同 key，不因本地 ready 顺序不同而死锁；
6. `ready_first_serial` 每轮下一选择发生在 global completion barrier 之后；
7. delayed workload 能观测到 static HOL idle，而 ready-first 能选择已全局 ready 的后续 task；
8. application end、communication drain、validation end 和 harness end 按顺序闭合；
9. runner 中断时 manifest 保持 incomplete，visualizer 拒绝将其当正式批次；
10. 小批次 `2 repetitions × all scenarios` 能生成 summary 和 SVG。

### 8.3 正式实验验收

- 所有 run 的 workload/profile digest 与 manifest 一致；
- 所有 task 正确性校验通过；
- 没有 completion observation 晚于 communication drain end；
- 没有必填 timestamp 缺失或逆序；
- 每个 repetition 包含完整场景集合；
- summary 样本数与 manifest 完全一致；
- 图表选择的代表 run 路径写入图旁元数据或 analysis；
- analysis 中每个数字都能追溯到 batch、summary 字段和原始 trace。

## 9. 实施顺序与关口

### Gate A：先修正测量语义

1. 移出逐 task 校验；
2. 拆分 application/underlying wait；
3. 增加 collective call start/return 和 completion-observed 命名；
4. 加入统一启动门和 application/drain/validation/harness 边界；
5. 更新 trace schema 与单元测试。

Gate A 未通过前不重跑性能实验。

### Gate B：修正策略定义和对照

1. 修正并重命名 LTF score 语义；
2. 实现跨 rank globally-ready-first control；
3. 增加 unbounded/serial 对照；
4. 更新 rank-local/global completion 验证措辞和字段；
5. 用两 rank delayed 小场景证明 ready-first 能绕开静态未 ready 队首。

### Gate C：修正 runner、统计和可视化

1. 引入 batch manifest 和不可覆盖目录；
2. 按 repetition 确定性交错运行；
3. summary 使用 manifest 和 paired comparison；
4. timeline 严格校验并拆分轨道；
5. 增加 scheduler-state/idle attribution；
6. 更新 analysis 的限制和措辞。

### Gate D：重新实验

1. 先运行 2 repetitions smoke batch；
2. 检查 trace schema、manifest、summary 和图；
3. 运行当前 CPU/Gloo 的 10 repetitions exploratory batch；
4. 旧批次只保留为历史，不与新批次混合；
5. 在目标 GPU/NCCL 环境运行 30–50 repetitions final batch。

## 10. 完成标准

只有同时满足以下条件，本轮修正才完成：

1. correctness validation 不再影响应用 task 推进和 application makespan；
2. 应用等待绑定和底层 Work wait 可独立统计、独立绘制；
3. LTF score 与 compute/communication overlap 假设一致且名称准确；
4. bare、scheduler overhead、serialization、static HOL 和 policy order 至少有上述成对对照；
5. 场景顺序按 repetition 交错且可由 seed 复现；
6. profile service time 与 replay completion observation 不再混称；
7. application、drain、validation、harness 四个边界闭合；
8. “串行”结论明确限定为 rank-local，除 ready-first global barrier 对照外不声称全局完成；
9. visualizer 对坏 trace fail closed，所有等待来源不再用互相覆盖的色块表达；
10. batch manifest 是 summary、analysis 和图表的唯一样本来源。
