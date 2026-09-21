# JobPacer Phase 1/2 修正与可视化实施计划

## 1. 背景与目标

Phase 1/2 已能使用同一 workload 分别运行无调度裸发和静态 Plan 调度，并输出每个 job
的 makespan、任务 trace、Plan 顺序与 completion telemetry。当前需要补齐三个问题：

1. workload 转换器仍放在 `src/` 顶层，但它属于 JobPacer 实验工具，不属于 scheduler
   核心库；
2. Phase 1 只记录 `wait_return_ts`，没有独立观察 collective 的物理完成时刻，导致其通信
   区间与 Phase 2 的 `submit_ts -> complete_ts` 不同口径；
3. 结果目前只有 JSON 和文字分析，缺少能够直接展示 job 竞争、串行发射和队首阻塞的图。

本次修正目标是：统一 Phase 1/2 的物理完成观测口径，在不改变 Phase 1 裸发语义的前提下
生成可比较的 trace，并提供一个最小可复用的可视化入口。修正后重新执行完整实验，不混用
旧 trace。

## 2. 文件整理：移动 workload 转换器

将：

```text
src/built_workload_from_previous_benchmark.py
```

移动并重命名为：

```text
examples/jobpacer/workload_builder.py
```

该脚本只服务于 JobPacer benchmark 的 chain-DAG 到 replay workload 转换，不应成为
`runtime_comm_scheduler` 核心包的一部分。移动时保留现有行为和命令行接口，只更新：

- `tests/unit/test_workload_converter.py` 的导入路径；
- `benchmark/phase1.2/README.md` 中的转换命令；
- 脚本 docstring 和仓库内所有旧路径引用。

新的转换命令为：

```bash
python examples/jobpacer/workload_builder.py \
  benchmark/phase1.2/dag \
  benchmark/phase1.2/workloads
```

转换器继续遵循以下约束：

- communication 必须显式提供真实 `num_bytes`，不能根据执行时间反推消息大小；
- 相邻 communication 仍是多个 collective，不能合并；
- 零时长 communication 仍保留，`duration` 只是调度估计，不决定通信是否存在；
- 每段 compute 只归属一次，避免同时成为前一通信的 consumer compute 和后一通信的
  producer compute；
- 输入必须是互不相交的 chain，分叉、环、未知依赖和重复 ID 明确失败。

## 3. Phase 1 物理完成观测

### 3.1 语义要求

Phase 1 必须继续通过：

```text
examples/jobpacer/run_phase1.py
```

运行，并保持以下语义：

- 不创建或调用 `AdmissionScheduler`；
- 不根据 FIFO、LTF 或其它 Plan 对 job 排序；
- 每个 job 线程到达通信点后直接调用
  `dist.all_reduce(..., async_op=True)`；
- 不限制 outstanding 数量，不因为 observer 等待其它 job；
- 不改变 collective 的提交顺序、并发关系或消费位置。

新增逻辑只能观察通信完成，不能参与 admission。

### 3.2 复用 completion probe

复用核心已有的 `WorkIsCompletedProbe`，其 `is_completed()` 对当前已验收的 Gloo 和 NCCL
路径表示物理完成。Phase 1 增加一个 rank-local observer：

```text
job threads
  -> raw async collective
  -> register underlying Work and submit timestamp
                         |
                         v
single completion observer thread
  -> WorkIsCompletedProbe.is_completed(work)
  -> record complete timestamp
```

observer 使用单个后台线程和待观察 work 集合，不为每个 collective 创建新线程。轮询间隔
复用 `--completion-poll-interval-s`，默认 `0.001` 秒，与 scheduler 的默认值一致。

observer 必须提供有界收尾：所有 job 线程结束后，主线程等待已注册 work 全部被观察为完成；
超过 `finish_timeout` 时明确失败，不能留下后台线程或挂起进程。任一 probe 异常应保留首个
错误并使本 rank 有界退出。

### 3.3 统一 trace 字段

修正后 Phase 1 和 Phase 2 的 task trace 使用同一时间含义：

| 字段 | Phase 1 bare | Phase 2 scheduler |
| --- | --- | --- |
| `ready_record_ts` | producer compute 结束 | producer compute 结束 |
| `submit_call_ts` | job 线程调用裸 collective | job 线程调用 `scheduler.submit()` |
| `admit_ts` | 等于实际 `submit_ts` | scheduler 准入时刻 |
| `submit_ts` | 底层 collective 实际发射 | worker 发射底层 collective |
| `complete_ts` | observer 观察到物理完成 | scheduler probe 观察到物理完成 |
| `actual_duration_us` | `complete_ts - submit_ts` | `complete_ts - submit_ts` |
| `first_wait_ts` | consumer 首次等待 | consumer 首次等待 |
| `wait_return_ts` | 底层 Work wait 返回 | ScheduledWork wait 返回 |

Phase 1 的 `complete_ts` 不能继续使用 `consumer_end_ts` 或 `wait_return_ts` 代替。对 NCCL，
`Work.wait()` 可能只向 consumer stream 插入依赖，因此 observer 必须继续独立推进，不能把
CPU wait 返回解释为 GPU 物理完成。

为避免 measurement bias，tensor 分配和各 ProcessGroup warmup 继续放在
`replay_start_ts` 之前。

## 4. 可视化工具

### 4.1 文件和依赖

新增：

```text
examples/jobpacer/visualize.py
```

使用 Matplotlib 输出 SVG。当前环境没有安装 Matplotlib，因此将其声明为可选依赖，而不是
让 scheduler 核心安装强制依赖：

```toml
[project.optional-dependencies]
visualization = ["matplotlib>=3.8"]
```

`visualize.py` 应延迟导入 Matplotlib；缺少依赖时给出明确安装提示。数据解析、代表运行选择
和区间构造保持为不依赖 Matplotlib 的纯函数，以便单元测试不需要绘图库。

### 4.2 Timeline 图

命令形式：

```bash
python examples/jobpacer/visualize.py timeline \
  --result-dir benchmark/phase1.2/result \
  --workload delayed-mixed-small \
  --rank 0 \
  --output benchmark/phase1.2/result/figures/delayed-timeline-rank0.svg
```

每个 workload 的 timeline 使用三列：

```text
Phase 1 bare | Phase 2 FIFO | Phase 2 LTF
```

每个 job 使用 compute 和 communication 两条子轨道。横轴是相对该 rank
`replay_start_ts` 的毫秒数。视觉编码固定为：

- 灰色：producer compute；
- 绿色：consumer-side overlap compute；
- 橙色：`ready_record_ts -> admit_ts`，表示 ready 后的 admission 等待；
- 蓝色：`submit_ts -> complete_ts`，表示观测到的物理通信区间；
- 红色：`first_wait_ts -> wait_return_ts`，表示 consumer wait 区间；
- 三角标记：ready；
- 竖线或短标记：admit、submit、complete。

每个场景从 10 次重复中选择 workload makespan 最接近该场景中位数的运行，不手工挑选最好
或最坏结果。标题中记录原始 `run-XX` 文件名、rank、策略和 makespan，保证图可追溯。

rank 间单调时钟没有同步。工具一次只画一个 rank，通过 `--rank` 切换；禁止把两个 rank 的
绝对时间戳直接合并到同一时间轴。

### 4.3 汇总分布图

命令形式：

```bash
python examples/jobpacer/visualize.py summary \
  --result-dir benchmark/phase1.2/result \
  --output benchmark/phase1.2/result/figures/makespan-summary.svg
```

汇总图按 workload 分面，横轴为 Phase 1 bare、Phase 2 FIFO、Phase 2 LTF，纵轴为
makespan。每次运行显示一个原始散点，并叠加中位数和 P10–P90；不使用只显示均值的柱状图。
至少包含：

- workload makespan；
- job-0/job-1 makespan；
- 样本数和时间单位。

### 4.4 重点解释图

第一版不增加通用绘图框架，只在 timeline 基础上支持两个重点场景：

1. `delayed-mixed-small`：放大首次通信，展示 job-1 已 ready 但静态 Plan 等待 job-0 的
   head-of-line blocking；
2. `tail-small`：对比 FIFO 的轮转顺序与 LTF 连续推进 job-0 后对短 job 完成时间的影响。

若普通 timeline 已足够清楚，不另写专用 renderer；通过 CLI 的时间范围或复用同一绘图函数
完成局部放大。

## 5. 重新执行实验

completion 口径改变后，旧 trace 不用于最终作图或新分析。重新运行完整矩阵：

```text
3 workloads
× 3 scenarios (Phase 1 bare, Phase 2 FIFO, Phase 2 LTF)
× 10 repetitions
= 90 replay runs
```

Phase 1 必须由 `run_phase1.py` 启动。Phase 2 使用静态 Plan 且
`max_outstanding=1`，保证前一通信被 probe 判断物理完成后才准入下一通信。三个场景使用同一
workload 和同一离线通信 profile，实验顺序执行，避免不同 replay 互相制造 CPU/Gloo 竞争。

每个 workload 的通信 profile 保持：

- warmup 3 次，不计入样本；
- 正式测量 10 次；
- p50 用作 LTF `estimated_comm_s`；
- profile、backend、world size、group membership 与 replay 一致。

本地 10 次重复用于功能验证、明显趋势和 head-of-line blocking 分析。若在目标 GPU/NCCL
环境形成论文或正式性能结论，应提升到至少 30–50 次，并报告原始点、中位数和 P10–P90，
而不是只报告一次运行。

## 6. 结果目录

重新运行后统一写入：

```text
benchmark/phase1.2/result/
  profiles/
  raw/
    <workload>/
      phase1_bare/run-XX.json
      phase2_fifo/run-XX.json
      phase2_ltf/run-XX.json
  figures/
    <workload>-timeline-rank0.svg
    makespan-summary.svg
  summary.json
  analysis.md
```

`summary.json` 和 `analysis.md` 必须由新 trace 更新。分析至少回答：

- Phase 1 是否出现多个实际发射序列和通信重叠；
- Phase 2 是否 10/10 次匹配 Plan；
- Phase 2 是否 10/10 次满足严格串行 admission；
- FIFO/LTF 相对 Phase 1 的 workload/job makespan 变化；
- delayed workload 的 ready-to-admit 等待是否构成静态队首阻塞；
- LTF 的目标是否与整体 makespan、短 job 完成时间发生冲突。

## 7. 测试与验收

### 7.1 单元测试

1. workload builder 移动后，三份 checked-in DAG 转换结果仍与
   `benchmark/phase1.2/workloads/` 完全一致；
2. fake Work 从未完成切换为完成时，Phase 1 observer 只记录一次 `complete_ts`；
3. observer 不改变 work 注册顺序和 job launch sequence；
4. observer timeout 和 probe 异常有明确错误并有界退出；
5. 代表运行选择函数确实选择最接近中位数的 trace；
6. timeline 区间全部以本 rank `replay_start_ts` 归一化，拒绝缺失或逆序时间戳。

### 7.2 两 rank 验收

- Phase 1 的 collective 正确，实际发射顺序不受 Plan 强制；
- Phase 1 每个 task 都满足 `submit_ts <= complete_ts`，且 `complete_ts` 来自 probe；
- Phase 2 FIFO/LTF 的 scheduler/group sequence 与 Plan 一致；
- `max_outstanding=1` 时所有相邻任务满足
  `next.admit_ts >= previous.complete_ts`；
- 90 次 replay 全部 `validation.status == "ok"`；
- timeline 和 summary SVG 能从新结果一条命令重新生成。

## 8. 实施顺序

1. 移动并重命名 workload builder，更新引用和转换测试；
2. 在 Phase 1 裸发路径加入 completion observer 和统一 telemetry；
3. 用小型 Gloo replay 验证 observer 不改变裸发顺序与正确性；
4. 实现 `visualize.py` 的纯数据解析、代表运行选择和 timeline/summary 输出；
5. 安装可选 visualization 依赖并运行绘图 smoke test；
6. 重新运行 3 × 3 × 10 实验，覆盖旧的正式结果；
7. 生成 SVG，更新 `summary.json` 和 `analysis.md`；
8. 完成全量单元/集成测试和 90 份 trace 的最终一致性检查。

## 9. 本阶段不做

- 不改变 FIFO/LTF 算法；
- 不让 Phase 1 使用 scheduler 或 Plan admission；
- 不实现跨 rank 时钟同步；
- 不把 CPU/Gloo 结果外推为 GPU/NCCL 结论；
- 不增加交互式 dashboard、Web 服务或新的绘图框架；
- 不在本阶段实现动态跳过未 ready 队首，该机制留给后续 runtime scheduling 实验。
