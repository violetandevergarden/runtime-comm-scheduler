# JobPacer Phase 2：将 multi-job replay 接入 scheduler

## 目标与边界

本阶段以 [JobPacer 实验计划](../260914项目计划讨论/JobPacer实验计划.md)的 Phase 1 手工通信—计算交替 workload 为输入，把各 job 线程原本直接调用 `torch.distributed` 的通信路径替换为 `AdmissionScheduler.submit(CommIntent)`。在同一套 replay 中运行 FIFO 和 longest-tail-first（LTF）等**预先确定的静态 Plan**，验证多线程请求能安全地按计划被提交，且非抢占式串行模式下，前一通信物理完成后才发射下一通信。

本次验收聚焦接入正确性、调度顺序、完成边界、故障退出和可观测性。**暂不进行“对比上一步”**：不在此阶段报告与裸发 Phase 1 baseline 的 makespan/throughput 优劣或原因分析；保留相同 workload 配置和原始结果，供后续比较。已有双 all-reduce 实验显示并发可能比强制顺序更快，因此串行模式是待研究的实验条件，不能预设性能收益。

## 当前代码基础与需要补齐的部分

- `src/runtime_comm_scheduler/plan.py` 的 `Plan(version, window_id, entries)` 以有序 `(TaskKey, op, num_bytes)` 表达一个 window；`digest()` 可用于跨 rank 一致性校验。
- `src/runtime_comm_scheduler/scheduler.py` 的 `AdmissionScheduler` 在每个 rank 内只有一个 host launch worker；`submit()` 仅校验并入队，worker 按 `Plan.local_projection()` 的队首发射。`max_outstanding=1` 可使 worker 在上一任务被 completion probe 判定物理完成后才发射下一项。默认值 `0` 不限制 outstanding，**只有 host 提交顺序，不构成物理完成后的串行执行**。
- NCCL/CUDA 使用 `TorchProcessGroupExecutor`、异步 `launch_fn(..., async_op=True)` 和 `WorkIsCompletedProbe`；`ScheduledWork.wait()` 建立消费侧依赖，不可用其 CPU 返回时间代替物理完成时间。CPU/Gloo 使用 `DirectLaunchExecutor`。
- 仓库目前没有 JobPacer 的 Phase 1 multi-job replay 实现；因此先将 Phase 1 的 workload 描述、线程执行器和观测输出落成可运行代码，再增加 scheduler 路径。复用已有 core，不另建一套 scheduler。

## 实验拓扑和 workload 契约

当前先固定两 rank、每 rank 一个 CPU/Gloo 进程；GPU 可用后改为每 rank 一张 GPU 的两 rank NCCL 测试。每个 rank 进程内启动相同数量的 job 线程，所有线程**共享同一个 rank-local scheduler**。跨 rank 不共享 Python scheduler 对象。每个 job 使用成员相同、但逻辑 ID 与实际 ProcessGroup 均独立的 communicator，使不同 job 的 collective 可安全交错；所有 rank 以相同顺序创建这些 ProcessGroup，使用稳定 ID（如 `job-0`），不能把一个 job 的局部任务随意重排到同一 group 内前一个任务之前。实现不把 world size 固定为 2，保留多 rank 的进程、group 和 Plan 投影接口；Phase 2 不增加 4 rank 或更大规模测试。

每个 job 定义为有序的 `compute -> communication -> compute ...` 序列。每个通信项至少记录 `job_id`、job 内 ordinal、op、dtype/shape/bytes、所属 group、前置 compute 时长或算子、后续消费位置。各 rank 的同一逻辑 collective 必须具有一致的 op、shape/bytes 和 key；tensor 值可以因 rank 不同。第一版只使用相同成员的 `all_reduce`，在 CPU/Gloo 上用确定性 sleep 计算窗口跑通；之后的 GPU/NCCL 验收可加入真实 tensor compute。若 GPU tensor 在非默认 producer stream 上产生，记录 CUDA ready event 并交给 `CommIntent`；CPU sleep 结束后才创建/submit intent。

`TaskKey` 的七个字段在所有 rank 确定性生成：`iteration=窗口编号`、`microbatch=0`、`parallelism="jobpacer"`、`process_group_id=job_id`、`layer_id=计算/通信阶段编号`、`bucket_id=0`、`ordinal=job 内稳定序号`。一个 window 覆盖每个 job 的一轮序列；重复轮次使用新 `window_id` 和 key，创建新的 scheduler。不要用线程启动顺序、时间戳、对象地址作为 key。Plan entry 的 `num_bytes` 按实际通信 tensor 大小计算，与 intent 完全一致。

## 接入实现顺序

1. **固化 replay 输入和裸发路径。** 将 Phase 1 workload 写成配置/manifest，保留原始 `torch.distributed` 裸发模式，抽出公共的 job 线程执行器、tensor 初始化、计算窗口、结果校验与 trace 输出。两条路径使用相同 workload，不让 scheduler 路径改动计算位置或 collective 参数。进程初始化、ProcessGroup 创建、warmup、销毁在主线程管理；job 线程只执行 workload。
2. **建立共享 Plan。** 在启动 job 线程前，从 manifest 构造该 window 全部通信项的确定性 entry。两 rank 安装同一 version/window/entries，启动前交换或由 driver 比较 `digest()`；不一致即失败。Plan 必须在每个 job 内保持原始通信顺序，并满足其 `compute -> communication -> consumer` 依赖。初期直接由相同配置在各 rank 本地生成静态 Plan，不引入动态 policy 或 rank 间 pending 协商。
3. **替换通信调用点。** 当线程运行到原始 collective 调用位置时，创建本 rank 的 `CommIntent`：填入 key、op、tensor、实际 ProcessGroup、bytes、ready event（如有）、`device`、必要的 `keepalive`，`launch_fn` 闭包调用原始 `dist.all_reduce(..., group=job_group, async_op=True)`。调用共享 scheduler 的 `submit()`，保存返回的 `ScheduledWork`。在原 workload 原本消费结果的位置调用 `wait()`，然后检查 collective 结果；不得在 `submit()` 后无条件立刻等待并把并发 compute 窗口抹掉。对异步 NCCL，消费端在正确 CUDA stream 上 `wait()` 后再使用 tensor。
4. **启用严格串行 admission。** 每 rank 构造 `AdmissionScheduler(plan, local_group_ids=所有 job group ID, executor=DirectLaunchExecutor(), max_outstanding=1)`；GPU/NCCL 验收时把 executor 换成 `TorchProcessGroupExecutor(local_rank)`。worker 的 completion probe 释放 outstanding slot 后才允许计划下一项。记录 `submit_ts`、`complete_ts` 及相邻任务的顺序关系；检查 `next.admit_ts >= previous.complete_ts`（允许时间戳记录粒度内的微小误差需明确说明）。不用 `work.wait()` 或 `torch.cuda.synchronize()` 人为控制串行。
5. **收尾和故障处理。** 所有 job 线程结束后由主线程 `finish_window(timeout=明确上限)`，确认 manifest 中每个本地 key 恰好 submit 一次且物理完成；随后读取 `sequence_log()`、`group_sequence_log()`、`timings()`，再 `close()` 和销毁 ProcessGroup。若任一线程失败，通知其它线程、停止继续提交、关闭 scheduler、按有界超时退出；不能让缺失的 Plan 队首或未绑定的 Work 使进程无限等待。错误报告保留首个异常和 pending key。

## 静态策略如何生成 Plan

FIFO 的定义是**预先固定的全局 manifest 顺序**，例如按 job 配置顺序轮转各 job 的第 0、1、… 个通信项；它不是运行时谁先到 pending 就先发射。LTF 在启动前，根据每个通信项完成后该 job 尚需的计算与通信时长之和估计 tail（来自 workload 配置或单独 profiling），在当前各 job 的下一项中选择 tail 最大者，选择后推进该 job 的指针，直到生成完整 Plan。相同 tail 用稳定 `job_id`/ordinal 破同分。记录 tail 的定义、估计来源与生成后的 key 序列，使两 rank 独立生成一致结果。

必须保证 Plan 是每个 job 序列的合法交错：不能把 job 内第 `i+1` 个 collective 排在第 `i` 个之前。若 job 的后续任务只能在前一通信消费后产生，Plan 队首等待它时，其前置通信必须已在 Plan 中更早的位置。第一版不让 Plan 等待一个尚未由任何线程产生且其产生依赖于 Plan 后续项的任务；生成后做依赖拓扑校验。静态 LTF 不能抢占已发射的 collective，也不会因实际 compute 时长扰动而临时跳过队首；此局限留给 Phase 3 runtime 决策。

建议至少准备三套小 workload：两个 job 均含 2–3 个同尺寸 all-reduce（验证交错）；一个 job 有较长剩余 compute tail、另一个较短（使 FIFO 与 LTF 顺序不同）；一个 job 的下一项因 sleep/compute 延迟到达（验证队首等待及有界退出）。通信尺寸包含小消息与较大消息，但控制总 GPU 内存，先跑正确性再增大规模。

## 输出、验证与验收

每次运行保存可重放配置、随机种子、backend、world size、PyTorch 版本、Plan version/window/digest、策略名及完整 key 顺序；GPU 测试再补充 GPU/NCCL 版本。每 rank 记录每个 task 的 `ready_record_ts`、`admit_ts`、`submit_ts`、`complete_ts`、`first_wait_ts`、错误阶段，以及线程侧 compute 起止、提交和消费位置。用同一进程的单调时间戳判断局部间隔；跨 rank 时间戳不直接相减。输出每 job 的正确性与 makespan 作为后续分析原始数据，本阶段只检查数据完整性和运行稳定性。

按以下顺序验收：

1. 当前两 rank Gloo/CPU 小 workload：多线程共享 scheduler 能按 FIFO 与 LTF Plan 发射；各 rank 的 Plan digest 相同，每个 group 的发射序列与 Plan 子序列一致，all-reduce 结果正确；重复 key、metadata mismatch、缺失 key 有明确错误且有界退出。
2. 当前两 rank Gloo/CPU 严格串行：在 `max_outstanding=1` 模式中，各 rank 分别证明每个相邻任务的 admission 不早于前一项被 completion probe 记录的物理完成。另跑 `max_outstanding=0` 的机制对照，确认它只保证 host 顺序，避免混淆两种模式。
3. GPU 可用后的两 rank NCCL/CUDA 验收：复跑 FIFO/LTF、结果正确性、group 顺序和物理完成后的串行条件；验证 ready event 与消费 stream 依赖。仅有两块 GPU，不安排更多 GPU rank 测试。
4. 延迟到达与失败：乱序 submit 最终仍按 Plan 发射；缺失队首触发 `finish_window` 的明确错误或超时；单 rank 异常时 driver 对全部 rank 设进程级超时并收集日志，不留挂起进程。
5. 重复运行固定种子场景，验证 key、Plan digest、发射序列和结果可复现；记录时序波动但暂不解释与 Phase 1 baseline 的性能差异。

交付为可从命令行选择 `bare`/`scheduler`、`fifo`/`ltf`、`max_outstanding`、backend、world size 和 workload 的 replay driver，静态 Plan 构造器、接入代码、针对顺序/依赖/完成边界的测试，以及运行记录。当前 Phase 2 完成判据是：在两 rank CPU/Gloo 真实 collective 上，所有 job 的任务正确结束，跨 rank 的同 group 顺序一致，FIFO/LTF 产生预期不同的合法顺序，严格串行条件被 telemetry 证实，错误场景有界退出。GPU/NCCL 两 rank 验收作为后续环境可用时的补充关口；系统接口仍以多 rank 为目标，不以大规模测试为本阶段前提。
