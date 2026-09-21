# JobPacer Phase 1：无调度 multi-job replay baseline

## 目标

Phase 1 使用真实 `torch.distributed` collective 重放多个通信-计算交替的线性 job，
但不创建或调用 `AdmissionScheduler`。每个 rank 是一个进程；同一 rank 内，每个 job
由独立线程执行。不同 job 使用独立 ProcessGroup，因此它们可以无协调地提交通信并
竞争相同的 GPU、PCIe 或网络资源。

Phase 1 与 Phase 2 必须使用同一 workload manifest 和线程执行器。两阶段唯一的核心
差别是通信调用点：

```text
Phase 1: dist.all_reduce(..., async_op=True)
Phase 2: scheduler.submit(CommIntent(...))
```

## Job 定义格式

定义位于 `examples/jobpacer/workloads.py`，层级为：

```text
Workload
  name, seed
  jobs: tuple[Job, ...]

Job
  job_id
  ranks: optional tuple of global ranks
  communications: tuple[CollectiveComm, ...]

CollectiveComm
  id
  op
  num_bytes
  producer_compute_s
  consumer_compute_s
  estimated_comm_s
```

一个 `CollectiveComm` 表示以下线性片段：

```text
producer compute/sleep
  -> collective ready and raw async submission
  -> independent consumer-side compute/sleep
  -> Work.wait() and result consumption
```

`producer_compute_s` 位于通信前；`consumer_compute_s` 位于异步提交和消费之间，形成
计算通信重叠窗口；`estimated_comm_s` 只供 Phase 2 的 LTF 静态 Plan 使用，不控制
Phase 1 的实际通信时长。通信项的 `id` 必须从 0 连续递增。每个 job 的
`process_group_id` 等于 `job_id`。

内置 workload 有 `balanced`、`tail` 和 `delayed`。也可以传入相同 schema 的 JSON
文件。当前计算窗口由 sleep 表示，通信是实际 Gloo/NCCL all-reduce，不使用模拟通信。

## 运行

CPU/Gloo baseline：

```bash
python examples/jobpacer/run_phase1.py \
  --workload balanced --backend gloo --world-size 2 \
  --output artifacts/jobpacer-phase1-gloo.json
```

两 GPU NCCL/PCIe baseline：

```bash
python examples/jobpacer/run_phase1.py \
  --workload balanced --backend nccl --world-size 2 \
  --output artifacts/jobpacer-phase1-nccl.json
```

为保证与 Phase 2 可直接比较，仍接受 `--policy` 参数来生成稳定的 Plan/key 元数据，
但 bare 执行不按 Plan 排序；实际提交顺序记录在每个 rank 的 `launch_sequence` 中。

## 输出与验收

输出 JSON 包含原始 workload、每 rank 实际提交顺序、每个 task 的 producer ready、
裸 collective 调用、返回、consumer compute、wait 和消费完成时间，以及：

- 每个 job 在各 rank 的 `makespan_us`；
- job 的全局 makespan，即成员 rank 中的最大值；
- workload makespan；
- collective 数值正确性；
- 完整的逐 task trace。

时间戳来自各进程单调时钟，只在同一 rank 内相减；跨 rank makespan 使用各 rank 自己
计算出的 duration，不直接相减绝对时间戳。Phase 2 比较实验必须使用相同 workload、
backend、world size 和进程拓扑。
