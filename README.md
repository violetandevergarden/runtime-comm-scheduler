# Runtime Communication Scheduler

面向单个混合并行训练任务的 runtime-adaptive collective scheduling 研究原型。项目研究如何在训练框架、PyTorch distributed runtime、NCCL 和网络之间建立可逐步下沉的 schedule layer，使训练 DAG 语义能够影响真实 collective 的执行。

本项目与 `SimAI/simai-flow-scheduler` 互补：SimAI 已用于 flow-level trace replay、带宽分配和 policy 验证；本项目研究真实 training stack 中究竟能够观察、控制和安全重排哪些通信事件。

## 项目范围与长期目标

当前聚焦单个 dense Megatron-style training job。DP、TP、PP 都可以作为训练语义来源，但第一个真实集成点只处理 DP gradient synchronization。multi-tenant 暂不纳入项目接口和实现，未来如有需要再扩展。

长期目标是支持从 high-level collective admission 到更细粒度 data-plane control 的演进：

```text
framework semantic scheduling
  -> ProcessGroup-level enforcement
  -> NCCL communicator/channel/chunk control
  -> NIC/network traffic control
```

当前不预设必须在哪一层结束。每次向下扩展都应由上一层机制不足以实现目标 policy 的证据驱动。

## 整体架构

```text
训练框架语义适配层
  - 识别 layer、microbatch、DP/TP/PP 角色、producer 和 consumer
  - 绑定本地 tensor、CUDA ready event 和原始 collective launcher
                         |
                         v
每 rank 的 schedule layer
  - 根据共享执行计划校验 CommIntent
  - 执行 runtime admission、delay 和安全的跨 process group 发射仲裁
  - 返回 ScheduledWork 并记录 telemetry
                         |
                         v
ProcessGroup / NCCL 数据面
  - 执行已经通过 admission 的 collective
  - 初期完全复用原始 ProcessGroupNCCL 和 NCCL
  - 后续可成为更细粒度控制的修改对象
                         |
                         v
GPU 互连 / NIC / 网络
```

前两层有意分离：训练框架是唯一掌握 training-DAG 语义的层级；schedule layer 是实际 gate collective submission 的执行点。

## Plan + Admission 调度模型

调度器不依赖单一 priority queue，而是结合两个互补机制。

### Plan：基础执行计划

Plan 是一个概念上的、带版本的基础执行计划，描述一个 scheduling window（初期为一个 iteration）的预期执行结构。它由 `CommIntent` 的确定性元数据和有序的 `TaskKey` 序列组成，不额外引入独立的 intent 类型：

- 每个预期 collective 的稳定 `TaskKey`；
- 每个 rank 上的有序 task 序列；
- 每个 process group 诱导出的 collective 子序列；
- collective 类型、字节数、训练语义和预测时延等 `CommIntent` 元数据。

Plan 在各 rank 间共享，初期只允许在 iteration 之间的安全边界切换。它建立了保证 collective 正确性所需的顺序契约。

### Dynamic Admission：运行时准入

运行时，训练框架在每个 rank 创建本地 `CommIntent`，将计划中的 `TaskKey` 绑定到实际 tensor、CUDA ready event 和原始 `torch.distributed` launcher。Admission 层可以：

- 等待 producer ready；
- 延迟计划中的 collective；
- 限制 outstanding collective 数量；
- 在不同 process group 之间选择 ready task 的发射时机；
- 记录运行时测量，为下一版本 plan 提供数据。

Admission 不能在本地独立改变同一个 process group 内的 collective 顺序。只有在每个受影响 process group 的诱导序列仍然对其所有成员 rank 一致时，才允许进行跨 group 仲裁。

## 任务生命周期

```text
CommIntent
  -> READY
  -> WAITING_FOR_ADMISSION
  -> ADMITTED
  -> SUBMITTED
  -> COMPLETED | FAILED
```

`READY`、`SUBMITTED` 和 `COMPLETED` 是不同事件。scheduler 只能在 `SUBMITTED` 之前介入；collective 一旦提交给 NCCL，就不能在当前层级取消或抢占。对 NCCL，`ScheduledWork.wait()` 表示把完成依赖接入 consumer current stream，不等同于 GPU 物理完成；后者由 completion probe 独立观察。

## 目录结构

- `docs/design/architecture.md`：长期维护的系统边界、事件模型、机制和正确性约束。
- `docs/phase1-plan.md`：runtime scheduler 核心 Phase 1 的目标、设计、里程碑和验收标准。
- `docs/JobPacer/plan/phase1.md`：JobPacer 多 job 裸发 replay baseline（Phase 1）及 workload 契约。
- `docs/JobPacer/plan/phase2.md`：JobPacer 接入 scheduler 的调度实验计划。
- `docs/experiments/`：实验记录和 profiler/Nsight 产物索引。
- `src/runtime_comm_scheduler/`：机制接口和后续实现。
- `src/runtime_comm_scheduler/adapters/`：框架语义适配器，首先适配 Megatron。
- `tests/`：单元测试以及 Gloo/NCCL distributed harness。
- `examples/`：最小可运行示例。
- `configs/`：预留给后续实验配置。

## 当前状态

- **M0（Gloo 行为 harness）已完成**：两 rank 场景覆盖 FIFO、固定重排、延迟
  ready 与 divergence 挂起，记录见 [docs/experiments/m0-m4.md](docs/experiments/m0-m4.md)。
- **M1（核心 schema 与 plan 校验）已完成**：确定性 `TaskKey`、`CommIntent`
  生命周期、`Plan` 表示与五类校验。
- **M2（V0 同步 admission）已完成**：`AdmissionScheduler` 的 submit → 校验 →
  准入 → 发射路径，两 rank Gloo harness 重现 FIFO 与固定重排且 sequence log
  一致，乱序提交被强制为计划顺序，错误场景 fail-stop 有界退出。
- **M3（NCCL/CUDA event 语义）已完成**：在两台 RTX 3090 上验证 CUDA ready
  event 的 producer dependency。M3 当时把 `Work.wait()` 错误解释为 GPU 物理
  完成等待并加入了设备同步；该解释和补偿已由 M4.5 纠正，历史记录保留在实验
  文档的勘误中。
- **M4（异步 admission worker）已完成**：deferred launch 移到专门 worker
  线程 + 显式 gate stream。producer 的 `submit` 只校验 + park +
  唤醒、立即返回；worker 线程按 plan 顺序 drain 发射，`out_of_order_submit`
  经 worker 仍强制计划顺序，`delayed_ready` 的 stream dependency 经 comm
  stream 正确，16 intent 场景 producer 继续执行与 collective 重叠。关口实验
  同时记录了硬限制：同一 communicator 多线程并发提交不安全（会打挂进程）。
  详见
  [docs/experiments/m0-m4.md](docs/experiments/m0-m4.md)。
- **M4.5（scheduler/Work 语义重构）已完成**：scheduler
  收敛为单 worker、rank-wide plan 投影和全局 outstanding；execution plane
  按 process group 使用独立 gate stream；`ScheduledWork.wait()` 只透传底层
  stream dependency，physical completion 由独立 probe 推进。原 3090 容器下线后，
  经批准使用 2× RTX 3080 Ti 完成 GPU/NCCL capability gate；实施设计见
  [docs/m4.5-refactor-plan.md](docs/m4.5-refactor-plan.md)，实验记录见
  [docs/experiments/m4.5-gpu3080.md](docs/experiments/m4.5-gpu3080.md)。
- **待开发**：Megatron DP adapter 与 plan 版本切换，
  详见 [docs/phase1-plan.md](docs/phase1-plan.md)。

### JobPacer replay

JobPacer Phase 1 提供 scheduler-free 的多 job 基线：每个 rank 内以线程执行
通信—计算交替 workload，通信通过真实 Gloo/NCCL `all_reduce` 裸发，并输出每个
job 的 makespan、实际提交顺序和逐 task trace。Phase 2 使用同一 workload 与输出
格式，将通信提交替换为 `AdmissionScheduler`。

```bash
PYTHONPATH=src:. python examples/jobpacer/run_phase1.py \
  --workload balanced --backend gloo --world-size 2 \
  --output artifacts/jobpacer-phase1-gloo.json
```

可用 workload：`balanced`、`tail`、`delayed`，也可传入符合
[`phase1.md`](docs/JobPacer/plan/phase1.md) schema 的 JSON manifest。
