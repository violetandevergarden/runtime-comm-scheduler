## 1. 基本定位： mechanism 而非 policy

本项目的 scheduler 位于 training framework 与原始 PyTorch ProcessGroup 之间。它不
替换 NCCL，也不直接控制 packet、flow、NCCL channel 或 collective 内部 chunk；它控制
的是一个更高层、但真实存在的边界：**某个已经具备训练语义的 collective，何时由 host
提交给 ProcessGroup**。

```text
framework / replay
  -> 产生带 tensor、ready dependency 和语义信息的 CommIntent
  -> scheduler mechanism：保存、校验、admit、launch、观察完成
  -> 原始 ProcessGroup / NCCL
```

这种分层的意图是把“安全地延迟和提交真实 collective”做成稳定机制，把“为什么要让 A
先于 B、要延迟多久、不同 tenant 如何公平”留给可替换的 policy

## 2. 当前已经实现的机制

### 2.1 共享、确定性的静态 Plan

一个 `Plan` 描述一个 scheduling window 的有序 task manifest。每项包含稳定的
`TaskKey`、collective operation 和字节数；所有 rank 安装同一份带 version/window/hash
的文档。`TaskKey` 包含 iteration、microbatch、parallelism、logical process group、
layer、bucket 和 ordinal 等确定性训练语义，避免把 Python 对象地址或本地时间戳当作
跨 rank 身份。

Plan 的顺序是第一版保守的**全局逻辑 host launch order**。每个 rank 根据自己参与的
logical process group 取得该顺序的本地投影；它不需要参与其他 group 的占位
collective。对同一 group 而言，所有成员得到相同的诱导子序列，这是 collective
matching 的核心正确性条件。

当前 Plan 是精确 manifest，而不是泛化的 template：key、operation、bytes 都必须与
运行时 intent 相同；一个 scheduler 在创建时绑定一份 Plan，尚不支持 window 内替换。

### 2.2 `CommIntent`：运行时执行绑定

训练框架或 replay 在某个 collective 原本要调用的位置创建 `CommIntent`。它持有本 rank
专属的 tensor、ProcessGroup、launcher、ready CUDA event、keepalive 引用以及可选的
producer/consumer 语义；这些对象不进入 Plan，也不在 rank 间传递。

intent 的生命周期是：

```text
CREATED -> READY -> WAITING_FOR_ADMISSION -> ADMITTED -> SUBMITTED
                                                        -> COMPLETED | FAILED
```

`SUBMITTED` 是不可逆边界：任务交给 ProcessGroup/NCCL 后，当前层只能观察完成或报告
错误，不能取消、抢占或重新排序。

### 2.3 `AdmissionScheduler`：唯一的 rank-local launch gate

每个 scheduler 对应一个 rank/GPU，并拥有唯一的 host worker。training/replay thread
调用 `submit()` 时只做校验并把 intent 放入 pending 区域，绝不直接调用底层 collective；
worker 是唯一能实际 launch 的组件。

在当前全序模式中，worker 只考虑本地 plan 投影的第一个尚未 launch 的 task。它在以下
条件都满足时才 admission：

- 该 task 已提交到 pending 表；
- CPU 侧 ready condition 已满足；
- 还未达到 rank 内全局 `max_outstanding` 上限。

因此，当前“策略”能够通过共享 Plan 决定跨 communicator 的 host submission 顺序；也
能够通过 intent 的到达/ready 时机以及 outstanding 上限影响何时发射。Plan 本身目前
**没有** release time、延迟时长、优先级、deadline 或 tenant weight 字段，scheduler
也**没有**可插拔的 policy callback。因此“由 plan 指定延迟”是合理的后续设计方向，
不是 M4.5 已经提供的能力。

全序是为避免 overlapping communicators 的循环 launch order 而采用的保守基线。它约束
CPU host 调用顺序，并不等待前一项通信物理完成；不同 ProcessGroup 的 GPU collective
仍可能发生重叠。

### 2.4 CUDA / Work / completion 语义

后台 worker 不在原 producer stream 上执行 launcher。为保持依赖，executor 在相应
process group 的 gate stream 上等待 intent 的 ready CUDA event，然后在该 stream
context 调用原始异步 ProcessGroup。不同 group 使用不同 gate stream，以避免因为共用
gate stream 而凭空建立跨 group 的 producer dependency。

返回给框架的是延迟绑定的 `ScheduledWork`：若任务尚未 launch，`wait()` 先等待真实
底层 Work 绑定；绑定后将 wait 交给底层 Work。对 NCCL，这通常只把通信完成 event
接入调用处的 consumer current stream，CPU 可以在通信物理完成前返回。scheduler 通过
独立 completion probe 观察物理完成、记录完成时间并释放 outstanding capacity；不会用
`torch.cuda.synchronize()` 把依赖传递变成设备级同步。

发生 launch、completion 或底层 Work 错误时，scheduler 采用 fail-stop：相关或尚未
完成的 work 会被唤醒并失败，而不是永久停在等待绑定的状态。

### 2.5 当前可获得的观测

每个任务记录 intent、ready record、admit、launch start、底层 submit、首次 consumer
wait、physical completion 和 error 等本 rank 单调时钟时间戳。这些数据可用于分析排队
等待、提交开销、完成时长和 consumer 使用关系；不同 rank 的本地时钟不能未经校准直接
作绝对时间比较。

## 3. 自定义 policy 在当前框架中的可行方式

对于静态、trace-driven workload，最简单且完全受当前机制支持的方式是把 policy 做成
**window 开始前的共享 Plan 生成器**：它读取 workload/历史 telemetry/拓扑信息，输出一
份所有 rank 一致的 task 全序。每个 rank 安装相同 Plan，scheduler 自动执行其本地投影。

这适合比较例如：

- tenant A、B 的 collective 是按 tenant 连续执行，还是按 layer/bucket 交错；
- 对同一批 trace task，哪些跨 communicator host submission order 更好；
- 不同 `max_outstanding` 与静态 release arrangement 的影响。

一个有效的静态 policy 必须同时满足：

1. 同一 logical process group 的成员看到相同的 collective 类型、payload 元数据和
   诱导序列；
2. 所有受影响 rank 从同一共享规则/Plan 得出顺序，不能各自根据本地队列任意换序；
3. 不让 task 在 producer ready 前实际 launch；
4. 不把已 `SUBMITTED` 的 task 纳入后续重排；
5. 不以 consumer `wait()` 或设备同步作为 admission/完成判断。

静态 plan 生成器是 multi-tenant trace replay 的推荐起点：它足以产生一组可控的 policy
对照，又不会过早把在线分布式协调问题混入实验结果。
