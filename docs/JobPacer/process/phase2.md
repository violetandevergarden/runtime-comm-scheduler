# Phase 2 补充实施计划：离线通信 Profiling

## 1. 目标

Phase 2 当前已经能够从 workload 构造 FIFO/LTF 静态 Plan，但
`CollectiveComm.estimated_comm_s` 仍由 workload 手工填写，内置 workload 中统一为
`0.001`。本补充工作的目标是：在正式 replay 前，用与实验一致的通信环境独立测量
collective 的无竞争执行时间，将结果保存为可审计、可复用的 profile，并在加载 workload
后覆盖 `estimated_comm_s`，供 LTF 的 remaining-tail 计算使用。

本阶段只 profile **通信时间**。`producer_compute_s` 和 `consumer_compute_s` 仍来自手工
workload；真实计算 profiling 留给后续阶段。profiling 也不改变 scheduler、Plan 或
`CommIntent` 的接口。

## 2. 时间口径

`estimated_comm_s` 定义为：

> 在无其他 job 通信竞争时，一次 collective 从所有参与 rank 到达测量边界，到所有参与
> rank 的该 collective 物理完成之间的时长估计。

具体采用多轮样本的中位数（p50）作为调度估计值，同时在 profile 中保留 p10、p90、均值、
标准差和样本数，便于判断稳定性。每轮先在该 ProcessGroup 上同步，再执行真实 collective
并等待完成；每个 rank 测得本地耗时后，通过 group 内归约取最大值作为该轮耗时。取最大值
是因为 collective 的可用完成边界由最慢参与 rank 决定。

- Gloo/CPU：用 `time.perf_counter_ns()` 包住 `dist.all_reduce(..., async_op=True)` 和
  `work.wait()`。
- NCCL/CUDA：测量前同步 device，在 group barrier 后记录单调时钟，发起异步 collective，
  `work.wait()` 后再次同步 device，再记录结束时间。不能只测 Python 调用返回时间，因为
  NCCL 工作仍可能在 CUDA stream 上执行。
- warmup 不计入样本，默认 10 轮；正式测量默认 50 轮。
- 第一版不扣除 barrier、Python 调用或同步开销。它们不放进计时区间；残余固定开销通过
  warmup 和中位数减弱。若以后需要研究小消息的亚毫秒差异，再单独增加空操作校准。

这里要测的是策略使用的“独占服务时间”，因此不能直接使用 multi-job replay 中
`submit_ts -> complete_ts` 的时长：该值混合了排队、链路竞争和 completion polling 延迟，
会让同一个策略的输入依赖其自身运行结果。

## 3. Profile 的匹配键和文件格式

同一通信大小在不同环境中的耗时可能不同，profile 记录分为环境元数据和通信记录两层。
通信记录的匹配键至少包含：

- `op`：当前为 `all_reduce`；
- `num_bytes`；
- `dtype`：当前 replay 固定为 `float32`，不能隐式省略；
- `group_size`；
- `backend`：`gloo` 或 `nccl`；
- `device_type`：`cpu` 或 `cuda`；
- reduction 类型：当前为 `sum`。

环境元数据另外记录 world size、实际 group ranks、主机名、GPU 型号、PyTorch/CUDA/NCCL
版本、warmup/measurement 次数和生成时间。机器拓扑不适合压进字符串 key，但应用 profile
时必须校验环境元数据，并把 mismatch 报出来。Phase 2 中相同签名的多个 job 共用一条
profile 记录；`job_id` 和 ordinal 不属于通信性能签名。

建议输出 JSON：

```json
{
  "schema_version": 1,
  "environment": {
    "backend": "nccl",
    "world_size": 2,
    "torch_version": "...",
    "cuda_version": "...",
    "device_names": ["..."]
  },
  "settings": {"warmup": 10, "iterations": 50},
  "records": [
    {
      "op": "all_reduce",
      "num_bytes": 16777216,
      "dtype": "float32",
      "group_size": 2,
      "backend": "nccl",
      "device_type": "cuda",
      "reduction": "sum",
      "p50_s": 0.00251,
      "p10_s": 0.00245,
      "p90_s": 0.00263,
      "mean_s": 0.00253,
      "stdev_s": 0.00007,
      "samples": 50
    }
  ]
}
```

## 4. 代码改动

### 4.1 新增 `examples/jobpacer/profile_communication.py`

增加独立 CLI，复用 `load_workload()` 和 `ranks_for_job()`：

1. 读取 workload，抽取并去重所有通信签名；
2. 与 replay 相同地初始化 distributed backend，并按 manifest 顺序创建 group；
3. 对每个实际出现的 `(group ranks, communication signature)` 单独运行 warmup 和测量；
4. 每轮重新初始化输入 tensor，避免 all-reduce 连续累加导致溢出；
5. 在参与 rank 内汇总每轮最大耗时，计算统计值；
6. rank 0 写出 profile JSON，其他 rank 只参与测量；
7. 任一 rank 失败时按明确 timeout 结束，不写半成品 profile。

CLI 至少提供：

```text
--workload PATH_OR_NAME
--backend gloo|nccl
--world-size N
--warmup 10
--iterations 50
--timeout 30
--output PATH
```

profiling 必须一次只运行一个通信签名，不启动 job 线程，不经过 scheduler，也不并发多个
ProcessGroup。这样测到的是基础通信成本，而不是 Phase 2 调度策略制造出的竞争成本。

### 4.2 新增 `examples/jobpacer/comm_profile.py`

集中放置以下纯数据逻辑，避免 profiling CLI 与 replay driver 重复解析代码：

- `CommSignature`：构造稳定匹配键；
- `ProfileRecord` / `CommunicationProfile`：校验和 JSON 序列化；
- `load_profile(path)`；
- `apply_profile(workload, profile, environment, *, strict=True) -> Workload`。

`apply_profile` 返回新的冻结 dataclass 对象，不原地修改 workload。它逐项复制
`CollectiveComm`，仅用匹配记录的 `p50_s` 覆盖 `estimated_comm_s`。默认 strict 模式下，
任何 task 缺少记录、签名重复、单位非法或 backend/device/group size 不一致都立即报错；
不允许静默退回 workload 中的 `0.001`。可选的非 strict 模式只用于调试，并在输出中明确
列出 fallback 项。

### 4.3 修改 replay 入口

在 `examples/jobpacer/run_replay.py` 增加：

```text
--comm-profile PATH
--profile-strict / --no-profile-strict
```

父进程先加载 workload 和 profile、完成覆盖，再用同一份解析结果构造 Plan；rank worker
也加载同一 profile 并在构造 Plan 前应用。最终 trace 增加：

- profile 文件路径和内容 digest；
- `schema_version`；
- 是否 strict；
- 每个 task 最终使用的 `estimated_comm_s`；
- profile 的环境摘要。

各 rank 在执行前交换“应用 profile 后 workload”的 digest；digest 不一致直接失败。这样可
保证 LTF 在所有 rank 上使用完全相同的估计值并生成相同 Plan。

为避免父进程和 worker 的参数漂移，`run_replay.py::_run_rank()` 必须把 profile 参数显式传给
`replay_worker.py`。profile 未指定时保留当前手工估值路径，以便复现既有 Phase 2 结果；但
正式性能实验必须指定 profile，trace 中要标记估值来源为 `offline_profile` 或 `manifest`。

### 4.4 workload 模型的小改动

当前 `CollectiveComm` 已有 `estimated_comm_s`，无需增加第二套字段。建议只做以下调整：

- 将注释从“预估计算时间”改为“预估通信时间（秒）”；
- 删除 `TODO` 默认值注释，默认值仍可暂时保留以兼容内置 workload；
- 在 trace 中显式输出估值来源，避免默认值被误当成实测值。

## 5. 执行流程

先生成 profile：

```bash
python examples/jobpacer/profile_communication.py \
  --workload path/to/workload.json \
  --backend nccl --world-size 2 \
  --warmup 10 --iterations 50 \
  --output artifacts/jobpacer/comm-profile-nccl.json
```

再将同一份 profile 用于 FIFO 和 LTF：

```bash
python examples/jobpacer/run_replay.py \
  --mode scheduler --policy ltf \
  --workload path/to/workload.json \
  --backend nccl --world-size 2 --max-outstanding 1 \
  --comm-profile artifacts/jobpacer/comm-profile-nccl.json \
  --output artifacts/jobpacer/ltf.json
```

FIFO 本身不依赖 `estimated_comm_s`，但仍加载同一 profile，以保证 FIFO/LTF 的 workload、
元数据和实验记录一致。若 profile 环境与本次 replay 不符，应重新 profile，而不是人工缩放
旧结果。

## 6. 测试与验收

新增单元测试：

1. 相同签名跨 job 去重，ordinal/job_id 不进入匹配键；
2. profile JSON round-trip 和 digest 确定性；
3. `apply_profile` 正确覆盖每个 `estimated_comm_s`，不修改其他字段；
4. 缺失签名、重复签名、错误 backend/device/group size 在 strict 模式下失败；
5. 应用 profile 后，LTF 顺序按实测值变化且仍保持每个 job 内顺序；
6. 两 rank 应用结果和 Plan digest 一致。

新增两 rank Gloo 小型集成测试：使用至少两个消息大小，少量 warmup/iterations 生成临时
profile，再用该文件运行 replay；断言所有记录为正数、样本数正确、workload digest 一致、
collective 结果正确。NCCL 测试放到 GPU 测试目录，在有两张 GPU 时验证同一路径，不把具体
毫秒值写成硬断言，只检查数值有限、为正且 p10 <= p50 <= p90。

完成标准：

- 同一 workload/profile 重复加载得到相同签名、profile digest 和 Plan digest；
- LTF 不再依赖统一手填的 `0.001`，trace 能追溯每个估值对应的 profile；
- profiling 与正式 replay 使用相同 backend、dtype、group membership 和设备类型；
- profile 缺失或环境不匹配时，正式实验能在启动 collective 前明确失败；
- FIFO/LTF 的对比使用完全相同的 workload 和同一份离线 profile。

## 7. 实施顺序

1. 实现 profile 数据模型、签名、严格匹配和单元测试；
2. 实现 Gloo profiling CLI，并生成一个小型示例 profile；
3. 将 profile 注入 replay/Plan 构造，补充 workload 与 profile digest；
4. 增加 Gloo 端到端测试；
5. 在目标 GPU 环境实现并验证 NCCL 测量路径；
6. 用最终实验机器重新生成 profile，再运行 Phase 2 FIFO/LTF 正式对比。

不要把示例机器上生成的绝对耗时当作跨机器通用模型。profile 是实验环境的一部分，应和
workload、代码版本及 replay 输出一起归档。
