# JobPacer Phase 2 示例

本目录提供多 job 线性通信 workload 的真实 PyTorch collective 重放工具。它可以直接并发
发射通信作为 baseline，也可以通过 `AdmissionScheduler` 按静态 FIFO、corrected LTF，或
`--selection ready_first` 的跨 rank globally-ready-first 控制发射顺序。LTF 保留 CLI 名称，
但 score 是零准入延迟假设下的估计剩余关键路径，不是通信完成后的 tail。正式比较前，
可先在相同通信环境中生成离线 profile，
用实测的无竞争通信时间替换 workload 中的手工估值。

以下命令均从仓库根目录执行。

## 快速开始

先运行无需 profile 的两 rank CPU/Gloo 示例：

```bash
python examples/jobpacer/run_replay.py \
  --mode scheduler \
  --policy fifo \
  --workload balanced \
  --backend gloo \
  --world-size 2 \
  --max-outstanding 1 \
  --output /tmp/jobpacer-fifo.json
```

`run_replay.py` 会启动每个 rank 的子进程。成功时终端输出验收摘要，`--output` 保存完整 JSON
trace。

正式运行 LTF 时，先测量通信 profile：

```bash
python examples/jobpacer/profile_communication.py \
  --workload balanced \
  --backend gloo \
  --world-size 2 \
  --warmup 10 \
  --iterations 50 \
  --output /tmp/jobpacer-gloo-profile.json
```

再将同一 profile 用于 FIFO 和 LTF：

```bash
python examples/jobpacer/run_replay.py \
  --mode scheduler --policy fifo --workload balanced \
  --backend gloo --world-size 2 --max-outstanding 1 \
  --comm-profile /tmp/jobpacer-gloo-profile.json \
  --output /tmp/jobpacer-profiled-fifo.json

python examples/jobpacer/run_replay.py \
  --mode scheduler --policy ltf --workload balanced \
  --backend gloo --world-size 2 --max-outstanding 1 \
  --comm-profile /tmp/jobpacer-gloo-profile.json \
  --output /tmp/jobpacer-profiled-ltf.json
```

Profile 与 replay 必须使用相同的 backend、device type、world size 和 ProcessGroup 成员。
默认 strict 模式还要求 profile 覆盖 workload 的全部通信签名；校验在启动 collective 前完成。

## Workload

内置 workload 有：

| 名称 | 用途 |
| --- | --- |
| `balanced` | 两个 job 各含三个通信，用于一般顺序和正确性检查 |
| `tail` | 两个 job 的剩余 tail 不同，使 FIFO 和 LTF 产生不同顺序 |
| `delayed` | FIFO 队首延迟 ready，用于检查队首等待和有界退出 |

`--workload` 也可以指向 JSON 文件：

```json
{
  "name": "two-sizes",
  "seed": 0,
  "jobs": [
    {
      "job_id": "job-a",
      "ranks": [0, 1],
      "communications": [
        {
          "id": 0,
          "num_bytes": 4096,
          "op": "all_reduce",
          "producer_compute_s": 0.005,
          "consumer_compute_s": 0.002,
          "estimated_comm_s": 0.001
        },
        {
          "id": 1,
          "num_bytes": 16777216,
          "op": "all_reduce",
          "producer_compute_s": 0.001,
          "consumer_compute_s": 0.001,
          "estimated_comm_s": 0.001
        }
      ]
    }
  ]
}
```

`id` 必须从 0 连续递增。当前只支持 float32、sum all-reduce，`num_bytes` 必须是 4 的正整数
倍。省略 `ranks` 时，该 job 使用所有 rank。`producer_compute_s` 表示提交通信前的计算窗口，
`consumer_compute_s` 表示提交后到调用 `wait()` 前的重叠窗口，单位都是秒。
`estimated_comm_s` 仅参与 LTF Plan 计算；加载 profile 后会由对应记录的 p50 覆盖。

## Replay 选项

主要参数如下：

| 参数 | 含义 |
| --- | --- |
| `--mode bare\|scheduler` | 直接调用 collective，或通过 scheduler 发射 |
| `--policy fifo\|ltf` | 静态 Plan 构造策略 |
| `--selection runtime_arrival\|ready_first` | 静态 Plan 发射，或跨 rank 协调全局 ready 集合 |
| `--backend gloo\|nccl` | CPU/Gloo 或 GPU/NCCL |
| `--world-size N` | rank 数，至少为 2 |
| `--max-outstanding N` | scheduler 最大在途任务数；`1` 表示等待物理完成后再准入下一项，`0` 不限制 |
| `--comm-profile PATH` | 可选的离线通信 profile |
| `--profile-strict` | 缺少签名时失败，默认启用 |
| `--no-profile-strict` | 调试模式；缺少签名时警告并使用 manifest 估值 |
| `--timeout SECONDS` | rank 进程、线程和窗口完成的超时 |
| `--output PATH` | 完整 replay trace 的写入路径 |

`bare` 用于观察多个 job 不经调度器协调时的行为。`scheduler` 模式下每个 rank 的 job 线程
共享一个 `AdmissionScheduler`，但每个 job 使用独立 ProcessGroup。FIFO 按 workload 中的 job
顺序轮转；LTF 用 `max(estimated_comm, consumer_compute)` 估计当前段，后续段再加 producer
compute，并保持每个 job 内顺序。ready-first 按全局 ready round 和稳定 TaskKey 选择，同时
保持 ProcessGroup 内顺序；serial 模式在全局 completion barrier 后进入下一轮。控制面耗时
单独记录，但 Gloo 控制流量可能干扰共享的 Gloo 数据面。

`--fault missing_key` 和 `--fault metadata_mismatch` 是故障路径验收参数，不用于性能实验。

## Profile 的测量和输出

`profile_communication.py` 对 workload 中的通信签名去重，并一次只测量一个签名。每轮流程为：

1. 重新初始化输入 tensor；
2. 在对应 ProcessGroup 上执行 barrier；
3. 发起异步 all-reduce 并等待物理完成；
4. 取参与 rank 本地耗时的最大值。

Warmup 不进入样本。Profile 保存 `profile_service_time_s`（兼容别名 `p50_s`）、API 调用
时长、return-to-sync 时长、p10、p50、p90、均值、标准差和样本数，并记录 backend、
world size、group ranks、PyTorch/CUDA/NCCL 版本、设备名称和生成时间。Replay 使用 p50 作为
`estimated_comm_s`。

Profile 是实验环境的一部分，不应把一台机器生成的绝对耗时直接用于另一台机器。相同通信
签名如果出现在不同 rank 集合上，当前 profiler 会明确拒绝，因为记录无法安全地共用。

## GPU/NCCL

两张 GPU 的基本用法为：

```bash
python examples/jobpacer/profile_communication.py \
  --workload balanced --backend nccl --world-size 2 \
  --warmup 10 --iterations 50 \
  --output /tmp/jobpacer-nccl-profile.json

python examples/jobpacer/run_replay.py \
  --mode scheduler --policy ltf --workload balanced \
  --backend nccl --world-size 2 --max-outstanding 1 \
  --comm-profile /tmp/jobpacer-nccl-profile.json \
  --output /tmp/jobpacer-nccl-ltf.json
```

父进程会按 rank 设置 `CUDA_VISIBLE_DEVICES`，每个 rank 在其可见设备的 `cuda:0` 上运行。
Profiler 在计时区间前后同步 CUDA device，避免只测到 Python API 返回时间。

## Trace 与验收字段

Replay trace 的顶层包含运行配置、validation 摘要和每个 rank 的完整记录。重点字段包括：

- `validation`：collective 正确性、Plan digest、scheduler/group 顺序、rank-local 串行与边界检查；
- `ranks[].workload_digest`：应用 profile 后的 workload digest，各 rank 必须一致；
- `ranks[].communication_profile`：profile 路径、digest、schema、环境和估值来源；
- `ranks[].plan`：Plan version、digest 和完整 key 顺序；
- `ranks[].jobs[].tasks[]`：schema v2 API/collective、application/backend wait、completion-observed 和 deferred validation 时间戳及派生耗时；
- `ranks[].launch_sequence`、`group_sequence`：scheduler 的实际发射顺序。

同一进程内的时间戳可以相减；不同 rank 的时间戳没有做时钟同步，不能直接比较。

逐 task tensor correctness scan 在所有 job 结束且 completion observer/scheduler 排空后执行，
不进入 application makespan。`application_makespan_us`、`communication_drain_makespan_us`、
`validation_total_us` 和 `harness_total_us` 分别报告 workload、通信排空与 harness 边界。
Visualizer 只接受完整 batch manifest 列出的 schema-v2 trace，并拒绝缺失/逆序字段。

## 文件说明

| 文件 | 职责 |
| --- | --- |
| `workloads.py` | workload 数据模型、内置 workload 和 JSON 加载 |
| `plan_builder.py` | FIFO/LTF 静态 Plan 构造和 job 内顺序校验 |
| `comm_profile.py` | profile 数据模型、digest、严格匹配和 workload 覆盖 |
| `profile_communication.py` | 多进程 Gloo/NCCL 离线测量入口 |
| `run_replay.py` | 启动各 rank、回收超时进程、汇总和验证输出 |
| `replay_worker.py` | 单 rank ProcessGroup、job 线程、scheduler 和 collective 执行 |

## 测试

```bash
pytest -q tests/unit/test_jobpacer_plan.py \
  tests/unit/test_jobpacer_comm_profile.py

pytest -q tests/integration/test_jobpacer_profile.py
```

集成测试需要允许本机 TCP loopback socket；没有该权限时测试会跳过。
