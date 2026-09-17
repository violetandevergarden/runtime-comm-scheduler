# JobPacer Phase 2 实现与验收结果

日期：2026-09-15

本阶段已完成两 rank CPU/Gloo replay。实现没有修改 `src/`，复用了已有的 `AdmissionScheduler`、`CommIntent`、`ScheduledWork`、`DirectLaunchExecutor` 和 completion probe。

## 实现内容

新增文件如下：

- `examples/jobpacer/workloads.py`：提供 `balanced`、`tail`、`delayed` 三个确定性 workload，也支持从 JSON manifest 加载。每个通信项记录 job 内 ordinal、op、bytes、通信前计算时长、通信后 consumer 计算时长和静态通信时长估计。
- `examples/jobpacer/plan_builder.py`：生成七字段 `TaskKey`，实现固定轮转 FIFO 和 longest-tail-first（LTF），并验证每个 job 的通信序列没有被重排或遗漏。
- `examples/jobpacer/replay_worker.py`：每个 rank 创建每个 job 独立的 ProcessGroup；scheduler 模式下所有 job 线程共享一个 rank-local `AdmissionScheduler`，通信调用通过 `CommIntent`/`ScheduledWork` 接入；bare 模式直接调用 Gloo collective。每个任务输出 producer、admission、submit、wait、complete telemetry。
- `examples/jobpacer/run_replay.py`：启动两 rank 子进程，设置 timeout，比较各 rank 的 plan digest、全局 launch sequence 和每个 ProcessGroup 的诱导序列，并可写 JSON trace。
- `tests/unit/test_jobpacer_plan.py`：覆盖 FIFO 顺序、LTF tail 顺序、job 内依赖和 digest/key 确定性。

命令行参数支持 `--mode bare|scheduler`、`--policy fifo|ltf`、`--workload`、`--backend gloo|nccl`、`--world-size` 和 `--max-outstanding`。例如：

```bash
python examples/jobpacer/run_replay.py \
  --mode scheduler --policy fifo --workload balanced \
  --backend gloo --max-outstanding 1 --output /tmp/jobpacer-fifo.json
```

`--fault missing_key` 和 `--fault metadata_mismatch` 用于有界故障路径验证；正常 replay 默认不注入故障。

## 验收配置

| 项目 | 值 |
| --- | --- |
| backend | Gloo |
| device | CPU |
| world size | 2 |
| PyTorch | 2.13.0+cu129 |
| workload | `balanced`、`tail`、`delayed` |
| communication | `all_reduce`，每项 4096 bytes |
| scheduler executor | `DirectLaunchExecutor` |
| 严格串行配置 | `max_outstanding=1` |
| 对照配置 | `max_outstanding=0` |
| rank 间 plan 校验 | 两 rank `all_gather_object` 比较 digest |

## 正常运行结果

所有正常运行均满足两 rank digest 相同、all-reduce 结果正确、scheduler launch sequence 与静态 Plan 相同、每个 job 的 ProcessGroup 子序列一致。

| 运行 | Plan digest | 预期 launch sequence | 结果 |
| --- | --- | --- | --- |
| scheduler / FIFO / `balanced` / max=1 | `995991f6170eb7e4932194acc468978f72aa0b039937c0dd1fe272fda41fafb2` | `job-0:0, job-1:0, job-0:1, job-1:1, job-0:2, job-1:2` | 通过 |
| scheduler / LTF / `tail` / max=1 | `c817cdf1e829da9b862dec0db93bfa445ee6efd33a2669fd901ea6f2e8de10df` | `job-0:0, job-0:1, job-0:2, job-1:0, job-1:1, job-1:2` | 通过 |
| bare / FIFO / `balanced` | `995991f6170eb7e4932194acc468978f72aa0b039937c0dd1fe272fda41fafb2` | scheduler sequence 不适用 | 通过 |
| scheduler / FIFO / `balanced` / max=0 | `995991f6170eb7e4932194acc468978f72aa0b039937c0dd1fe272fda41fafb2` | 与 FIFO Plan 相同 | 通过 |
| scheduler / FIFO / `delayed` / max=1 | `a1e3beeb0d15966ca314cc917e18877a8170f14b84b8b4b5857ef3dfbdae1e74` | `job-0:0, job-1:0, job-0:1, job-1:1` | 通过 |

`tail` workload 中 job-0 的通信之后仍有更长的估计 tail，因此 LTF 先完整推进 job-0；FIFO 仍按 job 配置顺序轮转。两种顺序都满足每个 job 内 ordinal 单调递增。

严格串行 trace 取 rank 0、按计划顺序排列后，连续任务的 `next.admit_ts - previous.complete_ts`（微秒）如下：

| 运行 | 相邻间隔（μs） |
| --- | --- |
| FIFO / `balanced` / max=1 | `18, 8411, 419, 15, 31` |
| LTF / `tail` / max=1 | `9667, 1139, 16, 883, 1290` |
| FIFO / `delayed` / max=1 | `51, 13925, 40` |
| FIFO / `balanced` / max=0 | `-1165, 9031, 781, -1132, 1060` |

max=1 的所有间隔均非负，说明下一任务的 admission 在前一任务被 completion probe 记录完成之后发生。delayed workload 中 job-0 的首个 producer sleep 为 80 ms；队首未 ready 时 job-1 的 intent 保持 pending，队首 ready 后仍按 Plan 发射。max=0 的负间隔说明它只限制 worker 的 host 发射顺序，不能作为物理完成后的串行保证。

每个 rank 的完整 JSON trace 由 `--output` 保存；本次验证使用的输出包括 `/tmp/jobpacer-fifo.json`、`/tmp/jobpacer-ltf.json`、`/tmp/jobpacer-bare.json`、`/tmp/jobpacer-unbounded.json` 和 `/tmp/jobpacer-delayed.json`。trace 中包含 workload manifest、随机种子、backend、world size、Plan version/window/digest/完整 key、每 job 任务记录和 scheduler timing。

## 错误与确定性验收

现有核心校验测试继续覆盖重复 key、缺失/意外 key、metadata mismatch、plan digest mismatch、ProcessGroup sequence divergence，以及 `finish_window` 缺失任务和 timeout。新增 Plan 单测覆盖静态策略的顺序、依赖和重复构造结果。

运行 `--fault missing_key`、两 rank、timeout=4 秒时，driver 在 4.16 秒内回收两个未能完成的 rank 并返回失败状态，没有留下挂起进程；这验证了 replay driver 的进程级有界退出。metadata mismatch 在 submit 前由核心 `validate_intent` 拒绝，错误不会进入 collective launch。

运行 `--fault metadata_mismatch`、timeout=8 秒时，rank 1 返回明确的 `ValidationError: op mismatch ... intent='all_gather' plan='all_reduce'`，另一 rank 因对端提前退出收到 Gloo connection error；driver 在 2.60 秒内回收两者。

## 测试结果

```text
python -m py_compile examples/jobpacer/*.py
pytest -q tests/unit tests/integration/test_gloo_order.py
......................................................s                  [100%]
54 passed, 1 skipped
```

跳过项是仓库原有 Gloo integration harness 在受限沙箱中无法创建本地 TCP rendezvous socket；上述 JobPacer 两 rank replay 使用本机 TCP rendezvous 在放行 socket 权限后实际完成。GPU/NCCL 路径已保留 CLI 和 `TorchProcessGroupExecutor` 接口，本阶段未将 GPU 作为 CPU/Gloo 验收的必要条件，也未进行 NCCL 结果宣称。

## 附加任务：离线通信 Profiling

日期：2026-09-16

已完成 `process/phase2.md` 中的补充工作。新增
`examples/jobpacer/profile_communication.py`，它从 workload 去重通信签名，按 manifest 顺序
创建 ProcessGroup，并逐签名执行无竞争 warmup 和正式测量。每轮在 group barrier 后计时真实
异步 all-reduce 直到物理完成，再在参与 rank 间取最大耗时。Gloo 使用单调 CPU 时钟；NCCL
路径在计时边界同步 CUDA device。rank 0 以临时文件加原子替换的方式写 profile，失败时不会
写出半成品。

新增 `examples/jobpacer/comm_profile.py`，提供稳定的 `CommSignature`、严格校验的
`ProfileRecord`/`CommunicationProfile`、确定性 JSON digest 和冻结 workload 的复制覆盖。
匹配键包括 op、bytes、dtype、group size、backend、device type 和 reduction；严格模式会在
启动 distributed collective 前拒绝缺失或重复签名，以及 backend、device、world size、
group membership 不匹配。非严格模式保留 manifest 估值并发出逐项 fallback 警告。

Replay 新增 `--comm-profile`、`--profile-strict/--no-profile-strict`。父进程和每个 rank 都在
构造 Plan 前应用同一 profile，各 rank 在执行前交换应用后 workload digest。输出 trace 现在
包含 profile 路径、内容 digest、schema version、strict 标志、环境摘要、workload digest，
以及每个 task 的最终 `estimated_comm_s` 和 `offline_profile`/`manifest` 来源。未传 profile
时保持原有手工估值路径。

CPU/Gloo 小轮次实测命令如下：

```bash
python examples/jobpacer/profile_communication.py \
  --workload balanced --backend gloo --world-size 2 \
  --warmup 1 --iterations 3 --timeout 20 \
  --output /tmp/jobpacer-profile.json

python examples/jobpacer/run_replay.py \
  --mode scheduler --policy ltf --workload balanced \
  --backend gloo --world-size 2 --max-outstanding 1 \
  --comm-profile /tmp/jobpacer-profile.json --timeout 20 \
  --output /tmp/jobpacer-profile-replay.json
```

该次 profile 对 4096-byte all-reduce 得到 p10/p50/p90 为
`0.476/0.581/0.894 ms`（3 个样本，仅用于功能验收）。应用 profile 后两 rank 的 workload
digest 均为 `4dbcf9b5dc660b63aa719957888c581b613bc14b516d58d09575f1959b4a1cec`，Plan
digest 均为 `d01d0c9e62e17c6662844b88d7bcf32d69a7b17f48bb79c112eab29a0c6d9f73`；LTF
顺序为 `job-1:0, job-0:0, job-1:1, job-0:1, job-1:2, job-0:2`。collective
结果、scheduler/group 顺序和严格串行 admission 检查均通过。

新增单元测试覆盖跨 job 签名去重、JSON round-trip/digest、不可变覆盖、缺失/重复/环境错误、
profile 改变 LTF 顺序；新增两 rank Gloo 集成测试，用 4096 和 8192 bytes 两种消息完成
profile 后再 replay，并校验样本、分位数、跨 rank workload digest、估值来源和 collective
正确性。完整 CPU 测试结果：

```text
pytest -q tests/unit tests/integration
...............................................................          [100%]
63 passed in 12.85s
```

当前机器未执行两 GPU NCCL 验收，因此不记录 NCCL 耗时或正确性结论。NCCL 测量与 replay
代码路径已经实现，需在目标双 GPU 实验机上重新生成 profile 后完成最终环境验收；本次 Gloo
绝对耗时不应跨机器复用。
