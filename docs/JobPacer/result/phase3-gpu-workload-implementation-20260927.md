# Phase 3 GPU workload implementation smoke 结果

日期：2026-09-27。本文只记录小规模语义与机制检查，不是性能对照结果。实施说明见[phase3-gpu-workload-implementation.md](../process/phase3-gpu-workload-implementation.md)，输入见[`gpu-linear/`](../../../benchmark/phase3/experiments/gpu-linear/)。

## 实施内容

新增独立 `jobpacer-gpu-linear` v1 parser/canonical digest、rank-local CUDA producer/independent/consumer 资源、compute profile loader/calibrator、NCCL communication profile 输入、GPU rank harness、linear runner、经验证图序驱动的 DAG bridge、FIFO/Static LTF/Lookahead hints 与父子 `run_phase3.py` 入口。TaskSpec 只带共同通信信息，tensor/stream/event/ProcessGroup/closure 留在本地。bridge 保持明确的通信提交与独立计算次序，并在 terminal event 后推进。策略 smoke 都使用同一固定输入与 profile；测试结果没有用于比较策略优劣。

真实设备测试中发现 scalar join 在当前 PyTorch 上必须向 `torch.sum` 提供 reduction dimensions。修正正式 consumer 和 warmup 后，矩阵结果与 SUM 结果均通过。backend launch 故障的第一次试验显示未匹配的 NCCL Work 会让 peer 卡住；新增 fail-stop 处理，在退出路径 abort rank-local NCCL groups 和 WORLD。修复后 launch fault 与其余故障注入都在有界时间内报告双 rank 错误或 replay deadline 错误。

## 环境与冻结摘要

使用仓库 `.venv` 的 PyTorch 2.13.0+cu126、CUDA 12.6、NCCL 2.29.3、Python 3.12.3。两张可见卡均为 NVIDIA GeForce RTX 4090；本轮未设置或覆盖 `CUDA_VISIBLE_DEVICES`：

- rank 0：`GPU-c024d768-866d-952a-36f6-899a9e5844e0`
- rank 1：`GPU-a376a831-0ace-b841-7b91-6b962b9c2398`

`L0-smoke.json` canonical manifest digest 为 `6f91c0f78435989c7a904858976f93f1e2a0148917e25a79a1ef0b6334ffb1c9`，raw manifest SHA-256 为 `81a7a440d61f8835cd1ce6e888f19488cf776e865541fedf5957230faf5d9a88`。

短机制校准分别使用 `warmup=1, iterations=3`。compute profile 记录每张设备 5 个签名，PyTorch precision 为 `highest`、TF32 关闭；NCCL profile 覆盖 1024、4096、16384 bytes 三种签名。profile digest：compute `551d69faec6af6bf07563c578c33de4c54097545d37bba7d7fffa06617787fc1`，communication `b6f7ebd44c7dab98221ec27a53b238a105209e31946267dde475c185d1804bd7`。这组少量样本仅用于检查 profile 签名、加载和策略入口。

正式 replay 源码摘要为 `1866f1e0e35d131d293043442ef439cf14012e80d8390f52c82468a6273895e6`，HEAD 为 `c1d5008478d9ed17cd3e5e55e73e1b177b53a99b`，运行时工作树为 dirty（任务开始前已有未提交工作）。运行 manifest、profile、每 rank 原始 JSON、故障输出、27 个源码/测试快照均保存在本机 Git 忽略目录 `benchmark/phase3/results/gpu-workload-implementation/L0-smoke/`；普通 Git 提交不会包含这些文件。

## 双卡结果

下表中的 makespan 是各自一次最小观测 smoke 输出的 `max_rank(local_end - release)`，单位微秒。每个 arm 只运行一次，不构成性能比较，也不说明策略稳定排序。

| 路径 | Policy | 验证 | 一次 makespan |
| --- | --- | --- | ---: |
| Linear | FIFO | 通过；task 全集、grant 投影、collective/independent/join 数值均正确 | 27092 |
| DAG bridge (pre-review mapping smoke) | FIFO | 原执行复用了 linear runner，不能证明 DAG 节点完成推进；仅保留为图映射 smoke | 26573 |
| Linear | Static LTF | 通过；profile-driven tail 签名和结果校验均正确 | 25067 |
| Linear | Lookahead | 通过；ready hint、结果校验均正确 | 24736 |

bridge digest 为 `582a061d973108defc3385765a5ebde29f8242a2fc742a69a63947f2310985ae`。审查发现该次运行虽验证图映射，却仍调用 linear runner 并事后补写节点事件；因此这次旧 bridge 数据只能标作图映射 smoke，其 makespan 不作为 DAG 执行或策略比较证据。修复后的真实节点推进结果会在下文“审查修复复验”单独记录。通用 DAG、其他图形及 performance equivalence 仍未验收。

六种 fault injection 均按预期以非零退出并生成 `validation.status=failed` 的诊断结果：

| fault | 退出时间 | 观察到的结果 |
| --- | ---: | --- |
| `compute_failure` | 7 s | rank 0 producer 失败，peer 收到 coordinator 的 fail-stop |
| `binding_failure` | 6 s | rank 0 在提交前绑定失败，peer 被唤醒 |
| `missing_task` | 12 s | 共用 replay deadline 到期，两个 rank 均报告 job runner timeout |
| `metadata_mismatch` | 5 s | coordinator 拒绝共同 task metadata 不一致 |
| `completion_probe_failure` | 6 s | completion probe 失败，peer 收到 coordinator fail-stop |
| `launch_failure` | 8 s | rank 0 backend launch 抛错；abort NCCL groups 后 peer 有界退出 |

这些是机制故障样例，不证明网络断连、任意 NCCL backend 错误或跨 group 故障都已覆盖。旧新 runtime 的双卡语义套件、Gloo 和回归测试结果如下：

| 命令 | 结果 |
| --- | --- |
| `RUN_JOBPACER_RUNTIME_NCCL=1 PYTHONPATH=src:. .venv/bin/python -m pytest -q tests/integration/test_runtime_replay_nccl.py` | 4 passed，25.75 s |
| `RUN_JOBPACER_RUNTIME_REPLAY=1 PYTHONPATH=src:. .venv/bin/python -m pytest -q tests/integration/test_runtime_replay.py` | 47 passed，137.46 s |
| `PYTHONPATH=src:. .venv/bin/python -m pytest -q tests/integration/test_jobpacer_phase1_gloo.py tests/integration/test_jobpacer_profile.py` | 2 passed，32.70 s；包含 Phase 1 H 与 Phase 2 profile→replay |
| `PYTHONPATH=src:. .venv/bin/python -m pytest -q` | 263 passed、48 skipped、3 failed，34.62 s |

全仓 3 个失败均为 `tests/unit/test_benchmark_paths.py` 中的历史迁移路径检查：工作区缺少 Git 忽略的 `benchmark/phase3/results/migration-map.json`、`benchmark/phase1.2/results/migration-map.json` 和迁移后的历史 manifest/raw，因此不能解析旧路径。未添加虚构 migration map 或结果占位文件。Phase 2 profile 集成在全仓首轮曾出现双 rank timeout；之后单项诊断复跑和本表集成运行均通过，没有找到可重复的超时根因，仍保留这一未归因观察。

## 审查修复复验

针对数值校验、静态顺序、LTF 评分和 bridge 推进问题完成修复后，在同一 L0 输入与冻结 profile 上各运行一次 linear FIFO 与 DAG bridge FIFO 双卡语义 smoke。命令如下：

```bash
env PYTHONPATH=src:. .venv/bin/python -m examples.jobpacer.scripts.run_phase3 \
  --gpu-linear benchmark/phase3/experiments/gpu-linear/L0-smoke.json \
  --policy fifo --backend nccl --world-size 2 --warmup-iterations 1 \
  --timeout 30 --setup-timeout 30 --observation-mode minimal \
  --output benchmark/phase3/results/gpu-workload-implementation/L0-smoke/review-fix-linear-fifo.json

env PYTHONPATH=src:. .venv/bin/python -m examples.jobpacer.scripts.run_phase3 \
  --gpu-bridge benchmark/phase3/experiments/gpu-linear/L0-smoke.json \
  --policy fifo --backend nccl --world-size 2 --warmup-iterations 1 \
  --timeout 30 --setup-timeout 30 --observation-mode minimal \
  --output benchmark/phase3/results/gpu-workload-implementation/L0-smoke/review-fix-bridge-fifo.json
```

两次运行的 manifest digest 均为 `6f91c0f78435989c7a904858976f93f1e2a0148917e25a79a1ef0b6334ffb1c9`，profile digests 与前述冻结文件一致，rank 使用同一对 RTX 4090 UUID，继承环境中的 `CUDA_VISIBLE_DEVICES`（未设置）。linear 与 bridge 的 `validation.status` 都为 `ok`，每 rank 的 3 个通信任务、grant/launch 投影和 tensor 校验通过；rank/task validation 开始时间均晚于本地 application end。

bridge 每 rank 实际推进并完成 10 个 DAG node：producer 与 join/terminal 通过 CUDA event 查询，independent 节点（有独立计算的片段）通过自己的 CUDA event 查询，comm 节点通过 runtime handle 的物理完成状态单独观察。父进程核验了每 rank node event 全集及 completed 状态。comm 的 host completion observation 可能晚于 terminal event 的 host query，因为两条观察由不同的完成探测路径驱动；结果分别保留，不将 terminal 时间复制为通信完成时间，也不把 observation timestamp 当作设备完成的精确时刻。这个 smoke 只验收当前线性映射 DAG 的真实推进，不覆盖 generic DAG 或故障/断连矩阵。

定向检查命令合计 `111 passed in 4.43 s`，包含 GPU parser/runner/bridge/profile/adapter 回归和 `tests/unit/runtime`；opt-in 双卡 NCCL runtime 语义套件 `4 passed in 23.92 s`；相关模块 `py_compile` 与 `git diff --check` 通过。静态 FIFO 新回归覆盖跨 group 序号不能破坏 job 片段依赖；Static/Dynamic LTF 回归对相同候选集比较同一版本化 tail 评分。此次未运行全仓测试、fault-injection batch 或 M3 配对实验。

复验 JSON、命令/硬件/source manifest 与 31 个源码/测试快照（18 个运行源码、13 个测试文件）保存在 Git 忽略目录 `benchmark/phase3/results/gpu-workload-implementation/L0-smoke/`；source digest 为 `9c32995d3789f50750aa234e15a18313fa64217a2b47a50752ba72ae100e7219`，清单文件为 `review-fix-manifest.json`。两次 smoke 是不同路径的单次语义检查，makespan 不作配对或性能解释。

## 2026-09-28 收尾与 profile 签名补充复验

linear worker 现在在全部本地 terminal 完成后先调用 `finish_epoch()` 关闭输入并排空协议，再开始 GPU→CPU 数值校验。单测用受控时钟模拟校验期间越过原 replay deadline，并断言 finish 收到校验前剩余的一秒预算；双卡结果也验证两 rank 的 `communication_drain_end_us` 均早于 `validation_start_us`。校验仍单独记录时长，不进入 application makespan 或 protocol drain。

compute profile 改为 schema v2。fill 和 sum_join 的签名带有 shape、dtype、layout；旧 v1 profile 被 loader 拒绝。`L0-smoke.json` 和 `L1-misaligned.json` 都改为引用 `gpu-compute-v2.json`，原先 L0 的 v1 `gpu-compute.json` 保留作历史文件。L0 用 `warmup=1, iterations=3` 在两张 RTX 4090 上重新校准，每卡 5 个签名、总计 10 条记录，UUID 为 `c024d768-866d-952a-36f6-899a9e5844e0` 与 `a376a831-0ace-b841-7b91-6b962b9c2398`；运行软件为 PyTorch 2.13.0+cu126、CUDA 12.6、NCCL 2.29.3。新 profile 的 runtime digest 为 `1dfde0fb69016ab0a3370d8d827df58b657d2245d610f4b564b211c919baec26`，文件 SHA-256 为 `c48228fabf03c191d9d0d2082848c8ec6972039d756645f18543d09ef8b8062a`。

首次策略 replay 发现校准入口仍硬编码输出 schema v1，rank 在应用开始前拒绝加载（两 rank 错误为 `unsupported GPU compute profile schema/version`，约 7.1 s）；入口已改为共享 profile schema 常量，v2 文件重新生成后完成以下两个双卡 smoke。该失败尝试的输出路径随后被成功复跑复用，未保留原始失败 JSON，错误原因在此记录：

```bash
env PYTHONPATH=src:. .venv/bin/python -m examples.jobpacer.scripts.run_phase3 \
  --gpu-linear benchmark/phase3/experiments/gpu-linear/L0-smoke.json \
  --policy static_ltf --backend nccl --world-size 2 --warmup-iterations 1 \
  --timeout 30 --setup-timeout 30 --observation-mode minimal \
  --output benchmark/phase3/results/gpu-workload-implementation/L0-smoke/review-20260928-static-ltf-v2.json

env PYTHONPATH=src:. .venv/bin/python -m examples.jobpacer.scripts.run_phase3 \
  --gpu-linear benchmark/phase3/experiments/gpu-linear/L0-smoke.json \
  --policy lookahead --backend nccl --world-size 2 --warmup-iterations 1 \
  --timeout 30 --setup-timeout 30 --observation-mode minimal \
  --output benchmark/phase3/results/gpu-workload-implementation/L0-smoke/review-20260928-lookahead-v2.json
```

两次输出均为 `validation.status=ok`，每次约 6.1 s；每 rank 的 3 个通信任务、grant/launch 投影及 collective、independent、join 校验通过。两个结果都记录了 v2 profile digest，且协议 drain 时间早于校验开始。L0 新 manifest digest 为 `447d45022c43e39089906556f6e7dc5efe5ad3bb186481bf1c3ca0b561b0cf86`；两次运行 source digest 为 `b6b9d51b91017ce5137b017bf42c620c9ccc2322158c4cdf816963a8f2858892`。每个策略仅一次运行，只证明 v2 profile 可驱动当前 L0 语义 smoke，不构成收益或排序证据。L1 profile 和 replay、M3 配对实验、重复 epoch 与断连验收仍未完成。

本轮验证命令 `env PYTHONPATH=src:. .venv/bin/python -m pytest -q tests/unit/runtime tests/unit/test_jobpacer_gpu_workload.py tests/unit/test_jobpacer_gpu_compute_profile.py tests/unit/test_jobpacer_gpu_linear_runner.py` 为 **83 passed in 3.66 s**；涉及模块 `py_compile` 与 `git diff --check` 均通过。v2 profile 和两次 replay JSON 位于 Git 忽略目录 `benchmark/phase3/results/gpu-workload-implementation/L0-smoke/`；复验命令、运行 manifest、SHA256SUMS 和 18 个实现文件快照另存于 `benchmark/phase3/results/gpu-workload-implementation/review-20260928-profile-v2/`。

## 未验收范围

M3 正式 paired workload、足量 calibration、seed blocks 和噪声对照未执行；本轮任何单次 makespan 都不能作为性能收益或策略排序结论。`L1-misaligned.json` 已加入输入集，但没有 profile 或 replay。长时间重复 epoch、断连、profile 漂移、跨多 group 的错误恢复和性能误差统计仍须单独验收。
