# GPU 七臂 v2 输入与执行入口

`revision-0/pilot` 包含 9101–9103 的 18 个展开输入；`revision-0/formal` 包含 8101–8105 的 30 个候选输入。每个目录的 `suite-inputs.json` 记录文件、输入、执行工作量和默认顺序 hash，以及逐节点 actual/nominal repeats。正式候选未通过 pilot 不可晋级。

| 场景 | 参数与机制 |
| --- | --- |
| L0 | 两 job、三段 producer→comm→tail→consumer 链，256² matmul，均衡对照 |
| L1 | 同链，job-0 首 producer 的 execution repeats=160（扰动前），nominal=8，制造未预知静态 HOL |
| D0 | 两阶段 fork/join，独立 matmul repeats=32，通信与计算分支汇合 |
| D1 | 独立 occupancy job 的 64 MiB collective；两 job 前沿通信 256 KiB，后继 repeats=8/96，提供真实竞争机会 |
| D2 | 三阶段 fork/join，producer 偏斜=96，独立计算=24，后继=8/32，追踪通信解锁 |
| D3 | 两阶段跨 group fork/join，组内规范序列及每 job 两个 sink |

这些是初始工作量参数，不承诺绝对微秒时长或 D1 触发率。扰动对实际 matmul repeats 乘 0.75/1/1.25；名义估计读取冻结的 nominal 签名，不读取运行扰动。完整差异在 manifest 中公布。默认通信顺序采用 `topological-layer-job-round-robin-v1`；层级不是运行屏障。

从仓库根目录、已有 `.venv` 运行：

```bash
source .venv/bin/activate
# profile 覆盖名义及实际签名。正式 seeds 不用于挑选参数。
PYTHONPATH=src:. python -m examples.jobpacer.scripts.run_gpu_compute_profile \
  --suite-input-dir benchmark/phase3/experiments/gpu-seven-arm-v2/revision-0/formal \
  --suite-input-dir benchmark/phase3/experiments/gpu-seven-arm-v2/revision-0/pilot \
  --warmup 5 --iterations 30 --output /tmp/v2-compute-profile.json
# D1 包含本 revision 的两个通信签名。
PYTHONPATH=src:. python -m examples.jobpacer.scripts.run_comm_profile \
  --dag benchmark/phase3/experiments/gpu-seven-arm-v2/revision-0/pilot/inputs/D1-asymmetric-frontiers/workload-9101.json \
  --backend nccl --world-size 2 --warmup 5 --iterations 30 --output /tmp/v2-comm-profile.json
```

`prepare_gpu_seven_arm freeze` 为每个 suite 应用两个 profile、生成统一 LTF 顺序；pilot 需指定 `--calibration-input-dir .../formal`。`run_gpu_seven_arm plan/run/analyze` 管理完整块、attempt、源码冻结与独立校验。新 batch 不接受旧 raw 臂，也不继承旧源码资格。

`qualify_gpu_seven_arm` 要求 `--mechanism-evidence`，先用 `run_bare_nccl_mechanism --output <新目录>` 生成，再验收六场景正常与缺请求/binding/launch/probe 故障。

`run_gpu_measurement_checks plan --suite <pilot/suite-manifest.json> --output <新目录>` 生成 A/A 与 minimal/full 各路径至少五对的独立诊断；`run` 显式启动，`analyze` 输出 measurement-audit。标签不进入 replay 参数。正式 `run` 还要求 software/backend/mechanism/measurement/rehearsal/recovery/budget 的 source/profile 绑定证据；仅完整计数不足以解锁。

原始结果存放在被 Git 忽略的 `benchmark/phase3/results/gpu-seven-arm/<batch-id>/`。已有冻结目录不可覆盖；参数修改须另起 revision，保留失败和未触发样例。
