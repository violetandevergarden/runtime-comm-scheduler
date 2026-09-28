# Runtime Communication Scheduler

研究真实 PyTorch collective 的运行时通信调度。当前主线是 **JobPacer Phase 3**：中心化 coordinator 根据各 rank 的在线就绪与完成状态决定通信准入，比较静态顺序与动态策略；线性 workload 和手写 DAG 共用新的 runtime。

本项目与 `SimAI/simai-flow-scheduler` 互补：前者用于 flow-level replay 与策略研究，本仓库验证真实执行栈中可观察、可控制和可安全重排的通信边界。

## 当前范围与架构

```text
线性 workload / DAG runner
  → RankRuntime.submit(TaskSpec, LocalBinding, TaskHint) → handle
  → OFFER → 中心化 coordinator：合法候选 → policy → GRANT
  → 各 rank 单 launch worker → PyTorch Gloo / NCCL all-reduce
  → SUBMITTED → 独立物理完成探测 → COMPLETED
```

`TaskSpec` 表达共同任务身份与 group 顺序，`TaskHint` 表达估计，本地 tensor、ProcessGroup、CUDA event 和 closure 保存在 `LocalBinding`。控制通道使用独立 TCP，不使用被调度的 collective 做 rendezvous。

当前保持单个全局通信容量 `max_inflight=1`，从 grant 提交到全部成员物理完成期间占用。每 rank 的实际 launch 必须保持共同 grant 的本地投影顺序。`submit()` 不等待 grant；host 完成等待与 consumer stream 依赖分别定义，不能把 NCCL `Work.wait()` 的 CPU 返回当作设备完成。

Phase 3.2 的 DAG runner 位于通信 runtime 上层，负责计算节点和完成依赖推进。GPU compute 使用固定次数 CUDA matmul 和完成事件；它是实验 workload，并非真实训练框架集成。多在途通信、多资源调度、独立 coordinator 部署及 Megatron 接入不属于当前已验收能力。

## 远程 GPU 实验状态

2026-09-27 首轮实验使用 2× RTX 4090、Python 3.12、PyTorch 2.13.0+cu126、CUDA runtime 12.6、NCCL 2.29.3。版本来自该轮环境记录，不代表其他 checkout 的配置。

- G2 固定通信链中，新 runtime 的整体路径明显慢于旧路径；每项额外闭环约 4.7–5.4 ms，不能归因为纯控制面成本。
- G3 L1 复现了静态队首等待和动态 FIFO 提前服务其他候选；同 runtime 内有收益，但未证明相对旧路径的净收益。
- G4 三臂有 54 个有效 replay，另保留失败重试记录；单次 Kineto 诊断未证明新 runtime 有有效、可重复的 GPU kernel overlap。
- G0 尚缺网络断连与同进程重复 epoch；G1 噪声批次未实现计划中的交错 A/B，不能标为完整验收。

完整历史记录见 [GPU 实验报告](docs/JobPacer/result/phase3-gpu-20260927.md)与[实施及实验合同](docs/JobPacer/process/phase3-gpu-experiments-and-fixes.md)。计划、批次执行、正确性验收与性能收益应分别阅读。

### 2026-09-27 工作树复核

首轮报告的全仓测试记录早于当前复核，不能作为当前工作树全通过的依据。本次使用 `.venv/bin/python` 执行 `PYTHONPATH=src python -m pytest -q`：**238 passed、48 skipped、5 failed，34.29 秒**。三项为历史 benchmark 路径/fixture 缺失；另两项为 Phase 1 Gloo bare replay 失败和 Phase 2 profile/replay 超时。

代码审查确认，共用 [replay_worker.py](examples/jobpacer/runtime/replay_worker.py) 新增 S lane 后，`application_wait_start_ts` 只在 S 分支赋值，而 H 分支也读取它。Phase 1 默认 H 路径已实际触发 `UnboundLocalError`；Phase 2 H 路径也存在同一未赋值读取。应先恢复旧路径回归，再使用当前源码继续 H lane 对照；已有批次应使用各自归档源码复现。

G5 的 bridge 批次成功执行，但尚不能认定为等价迁移：所选 [DAG 输入](benchmark/phase3/experiments/bridge/G0-linear-bridge.json) 是 `matmul → comm` 串行链，线性 S lane 则在 producer fill 完成后启动独立 matmul 与通信，并在 consumer 汇合。两者依赖、consumer 工作和应用终点不同；相同 matmul 尺寸和通信集合不足以验收 bridge。diamond 的单独语义证据应与此区分。

本次另运行 `PYTHONPATH=src RUN_JOBPACER_RUNTIME_NCCL=1 .venv/bin/python -m pytest -q tests/integration/test_runtime_replay_nccl.py`：**4 passed，24.87 秒**；单独重跑 `tests/integration/test_jobpacer_profile.py` 仍超时失败（24.67 秒）。pytest 产物位于本机 `/tmp/pytest-of-liuxunpeng/pytest-13`（全仓）、`pytest-14`（NCCL）及 `pytest-15`（profile 重试），属于临时文件。`git diff --check` 通过。

上述是此前复核时的状态。2026-09-28 已继续实施 schema-v2 GPU 统一 DAG、old/new 通信 adapter 与 raw-ordered 受控参考；D0 双卡语义 smoke 和本轮 Gloo/NCCL 回归见[实施过程记录](docs/JobPacer/process/phase3-gpu-unified-dag-correction.md)及[结果记录](docs/JobPacer/result/phase3-gpu-unified-dag-correction-20260928.md)。原始 bare、六场景 pilot、重复 epoch、断连验收和正式 1,050 次性能矩阵仍未完成，不能据单次 smoke 判断收益。

## 目录与阅读顺序

| 路径 | 职责 |
| --- | --- |
| `src/runtime_comm_scheduler/runtime/` | 模型、coordinator、policy、控制通道、本地执行及完成探测 |
| `src/runtime_comm_scheduler/dag/` | 图模型、校验、计算与通信节点推进 |
| `examples/jobpacer/runtime/` | workload 映射、rank harness、GPU compute |
| `examples/jobpacer/scripts/` | 单次 replay、通信 profile、实验批次入口 |
| `examples/jobpacer/analysis/` | 结果校验、汇总与可视化 |
| `benchmark/phase3/experiments/` | 语义分类的实验输入 |
| `benchmark/phase3/results/` | 本地产物，默认被 Git 忽略 |
| `tests/unit/runtime/`、`tests/integration/` | 单元与真实通信检查 |

设计先读 [讨论总结](docs/JobPacer/plan/discussion.md)、[Phase 3.1](docs/JobPacer/plan/phase3.1.md) 和 [Phase 3.2](docs/JobPacer/plan/phase3.2.md)。CPU 阶段事实见 [Phase 3.1 结果](docs/JobPacer/result/phase3.1.md)、[Phase 3.2 结果](docs/JobPacer/result/phase3.2.md)；协作约束见 [AGENTS.md](AGENTS.md)。

## 运行与验证

从仓库根目录使用已有项目环境；远程环境为 `.venv`，不要为复现实验擅自升级 PyTorch/CUDA/NCCL。

```bash
source .venv/bin/activate

# 新 runtime 单元检查及全仓回归
PYTHONPATH=src python -m pytest -q tests/unit/runtime
PYTHONPATH=src python -m pytest -q

# 真实双 rank CPU/Gloo 集成，需要本地 TCP socket 权限
PYTHONPATH=src RUN_JOBPACER_RUNTIME_REPLAY=1 \
  python -m pytest -q tests/integration/test_runtime_replay.py

# GPU 检查前确认可见设备；继承资源分配的 CUDA_VISIBLE_DEVICES
nvidia-smi --query-gpu=index,name,uuid --format=csv
PYTHONPATH=src RUN_JOBPACER_RUNTIME_NCCL=1 \
  python -m pytest -q tests/integration/test_runtime_replay_nccl.py

# 新 runtime 单次 Gloo replay
PYTHONPATH=src:. python -m examples.jobpacer.scripts.run_phase3 \
  --policy fifo --workload balanced --backend gloo \
  --world-size 2 --timeout 20 --output /tmp/jobpacer-phase3-gloo.json

# 新 runtime 双卡 NCCL smoke；这不是性能矩阵
PYTHONPATH=src:. python -m examples.jobpacer.scripts.run_phase3 \
  --policy fifo --workload balanced --backend nccl \
  --world-size 2 --warmup-iterations 5 --setup-timeout 60 --timeout 30 \
  --output /tmp/jobpacer-phase3-nccl.json

git diff --check
```

未开启 opt-in 的跳过项不算通过。GPU/NCCL 正常与失败路径、实际 launch 投影、成员覆盖、tensor 数值和设备完成边界需分别检查；小规模语义测试不证明稳定性能收益。

## 实验产物与复现

本轮远程产物位于 `benchmark/phase3/results/gpu-readiness/preflight-20260927-c1d5008/`，包括 raw、batch manifest、命令与日志、profile、源码归档和 SHA-256 清单。该目录被 Git 忽略，普通 clone/commit 不会带走这些数据；需单独备份并按批次恢复对应源码。起始 HEAD 为 `c1d5008`，实验实现包含未提交修改，不能只检出该 commit 就声称恢复了实验。

本次复核 `inputs/SHA256SUMS.final.txt` 所列文件全部匹配。该清单核对不等于递归验证所有 raw 文件，也不代表结果已提交。

## 历史路径

包根目录的 `Plan`、`AdmissionScheduler`、`ScheduledWork` 及 Phase 1/2 replay 保留作为历史实现和对照。新 runtime 不依赖这些旧核心接口。原 M0–M4.5 记录见 [Gloo/早期 NCCL 实验](docs/experiments/m0-m4.md)、[M4.5 GPU 验收](docs/experiments/m4.5-gpu3080.md)；这些结果不替代 Phase 3 新路径的验收。
