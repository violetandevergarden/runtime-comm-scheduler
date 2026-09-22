# Phase 3.1 / 3.2 结构整理验收

日期：2026-09-21。状态：完成（CPU/Gloo 结构回归）；NCCL/GPU 未验收。

本结果对应[实施计划](../process/phase3.12fix.md)。这是职责迁移和回归，不是重写 runtime，也不表示性能收益。

## 实施结果

- 将 `src/runtime_comm_scheduler/dag.py` 拆为 `dag/model.py` 与 `dag/runner.py`，包入口仍是
  `runtime_comm_scheduler.dag`。DAG 库只含图模型、校验、tail、静态序列和本地依赖推进，不读取 JSON、
  seed、sleep 样本或 examples。
- 联合前驱边由 `joint_predecessors()` 统一构造，供全图环校验、静态顺序生成和静态顺序校验使用；
  tail 复用相同的边构造逻辑，但只纳入本 job 的数据依赖与 group 顺序。直接构造 `DagGraph` 也可调用
  `validate_graph()` 做完整验证。
- `DagRunner` 现在必须接收 `compute_fn(ComputeNode, stop_event)` 和绝对 deadline；它不再携带 replay
  seed、execution map、jitter 或默认 sleep。compute 样本及采样事件由 adapter 闭包生成；runner 记录
  启动/完成及估计依据。Lookahead、完成解锁、超时诊断和 binding/handle 保留时点未改变。
- 线性和 DAG 单 job 逻辑各自保留，通过 `_run_jobs()` 共享 barrier、首错传播、stop、abort 和同一
  deadline 的 join。初始化期间已创建的 runtime、coordinator、ProcessGroup 均由 `finally` 收尾；失败路径
  不会继续调用 `finish_epoch()`。线性 submit 后计算/消费等待次序保持原样。
- `runtime_adapter.py` 负责 DAG schema、canonical digest、静态 order 文件、确定性 compute sampling、
  torch collective binding，以及 Phase 2 Plan 到线性 runtime task ID 的显式桥接。Binding 依声明的
  shape/dtype 构造 tensor；当前 replay 只执行 sum reduction，其他 reduction 明确拒绝。
- 结果全集校验与指标迁入不依赖 torch 的 `runtime_results.py`。移除了无行为差异且仓库内无调用方的
  `GlooExecutor` 子类；`DirectExecutor` 保持不变。
- benchmark 的 canonical DAG digest 与实施前逐一相同；CLI、DAG 输入 schema、静态排序破同分规则及
  结果校验字段保持。旧 Phase 3.2 结果和 validation batch 未覆盖或改写；其原路径说明保留为历史记录。

## 基线与验收

开始时 worktree 干净，HEAD 为 `232f4495309cba59b7f874e9aafb460b5da2a0bb`。实施前执行
`PYTHONPATH=src pytest -q`：`116 passed, 30 skipped`，4.86s。Python 3.13.15、PyTorch 2.13.0+cu129；
CUDA 不可用。输入文件实施前 SHA-256：

| 输入 | SHA-256 |
| --- | --- |
| `benchmark/phase3/linear.json` | `fb2ef8a75e3f3e71dbb3856a222b49b05d4db7c14f2b7a6e4303d7925645fba7` |
| `benchmark/phase3/diamond.json` | `47a4a724b46718f60f6bfd7fd87444c7600dc9f0f9499494563f8ad1aafa665d` |
| `benchmark/phase3/multi-group.json` | `1c9e92a5aa672117e28146c31b6508057d8097d49b6e699531d15d1674391a59` |

最终验证（全部从仓库根目录）：

| 命令 | 结果 |
| --- | --- |
| `PYTHONPATH=src pytest -q tests/unit/runtime tests/unit/test_jobpacer_dag.py tests/unit/test_jobpacer_runtime_adapter.py tests/unit/test_jobpacer_runtime_worker.py tests/unit/test_jobpacer_runtime_results.py` | `68 passed`，3.00s |
| `PYTHONPATH=src pytest -q` | `131 passed, 32 skipped`，4.14s |
| `env PYTHONPATH=src RUN_JOBPACER_RUNTIME_REPLAY=1 pytest -q tests/integration/test_runtime_replay.py` | `26 passed`，79.36s；CPU/Gloo 双 rank |
| `env PYTHONPATH=src RUN_JOBPACER_RUNTIME_REPLAY=1 pytest -q tests/integration/test_runtime_replay.py -k linear_metadata_mismatch` | `1 passed, 26 deselected`，4.12s；CPU/Gloo 双 rank |
| `git diff --check` | 通过 |

语法编译检查使用以下命令并通过：

```bash
python -m py_compile \
  src/runtime_comm_scheduler/dag/__init__.py \
  src/runtime_comm_scheduler/dag/model.py \
  src/runtime_comm_scheduler/dag/runner.py \
  src/runtime_comm_scheduler/runtime/*.py \
  examples/jobpacer/runtime_adapter.py \
  examples/jobpacer/runtime_worker.py \
  examples/jobpacer/runtime_results.py \
  examples/jobpacer/run_runtime_replay.py \
  tests/unit/test_jobpacer_dag.py \
  tests/unit/test_jobpacer_runtime_adapter.py \
  tests/unit/test_jobpacer_runtime_worker.py \
  tests/unit/test_jobpacer_runtime_results.py \
  tests/integration/test_runtime_replay.py
```

完整 Gloo 矩阵覆盖线性与 DAG replay、FIFO/Static FIFO/Static LTF/LTF/Lookahead、自动及外部静态顺序、
rank skew 和三类 DAG 有界故障。随后新增的线性 metadata-mismatch 故障用例单独通过；该用例确认
worker 仍以原始 tensor binding 与不匹配 TaskSpec 提交，从而在 launch 前拒绝，不把 fault 改成另一种
失败语义。完整矩阵的 26 项运行早于该单独用例加入；它之后独立通过，因此最终 27 个 opt-in 集成场景
均有执行证据，但并非一次命令运行 27 项。默认无故障 Gloo 路径通过完整矩阵。

## 持久产物与边界

[multi-group Static LTF 双 rank Gloo JSON](phase3.12fix-validation/static-ltf-multi-group.json) 保存了完整
两 rank 事件、coordinator decisions、任务全集及 tensor 验收；`validation.status` 为 `ok`，
`all_collectives_correct` 为 `true`，错误列表为空。该运行配置记录了 benchmark digest、Git HEAD 和
`working_tree_dirty=true`。输入与产物校验摘要见
[validation manifest](phase3.12fix-validation/manifest.json)。

源码快照摘要为 `7d158eda7e32ff21b42ed2fcfd475480beec894ea6fa82d9ede604db459fa06d`；tracked diff 摘要为
`4e9fe1bc02d1225ce9c656593a3ada92c6ea52611453a3f33ad5d25be4878941`。最终代码仍在 dirty worktree，
没有提交 commit；实际快照由 manifest 中的逐文件 SHA-256 标识。未安装
依赖。NCCL/GPU、真实训练算力共用、在线 workload、性能收益、多通信在途与跨 host 均不在本次验收范围。

## 后续边界修复（2026-09-21）

对初次验收后提出的六项问题完成修正。`--setup-timeout` 单独约束初始化，`--timeout` 是初始化后唯一的 replay
deadline；同一 rank 上 runner 与 `finish_epoch()` 使用同一期限的剩余量。Coordinator timeout 包含 setup 余量，
用作启动于首个控制事件的 epoch watchdog。指标输出现为 `coordinator_epoch_duration_s`，含 group 注册时间，
不解释为 replay makespan。DAG parser 预检拒绝非 `sum` reduction；tail map 每 rank 初始化一次并由静态排序和
job runners 共用。缺失的 runner/worker/Gloo fault 检查已补，worker 单测不再用 Timer、短 sleep 或耗时范围断言。

验证命令与结果：

| 命令 | 结果 |
| --- | --- |
| `PYTHONPATH=src pytest -q tests/unit/test_jobpacer_dag.py tests/unit/test_jobpacer_runtime_adapter.py tests/unit/test_jobpacer_runtime_worker.py tests/unit/test_jobpacer_runtime_results.py` | `40 passed`，1.60s |
| `PYTHONPATH=src pytest -q` | `139 passed, 34 skipped`，4.01s |
| `env PYTHONPATH=src RUN_JOBPACER_RUNTIME_REPLAY=1 pytest -q tests/integration/test_runtime_replay.py` | `30 passed`，86.78s；双 rank CPU/Gloo |
| `git diff --check` 与涉及模块的 `py_compile` | 通过 |

[后续 Static LTF multi-group Gloo JSON](phase3.12fix-validation/static-ltf-multi-group-followup.json) 的
`validation.status` 为 `ok`，tensor 校验为 true，错误列表为空；metrics 使用新 epoch-duration 字段。其
SHA-256 为 `94da9e94317c6349318f0fca224498fc496be7611a06f4dea07a3e00051cd462`。环境仍为 Python 3.13.15、
PyTorch 2.13.0+cu129、CUDA 不可用，源码 HEAD 未提交且 worktree dirty。旧 artifact 与原验收记录保留，
新增产物和校验摘要已追加至同目录 `manifest.json`。这只是机制/故障覆盖，不构成性能结论；NCCL/GPU 与跨 host
仍未验收。
