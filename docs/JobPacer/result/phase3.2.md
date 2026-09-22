# JobPacer Phase 3.2 实施与验收结果

日期：2026-09-20；架构迁移复核：2026-09-21。状态：完成（CPU/Gloo）；NCCL/GPU 未验收。

本结果对应 [Phase 3.2 计划](../plan/phase3.2.md) 和
[详细实现方案](../process/phase3.2.md)。完成标准是手写 DAG 在现有 runtime 上正确推进真实
collective，并验证策略输入、顺序和失败语义；不以策略性能胜负作为完成条件。

## 本轮修复

- `tail(v)` 现按 `max(duration(child) + tail(child))` 递推。非对称链测试验证
  `0.003 + 0.011 = 0.014s`，不再把当前通信时长错误计入 tail；diamond 测试也使用了不相等的
  首尾通信估值。
- `multi-group.json` 增加有不同非零后继 tail 的多前沿和 lookahead 预测前沿：
  `comm-a0=0.021s`、`comm-b0=0.003s`、`comm-c0=0.051s`。以该 DAG 产生的真实
  `TaskHint` 驱动 coordinator state-machine 测试，在容量占用期间积累两个候选后，断言 FIFO
  选 `comm-b0`、LTF 选 `comm-a0`。真实 Gloo replay 验证相同 DAG 可执行；首次 dispatch 会受
  真实 OFFER 到达顺序影响，不将单次到达顺序当作策略区分性证据。
- Static LTF 自动生成顺序和外部 `--static-order` 均有两 rank Gloo 集成测试。固定外部顺序
  文件及实际 JSON 保存在 validation batch 中。
- Lookahead 安全声明只检查目标通信的直接前驱；job 内无关的 pending/ready compute 或运行中
  communication 不再阻挡声明。DAG runner 单测检查这些无关节点同时存在时仍会安全声明。
- `comm_declared` 保存调用 `declare()` 前的 `prediction_base_us` 和 `predicted_ready_at_us`；
  `predicted_ready_error_s` 直接使用该预测绝对时刻，而不是 `comm_declared.time_us`。测试检查
  目标前沿、非零预测、Gloo ready 事件及指标锚点。独立 coordinator 测试验证
  `ACTIVE_LOOKAHEAD`、目标到达，以及 deadline 记录和 `LOOKAHEAD_DEADLINE_FALLBACK`。

## 验收

环境：Python 3.13.15；PyTorch 2.13.0+cu129；CPU/Gloo 可用；CUDA 不可用；world size 2；
`max_inflight=1`。

```text
PYTHONPATH=src pytest -q tests/unit
115 passed

PYTHONPATH=src pytest -q
116 passed, 30 skipped

env PYTHONPATH=src RUN_JOBPACER_RUNTIME_REPLAY=1 \
  pytest -q tests/integration/test_runtime_replay.py
25 passed

PYTHONPATH=src python -m py_compile \
  src/runtime_comm_scheduler/dag.py examples/jobpacer/runtime_adapter.py \
  examples/jobpacer/runtime_worker.py examples/jobpacer/run_runtime_replay.py \
  src/runtime_comm_scheduler/runtime/runtime.py \
  src/runtime_comm_scheduler/runtime/coordinator.py \
  src/runtime_comm_scheduler/runtime/policy.py \
  tests/unit/test_jobpacer_dag.py tests/unit/runtime/test_runtime_core.py \
  tests/integration/test_runtime_replay.py
git diff --check
```

25 个 opt-in Gloo replay 包含原 Phase 3.1 四种线性回归、三个 DAG 样例分别执行
FIFO/Static FIFO/Static LTF/LTF/Lookahead、multi-group rank skew、外部 Static LTF order、三类
有界故障和启动前拒绝非法 schema。成功运行校验两 rank tensor 正确、预期 task/node 全集、
group launch 投影和 coordinator dispatch 集合；所有 25 项通过。Lookahead 的具体到达时点受
rank OFFER 传输与本地调度影响，因此 `ACTIVE_LOOKAHEAD` 到达/回退语义另以确定性 coordinator
事件序列测试验收，不用任意 sleep 或单次 Gloo 到达时序冒充确定性保证。

## 持久产物与边界

[validation batch](phase3.2-validation/README.md) 固定保存 FIFO、LTF、Static FIFO、Static LTF
自动顺序、外部 Static LTF 顺序和 Lookahead 的完整 JSON；其中包含 rank 事件、coordinator
decision records、配置及 DAG digest。`manifest.json` 记录所有相关源码/测试文件的 SHA-256、
整体源码快照摘要、Git HEAD 和每个产物摘要。这样摘要覆盖 `src` 中的 DAG 实现、benchmark 输入及结果，
不依赖 pytest 的 `/tmp` 目录，也不把只绑定输入的 manifest digest 当作源码版本。

源码仍在 dirty worktree，当前 Git HEAD 不单独包含本轮实现；固定 batch 的源码快照摘要用于
准确标识本次验收内容，未执行提交操作。DAG 库实现位于
`src/runtime_comm_scheduler/dag.py`，三个可复现实验输入位于 `benchmark/phase3/`；
`examples/jobpacer/` 只保留启动、worker 和历史线性 workload 适配。结果只证明两 rank CPU/Gloo 和 CPU sleep replay，不
外推 GPU/NCCL、真实训练算力共享、在线到达、多通信在途、多资源或跨 host 行为。
