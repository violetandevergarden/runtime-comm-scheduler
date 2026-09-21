# Phase 3 DAG benchmark 输入

这里保存 Phase 3 runtime replay 使用的确定性 DAG 输入；它们是实验数据，不是示例程序源码。

- `linear.json`：两个线性 job，用于验证 DAG 路径与基础策略。
- `diamond.json`：含分叉和汇合，用于验证依赖推进与关键路径 tail。
- `multi-group.json`：多 group、多前沿输入，用于 FIFO、LTF、Static 和 Lookahead 对照。

从仓库根目录运行示例：

```bash
PYTHONPATH=src python examples/jobpacer/run_runtime_replay.py \
  --policy fifo --dag benchmark/phase3/multi-group.json \
  --backend gloo --world-size 2 --timeout 20 \
  --output /tmp/jobpacer-phase3-fifo.json
```

DAG 模型、校验、tail、静态序列与通用依赖 runner 位于
`src/runtime_comm_scheduler/dag/`；JSON schema、canonical digest、CPU compute sampling 和
rank-local collective binding 位于 `examples/jobpacer/runtime_adapter.py`。结果校验/指标在
`runtime_results.py`，启动与生命周期在 `runtime_worker.py` 和 `run_runtime_replay.py`。

依赖方向是 `examples → dag → runtime`；正式 `src/` 不依赖 examples，runtime 不依赖 DAG。
benchmark 输入 schema 和策略估计不变。重构后的 CPU/Gloo 回归记录见
[`docs/JobPacer/result/phase3.12fix.md`](../../docs/JobPacer/result/phase3.12fix.md)。
