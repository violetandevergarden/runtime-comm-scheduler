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

DAG 的解析、校验、策略摘要和本地推进实现位于
`src/runtime_comm_scheduler/dag.py`；`examples/jobpacer/` 只保留可执行 harness。
