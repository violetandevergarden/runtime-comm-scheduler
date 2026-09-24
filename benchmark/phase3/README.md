# Phase 3 benchmark

本目录只保存实验输入、suite 配置、结果和说明，不放执行脚本。实验入口位于
`examples/jobpacer/scripts/`，分析实现位于 `examples/jobpacer/analysis/`。

- [`experiments/README.md`](experiments/README.md)：按研究问题浏览输入与验收条件。
- [`results/README.md`](results/README.md)：按同一分类浏览校准、场景批次和 suite 汇总。
- [`results/migration-map.json`](results/migration-map.json)：旧路径到新路径的迁移索引；历史 manifest、命令及
  raw 内容未改写。
- [`results/suites/20260923-compact/`](results/suites/20260923-compact/)：已有 compact 批次总索引。

根目录原有 `linear.json`、`diamond.json`、`multi-group.json` 已移动到
`experiments/dag-semantics/smoke/`，只作为小型机制/正确性样例，不代表 L0–L5 或 G0–G4 的正式验收。

从仓库根目录运行 smoke：

```bash
PYTHONPATH=src:. python -m examples.jobpacer.scripts.run_phase3 \
  --policy fifo --dag benchmark/phase3/experiments/dag-semantics/smoke/multi-group.json \
  --backend gloo --world-size 2 --timeout 20 \
  --output /tmp/jobpacer-phase3-fifo.json
```

正式输入要求提供匹配的 `--comm-profile`。DAG 模型、校验和 runner 位于
`src/runtime_comm_scheduler/dag/`；示例输入映射在 `examples/jobpacer/runtime/runtime_adapter.py`，
启动入口为 `examples/jobpacer/scripts/run_phase3.py`。
