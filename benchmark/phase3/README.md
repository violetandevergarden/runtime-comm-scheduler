# Phase 3 benchmark

本目录只保存实验输入、suite 配置、结果和说明，不放执行脚本。实验入口位于
`examples/jobpacer/scripts/`，分析实现位于 `examples/jobpacer/analysis/`。

- [`experiments/README.md`](experiments/README.md)：按研究问题浏览输入与验收条件。
- [`results/README.md`](results/README.md)：按同一分类浏览校准、场景批次和 suite 汇总。
- [`results/migration-map.json`](results/migration-map.json)：旧路径到新路径的迁移索引；历史 manifest、命令及
  raw 内容未改写。
- [`results/suites/20260923-compact/`](results/suites/20260923-compact/)：已有 compact 批次总索引。

Phase 3 保存结果的绘图入口仿照 Phase 1/2 的 SVG 汇总图，读取既有 CSV/JSON，不会重新运行实验：

```bash
PYTHONPATH=src:. python -m examples.jobpacer.analysis.visualize_phase3 suite \
  --results-root benchmark/phase3/results \
  --output-dir benchmark/phase3/results/suites/20260923-compact/figures
```

只生成两种图：整体 makespan 对比、策略/条件时间线对比。suite 目录保存 L0/L1/L5 主批次总览和默认 L1 时间线；各多
策略批次也在自身 `figures/` 下保存这两类图。G0 bridge 在父目录对比 linear/DAG，noise 在父目录按 A/B 汇总；只有一个
臂且没有显式对照的子批次不单独绘图。makespan 图展示运行样本、中位数和描述性 P10–P90；时间线每个策略/条件一个
panel，按组内 makespan 中位数选代表 trace。默认时间线使用 rank 0；可用 `--timeline-batch-dir` 和 `--rank` 选择
总览时间线的批次与 rank。Matplotlib 为可选依赖，缺少时先运行
`python -m pip install -e '.[visualization]'`。

时间线沿用 Phase 1.2 的逐策略分 panel 方式，只显示所选 rank 的本地时钟，并以该 rank 的
`application_release_ts` 为零点；rank 间不对齐、不相减。可单独重画某批次时间线：

```bash
PYTHONPATH=src:. python -m examples.jobpacer.analysis.visualize_phase3 timeline \
  --batch-dir benchmark/phase3/results/readiness/L1-head-misalignment/20260923-compact-main \
  --rank 0 \
  --output /tmp/phase3-L1-rank0-timeline.svg
```

时间线根据旧 Phase 1/2 路径与新 runtime 各自的 trace 字段绘制 producer/compute、准入、通信、应用等待和 validation
泳道；DAG 输入按节点事件绘制。

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
