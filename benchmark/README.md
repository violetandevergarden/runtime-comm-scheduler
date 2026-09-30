# Benchmark data

`benchmark/` contains experiment inputs, result artifacts, plots, logs, and documentation. Executable Python and shell
scripts live under `examples/jobpacer/scripts/`; analysis implementation lives under `examples/jobpacer/analysis/`.

- [`phase1.2/`](phase1.2/README.md) keeps the historical Phase 1/2 comparison together, organized by experiment topic.
- [`phase3/`](phase3/README.md) organizes current runtime experiments and results by research question.

Each phase's `results/migration-map.json` records old-to-new paths. Historical manifests, commands, raw traces, and result
bytes were not rewritten as part of the directory migration. Raw result files remain excluded from Git unless already
tracked; experiment inputs and navigation documentation are tracked.
