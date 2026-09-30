# 批次清单

`formal.json` 保留原 6570 次方案供追溯。后续 CPU/Gloo 实验使用
`compact.json`：核心 660、机制 90、isolated 诊断 30，共 780 次；
Lookahead 机制成立后可再跑 120 次。`run_compact_suite.py` 只展开这份固定清单，
默认只读预览；`run_experiments.py` 仍可单独运行某个场景。

运行输出由 runner 按研究类别写入 `benchmark/phase3/results/`：各场景批次位于
`<category>/<scenario>/<suite-id>-<phase>/`，suite manifest/index 位于
`results/suites/<suite-id>/`。`--output-dir` 应提供这一 suite index 路径；
它不能再指向 `/tmp`。raw 结果不会复制到 suite 总索引。

先在最终 CPU affinity 和线程环境下生成新 profile；新 profile 记录这两项，
正式套件会核对。不带这些字段的旧 profile 必须重新校准。先将
`JOBPACER_CPUS` 设为实际可用 CPU 列表，`JOBPACER_TIE` 设为独立 pilot 后
预先确定的持平阈值，再在两条命令中使用同一 CPU 列表：

```bash
taskset -c "$JOBPACER_CPUS" env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  PYTHONPATH=src:. python -m examples.jobpacer.scripts.run_comm_profile \
  --workload benchmark/phase3/experiments/calibration/multi-scale.json \
  --backend gloo --world-size 2 --warmup 5 --iterations 30 \
  --output /tmp/jobpacer-phase3-compact-profile.json

PYTHONPATH=src:. python -m examples.jobpacer.scripts.run_compact_suite \
  --output-dir benchmark/phase3/results/suites/20260924-compact --phase all
```

预览列出每个场景的组数、seed、repeat 和次数；提供已有 `summary.csv` 或批次目录
给 `--history-summary` 才会估算对应场景/组的耗时，覆盖不足时显示 `null`。
执行时显式指定阶段、profile、CPU 列表和 `--execute`；主实验与可选 Lookahead
还必须通过 `--tie-threshold` 写入预先确定的持平阈值：

```bash
PYTHONPATH=src:. python -m examples.jobpacer.scripts.run_compact_suite \
  --output-dir benchmark/phase3/results/suites/20260924-compact --phase mechanism --execute \
  --comm-profile /tmp/jobpacer-phase3-compact-profile.json --cpu-affinity "$JOBPACER_CPUS"

PYTHONPATH=src:. python -m examples.jobpacer.scripts.run_compact_suite \
  --output-dir benchmark/phase3/results/suites/20260924-compact --phase main --case L0 --execute --resume \
  --comm-profile /tmp/jobpacer-phase3-compact-profile.json --cpu-affinity "$JOBPACER_CPUS" \
  --tie-threshold "$JOBPACER_TIE"
```

完整主矩阵可省略 `--case`；L2/G2 在各策略 5 次 pilot 中需至少 3 次满足严格
候选竞争才可扩批。可选 Lookahead 要求 L3 准时到达与 L4 超时回退各至少 4/5，
不足时停止。重新执行同一 suite 目录需 `--resume`；输入、profile、源码、
CPU affinity 或线程环境变化会拒绝续接。失败重试写新 attempt 文件，原始记录保留。
`--phase isolated` 调用 shared/job-0/job-1 随机交错的 30 次诊断，不并入主结果。
每个场景保留完整 `paired-summary.csv`，套件另根据清单写 `primary-paired.csv`；
`mechanism-summary.csv` 逐组汇总 gate、指定候选竞争与 Lookahead 到达率。

同一 suite 目录可以分阶段运行；首次运行建立 suite manifest，以后阶段使用
`--resume`。`--case` 可只执行已核对过的配置；未完成的场景不会被误计为已完成。

`screening.json` 固定了 2026-09-22 零扰动批次，只能反映固定输入下的系统噪声。
`perturbation-screening.json`、`priority-pilot.json`、`wait-window-pilot.json` 和
`formal.json` 均为精简前的计划或 pilot 配置，保留供历史结果核对，不再作为
本轮执行入口。旧 1 MiB profile 的时长、旧批次的失败与负收益也不得直接并入
新环境下的配对统计。

L2/G2 自然到达的全部 run 都应保留，分别报告 gate-first、指定候选竞争与
机制触发率；不得事后只选 gate-first 样本计算性能收益。L3/L4/G4 必须先通过
等待机制检查，再考虑可选性能批次。G1 保留为无候选竞争的负对照。

此前分批测得 shared JCT 比 isolated 更短，旧 slowdown 暂不解释。当前套件的
isolated 阶段由 `run_interleaved_isolated.py` 在固定环境中随机交错执行三种输入；
这 30 次只用于诊断，不能充当主矩阵不同扰动样本的正式 slowdown 分母。
