# Control-path decomposition (E2)

E2 compares the legacy Phase 2 FIFO path with Phase 3 Static FIFO/precreate for two fixed inputs:
4 KiB × 32 and 1 MiB × 32. Each input/arm has five repeats at seed 6200. The runner randomizes all four
configurations inside each repeat block, then executes one replay at a time. Both paths use one warmup, 1 ms
completion polling, zero compute perturbation, two CPU/Gloo ranks and a shared calibrated communication profile.

Run:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONPATH=src:. python -m examples.jobpacer.scripts.run_control_path_diagnostic \
  --output-dir benchmark/phase3/results/runtime-overhead/control-path/20260924-E2-main \
  --workload-4k benchmark/phase3/experiments/runtime-overhead/communication-chain/4KiB-32x.json \
  --workload-1m benchmark/phase3/experiments/runtime-overhead/communication-chain/1MiB-32x.json \
  --comm-profile benchmark/phase3/results/runtime-overhead/communication-chain/20260924-phase-C-main/4KiB-32x/inputs/comm-profile.json
```

The batch freezes both input files, the profile, source hashes, environment and randomized order. Makespan is the
maximum rank-local application-release→application-end duration on both paths. `summary.csv` contains one row per
replay; `task-timings.csv` retains per-collective intervals, including first-probe delay and probe spacing;
`coordinator-events.csv` retains queue, processing, policy, capacity-release and grant-publication boundaries;
`coordinator-task-timings.csv` classifies capacity wait, candidate readiness and post-readiness processing.
`paired-summary.csv` is descriptive within-repeat data for each fixed input; it does not treat the 32 collectives
as independent samples or claim seed-level uncertainty.

The legacy `legal_candidate_present_during_gap` summary remains for backward compatibility, but includes the
next dispatch snapshot. Use the strict-before-dispatch and first-eligible/capacity-release boundaries for diagnosis.

Stage E3 uses the same runner and conditions, but compares the original periodic probe against the opt-in
submission wakeup. The candidate must still preserve periodic polling:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONPATH=src:. python -m examples.jobpacer.scripts.run_control_path_diagnostic \
  --stage E3 \
  --output-dir benchmark/phase3/results/runtime-overhead/control-path/20260924-E3-completion-wakeup \
  --workload-4k benchmark/phase3/experiments/runtime-overhead/communication-chain/4KiB-32x.json \
  --workload-1m benchmark/phase3/experiments/runtime-overhead/communication-chain/1MiB-32x.json \
  --comm-profile benchmark/phase3/results/runtime-overhead/communication-chain/20260924-phase-C-main/4KiB-32x/inputs/comm-profile.json
```

For an existing completed batch, `--rebuild-derived` regenerates CSV/analysis outputs from SHA-verified raw files
without changing the manifest, run ledger or raw JSON. Render separate per-size makespan and timeline figures
(including a two-rank paired local-clock view) with:

```bash
PYTHONPATH=src:. python -m examples.jobpacer.analysis.visualize_phase3 control-path \
  --batch-dir benchmark/phase3/results/runtime-overhead/control-path/20260924-E3-completion-wakeup
```

The 2026-09-25 follow-up re-analyzes E2 without rerunning it. It classifies each next-grant gap using coordinator-local
OFFER, eligibility, capacity-release, decision and writer boundaries, and supplements E2's missing rank-local DECLARE
start/end with a separate 10-run E1 batch. The run-level segmented tables and paired two-rank local-clock timeline are
in [the follow-up analysis](../../../results/runtime-overhead/instrumentation/20260925-E1-declare-send-lock/followup-analysis.md).
The repeated central wait is predominantly for the final member OFFER to reach the coordinator input queue; that
timestamp is not interpreted as one-way network latency. Policy/decision processing and launch-worker dequeue are
smaller. Diagnostic instrumentation perturbs makespan materially, so its absolute performance is not treated as a
minimal-observation baseline.

## Declaration-mode ablation (F)

[`declaration-mode.json`](declaration-mode.json) freezes the single hypothesis and inputs. F1 and F2 each compare
five paired repeats of `before-producer` and `on-submit` on the same 4 KiB × 32 CPU/Gloo chain. Both use
Static FIFO, precreated bindings, one 1 ms completion poll, one warmup, zero jitter, and serial randomized arm order.
F1 uses minimal observation; F2 uses the existing compact diagnostic events. No E2/E3 baseline is reused.

Run each gated phase in a fresh output directory:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONPATH=src:. python -m examples.jobpacer.scripts.run_control_path_diagnostic \
  --stage F1 \
  --output-dir benchmark/phase3/results/runtime-overhead/control-path/20260925-F-declaration-mode/minimal-4KiB \
  --workload-4k benchmark/phase3/experiments/runtime-overhead/communication-chain/4KiB-32x.json \
  --comm-profile benchmark/phase3/results/runtime-overhead/communication-chain/20260924-phase-C-main/4KiB-32x/inputs/comm-profile.json

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONPATH=src:. python -m examples.jobpacer.scripts.run_control_path_diagnostic \
  --stage F2 \
  --output-dir benchmark/phase3/results/runtime-overhead/control-path/20260925-F-declaration-mode/diagnostic-4KiB \
  --workload-4k benchmark/phase3/experiments/runtime-overhead/communication-chain/4KiB-32x.json \
  --comm-profile benchmark/phase3/results/runtime-overhead/communication-chain/20260924-phase-C-main/4KiB-32x/inputs/comm-profile.json
```

F1's `summary.csv` records per-rank application durations, rank-local application-to-drain gaps, CPU time,
context switches, correctness and expected protocol message counts. `paired-summary.csv` contains all five paired
observations, while `paired-summary-statistics.csv` reports the median and range without treating collectives as
independent observations. F2 additionally emits `run-segments.csv`: task intervals are first reduced within each
replay (median/P90/max), with rank-local and coordinator-clock quantities kept separate. Render its makespan and
paired rank-local timeline using the existing analyzer:

```bash
PYTHONPATH=src:. python -m examples.jobpacer.analysis.visualize_phase3 control-path \
  --batch-dir benchmark/phase3/results/runtime-overhead/control-path/20260925-F-declaration-mode/diagnostic-4KiB
```

If F1's paired minimal makespan and F2's run-reduced mechanism point in the same direction, the gated 1 MiB check is:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONPATH=src:. python -m examples.jobpacer.scripts.run_control_path_diagnostic \
  --stage F3 \
  --output-dir benchmark/phase3/results/runtime-overhead/control-path/20260925-F-declaration-mode/minimal-1MiB \
  --workload-1m benchmark/phase3/experiments/runtime-overhead/communication-chain/1MiB-32x.json \
  --comm-profile benchmark/phase3/results/runtime-overhead/communication-chain/20260924-phase-C-main/4KiB-32x/inputs/comm-profile.json
```

Minimal-observation F1/F3 figures contain the makespan comparison only; F2 also has rank-local timelines. F3's
paired five-repeat result was directionally consistent with F1 and its F2 mechanism; the 1 MiB check completed
without a correctness or adverse-tail signal. This opened the F4 application regression gate.

F4 compares declaration modes using the formal L0/L1 inputs, `new_fifo`, five fixed seed blocks and three paired
repeats per workload/seed/mode (60 serial CPU/Gloo replays total), with 0.3 execution jitter. The calibrated
multi-scale profile, 1 ms completion poll, precreated bindings, one warmup and minimal observation remain fixed.
Modes are randomized within each `(workload, seed, repeat)` pair. Execution samples must match exactly across the
two modes and match the expected producer/consumer sample count. Results include repeat-level pairs, within-seed
median pairs, per-job JCT, protocol counts, CPU/context-switch measures and raw outputs.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONPATH=src:. python -m examples.jobpacer.scripts.run_control_path_diagnostic \
  --stage F4 \
  --output-dir benchmark/phase3/results/runtime-overhead/control-path/20260925-F-declaration-mode/application-L0-L1 \
  --workload-l0 benchmark/phase3/experiments/baseline/L0-balanced.json \
  --workload-l1 benchmark/phase3/experiments/readiness/L1-head-misalignment.json \
  --comm-profile benchmark/phase3/results/calibration/multi-scale/phase3-compact-20260923/profile.json \
  --seeds 5300,5301,5302,5303,5304 --repeats 3 --compute-jitter 0.3
```

F4 is a fixed L0/L1 application regression over five seed blocks, not evidence for broader workload generalization.
Its minimal-observation figures show makespan only. Do not extend to additional policies or DAGs from this batch alone.
