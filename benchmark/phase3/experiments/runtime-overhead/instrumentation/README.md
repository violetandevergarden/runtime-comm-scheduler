# Instrumentation overhead (E1)

This fixed-input check quantifies the cost of diagnostic timestamps before using detailed traces to explain E2.
It compares `new-static_fifo-minimal` with `new-static_fifo-diagnostic`; both use the same Phase 3 code path,
precreated task bindings, 1 ms completion polling, one collective warmup, and the 4 KiB × 32 communication-chain
input. The five repeats use one fixed epoch and are randomized within each repeat block. They are system-noise
observations, not independent seed blocks.

Run from the repository root with CPU/Gloo thread variables fixed to one:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONPATH=src:. python -m examples.jobpacer.scripts.run_experiments \
  --output-dir benchmark/phase3/results/runtime-overhead/instrumentation/20260924-E1-main \
  --workload benchmark/phase3/experiments/runtime-overhead/communication-chain/4KiB-32x.json \
  --comm-profile benchmark/phase3/results/runtime-overhead/communication-chain/20260924-phase-C-main/4KiB-32x/inputs/comm-profile.json \
  --arms new-static_fifo-minimal,new-static_fifo-diagnostic \
  --baseline new-static_fifo-minimal --secondary-baseline new-static_fifo-diagnostic \
  --seeds 6200 --repeats 5 --compute-jitter 0 --poll-interval 0.001 \
  --binding-preparation precreate --warmup-iterations 1 --bootstrap-samples 0
```

Compare makespan, summed rank process CPU time, and context switches. Diagnostic traces must not be “corrected” by
subtracting one fixed logging cost. If the observed perturbation is comparable to an E3 candidate gain, reduce
instrumentation first. A run is valid only when both modes validate all collectives and report the requested mode.

The 2026-09-24 instrumentation iterations are preserved under
`benchmark/phase3/results/runtime-overhead/instrumentation/`. The initial and intermediate diagnostic modes added
about 50–52 ms to this 32-collective chain, so E2 was held while coordinator snapshots were removed from the
compact diagnostic mode. The final comparison is
[`20260924-E1-compact-diagnostic`](../../../results/runtime-overhead/instrumentation/20260924-E1-compact-diagnostic/):
10/10 runs validated; median makespan was 88.545 ms in minimal mode and 99.764 ms in diagnostic mode. This
11.219 ms difference remains a material measurement perturbation and is reported, not subtracted from later runs.

The 2026-09-25 follow-up is a separate instrumentation revision, not another repeat of the 2026-09-24 batch:
[`20260925-E1-declare-send-lock`](../../../results/runtime-overhead/instrumentation/20260925-E1-declare-send-lock/).
It uses the same 4 KiB × 32 Static FIFO/precreate setup, 1 ms polling and five interleaved repeats per observation
mode, adding only DECLARE call boundaries and per-control-send lock/write timing. All 10 runs validated. Its
minimal/diagnostic makespan medians were 102.038/114.429 ms (paired-repeat difference median +8.919 ms); diagnostic
CPU time was also higher. Treat this as a material trace perturbation and do not pool it with the earlier E1 batch.
The E2 trace review and segmented tables are in the batch's followup-analysis.md.
