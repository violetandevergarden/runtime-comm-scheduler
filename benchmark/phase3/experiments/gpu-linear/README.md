# Archived GPU-linear inputs

The `L0-smoke.json` and `L1-misaligned.json` schema-v1 manifests are retained as historical experiment inputs. The dedicated GPU-linear worker, runner, bridge execution loop, and `--gpu-linear` / `--gpu-bridge` replay options have been retired. These files are not accepted by the current Phase 3 replay or profiling commands.

New GPU runs use schema-v2 DAG inputs with `run_phase3 --dag`; compute and communication profiles also consume that DAG. See [the unified DAG implementation record](../../../../docs/JobPacer/process/phase3-gpu-unified-dag-correction.md) and [the GPU workload plan](../../../../docs/JobPacer/plan/phase3-gpu-workload-and-scheduling.md).

Earlier commands and results remain documented in [the 2026-09-27 result record](../../../../docs/JobPacer/result/phase3-gpu-workload-implementation-20260927.md). Reproducing that historical run requires its archived source snapshot; the current worktree no longer contains its dedicated execution path.
