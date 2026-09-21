# Phase 3.2 validation batch

This fixed directory preserves the targeted two-rank CPU/Gloo evidence instead of relying on
pytest's temporary directory. `manifest.json` records the source snapshot and artifact digests.
The JSON outputs include the complete rank events and coordinator decision records.
Deterministic `CoordinatorState` unit tests use the same DAG-derived hints to verify an
`ACTIVE_LOOKAHEAD` wait, target arrival, and deadline fallback without depending on OS scheduling.

| Artifact | Scenario |
| --- | --- |
| `fifo-multi-group.json` | Dynamic FIFO on the multi-frontier DAG |
| `ltf-multi-group.json` | Dynamic LTF using distinct nonzero DAG tails |
| `static-fifo-multi-group.json` | Generated static FIFO order |
| `static-ltf-generated-multi-group.json` | Generated static LTF order |
| `static-ltf-external-multi-group.json` | External `--static-order` file below |
| `lookahead-multi-group.json` | Two-rank Gloo replay with the named nonzero DAG prediction and its timestamp-based error metric |

Reproduce the batch from the repository root:

```bash
PYTHONPATH=src python examples/jobpacer/run_runtime_replay.py --policy fifo --dag benchmark/phase3/multi-group.json --backend gloo --world-size 2 --timeout 20 --epoch 7 --output docs/JobPacer/result/phase3.2-validation/fifo-multi-group.json
PYTHONPATH=src python examples/jobpacer/run_runtime_replay.py --policy ltf --dag benchmark/phase3/multi-group.json --backend gloo --world-size 2 --timeout 20 --epoch 7 --output docs/JobPacer/result/phase3.2-validation/ltf-multi-group.json
PYTHONPATH=src python examples/jobpacer/run_runtime_replay.py --policy static_fifo --dag benchmark/phase3/multi-group.json --backend gloo --world-size 2 --timeout 20 --epoch 7 --output docs/JobPacer/result/phase3.2-validation/static-fifo-multi-group.json
PYTHONPATH=src python examples/jobpacer/run_runtime_replay.py --policy static_ltf --dag benchmark/phase3/multi-group.json --backend gloo --world-size 2 --timeout 20 --epoch 7 --output docs/JobPacer/result/phase3.2-validation/static-ltf-generated-multi-group.json
PYTHONPATH=src python examples/jobpacer/run_runtime_replay.py --policy static_ltf --dag benchmark/phase3/multi-group.json --static-order docs/JobPacer/result/phase3.2-validation/static-ltf-order.json --backend gloo --world-size 2 --timeout 20 --epoch 7 --output docs/JobPacer/result/phase3.2-validation/static-ltf-external-multi-group.json
PYTHONPATH=src python examples/jobpacer/run_runtime_replay.py --policy lookahead --dag benchmark/phase3/multi-group.json --backend gloo --world-size 2 --timeout 20 --epoch 7 --output docs/JobPacer/result/phase3.2-validation/lookahead-multi-group.json
```
