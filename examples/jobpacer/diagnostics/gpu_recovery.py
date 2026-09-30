"""Exercise real startup collision and interrupted-ledger recovery in separate pilot blocks."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from examples.jobpacer.experiments.phase3 import batch
from examples.jobpacer.experiments.phase3.suite import MECHANISM_PILOT_ARMS, ORDER_SEED

# The held socket forces rank zero's real TCPStore bind to fail before any
# communicator or application can start. Subsequent processes allocate normally.
COLLISION_WRAPPER = '''import socket, sys
from examples.jobpacer.runtime import replay_launcher as replay
held = socket.socket()
held.bind(("127.0.0.1", 0))
held.listen()
original = replay._free_port
calls = 0
def occupied(excluded=None):
    global calls
    calls += 1
    return held.getsockname()[1] if calls == 1 else original(excluded)
replay._free_port = occupied
raise SystemExit(replay.main(sys.argv[1:]))
'''


def run(suite_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    checks = {}
    artifacts = []
    for kind in ("startup", "interrupt"):
        root = output / kind
        batch.create_batch(suite_path, root, order_seed=ORDER_SEED, repeats=1,
                           arms=MECHANISM_PILOT_ARMS, timeout_s=30, max_blocks=1)
        if kind == "startup":
            original = batch._run_command
            fired = False
            def collision(*args, **kwargs):
                nonlocal fired
                command = original(*args, **kwargs)
                if not fired:
                    fired = True
                    return [command[0], "-c", COLLISION_WRAPPER, *command[3:]]
                return command
            batch._run_command = collision
            try:
                batch.run_batch(root, timeout=30, warmup=5, allow_pilot=True, resume=False)
            finally:
                batch._run_command = original
        else:
            original_child = batch._run_child
            def interrupt(*args, **kwargs):
                original_child(*args, **kwargs)
                raise KeyboardInterrupt("controlled interruption after child exit, before completion ledger")
            batch._run_child = interrupt
            try:
                batch.run_batch(root, timeout=30, warmup=5, allow_pilot=True, resume=False)
            except KeyboardInterrupt:
                checks["interruption_observed"] = True
            finally:
                batch._run_child = original_child
            batch.run_batch(root, timeout=30, warmup=5, allow_pilot=True, resume=True)
        rows = batch._read_ledger(root / "runs.jsonl")
        accepted = batch._accepted_blocks(rows, MECHANISM_PILOT_ARMS)
        if kind == "startup":
            failures = [row for row in rows if row["failure_class"] == "environment_port_conflict"]
            checks["startup_failure"] = len(failures) == 1
            checks["bounded_cleanup"] = bool(failures) and all(row["child_process_group_exited"] for row in rows)
            checks["whole_block_retry"] = len(accepted) == 1 and set(accepted.values()) == {2}
            ports = []
            for row in rows:
                raw = json.loads((root / row["raw_path"]).read_text())
                ports.extend(item["rendezvous_port"] for item in raw["config"]["rendezvous_startup_attempts"])
            checks["fresh_rendezvous"] = len(set(ports)) == len(ports)
        else:
            checks["interruption_resume"] = (len(accepted) == 1 and set(accepted.values()) == {2}
                and any(row["failure_class"] == "parent_interrupted_before_ledger" for row in rows))
        manifest = json.loads((root / "manifest.json").read_text())
        order = json.loads((root / "order.json").read_text())
        batch._check_resume(root, manifest, order)
        checks["hash_checks"] = True
        for path in root.rglob("*"):
            if path.is_file():
                artifacts.append({"path": str(path.resolve()), "sha256": batch._sha(path.read_bytes())})
    suite = json.loads(suite_path.read_text())
    report = {"gate": "recovery", "passed": all(checks.values()), "checks": checks,
              "source_snapshot_sha256": batch.source_snapshot()["digest"],
              "generator_version": suite["generator_version"],
              "profiles": {key: value["sha256"] for key, value in suite["profiles"].items()},
              "artifacts": artifacts}
    batch._write_json(output / "recovery-audit.json", report)
    return report
