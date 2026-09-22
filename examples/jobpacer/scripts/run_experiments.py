"""Run a small, reproducible Phase 3 experiment batch sequentially."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
RUNTIME_REPLAY = ROOT / "examples/jobpacer/scripts/run_phase3.py"
OLD_REPLAY = ROOT / "examples/jobpacer/scripts/run_phase2.py"
PHASE1_REPLAY = ROOT / "examples/jobpacer/scripts/run_phase1.py"
SOURCE_SNAPSHOT_ROOTS = (
    ROOT / "src/runtime_comm_scheduler/runtime",
    ROOT / "examples/jobpacer/scripts/run_phase3.py",
    ROOT / "examples/jobpacer/runtime/runtime_worker.py",
    ROOT / "examples/jobpacer/analysis/runtime_results.py",
    ROOT / "examples/jobpacer/runtime/runtime_adapter.py",
    ROOT / "examples/jobpacer/workloads.py",
    ROOT / "examples/jobpacer/comm_profile.py",
    ROOT / "examples/jobpacer/scripts/run_experiments.py",
)


def _git(*args: str) -> str | None:
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def _load(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _source_snapshot() -> dict[str, Any]:
    """Hash the source files that can affect a Phase 3 replay."""
    files: list[dict[str, str]] = []
    for root in SOURCE_SNAPSHOT_ROOTS:
        candidates = sorted(root.rglob("*.py")) if root.is_dir() else [root]
        for path in candidates:
            relative = path.relative_to(ROOT).as_posix()
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            files.append({"path": relative, "sha256": digest})
    canonical = "\n".join(f"{item['path']} {item['sha256']}" for item in files).encode()
    return {"digest": hashlib.sha256(canonical).hexdigest(), "files": files}


def _command(args: argparse.Namespace, policy: str, epoch: int, output: Path, *, old: bool = False,
             old_bare: bool = False) -> list[str]:
    source = ("--dag", str(args.dag)) if args.dag else ("--workload", args.workload)
    if old:
        command = [sys.executable, str(PHASE1_REPLAY if old_bare else OLD_REPLAY)]
        if not old_bare:
            command.extend(("--mode", "scheduler"))
        command.extend([
            "--policy", policy,
            "--selection", "runtime_arrival", *source,
            "--backend", args.backend, "--world-size", str(args.world_size),
            "--max-outstanding", "1", "--timeout", str(args.timeout),
            "--completion-poll-interval-s", str(args.poll_interval),
            "--compute-jitter", str(args.compute_jitter), "--epoch", str(epoch),
            "--output", str(output),
        ])
    else:
        command = [
            sys.executable, str(RUNTIME_REPLAY), "--policy", policy,
            *source, "--backend", args.backend,
            "--world-size", str(args.world_size), "--epoch", str(epoch),
            "--compute-jitter", str(args.compute_jitter), "--wait-budget-s", str(args.wait_budget_s),
            "--poll-interval", str(args.poll_interval), "--timeout", str(args.timeout),
            "--output", str(output),
        ]
    if args.comm_profile:
        command.extend(("--comm-profile", str(args.comm_profile)))
    return command


def _measure_row(record: dict[str, Any], result: dict[str, Any] | None, result_path: Path) -> dict[str, Any]:
    config = record["config"]
    status = result.get("validation", {}).get("status") if result else "process_failed"
    if result and "performance" in result:
        performance = result["performance"]
        makespan = performance.get("workload_makespan_us", 0) / 1_000_000
        jobs = performance.get("job_makespans", [])
        jcts = [item.get("makespan_us", 0) / 1_000_000 for item in jobs]
    elif result:
        metrics = result.get("metrics", {})
        jcts = list(metrics.get("job_duration_s", {}).values())
        makespan = max(jcts, default=0.0)
    else:
        jcts, makespan = [], None
    return {
        "run_id": record["run_id"],
        "group": record["group"],
        "policy": config["policy"],
        "workload": config["workload"],
        "epoch": config["epoch"],
        "repeat": config["repeat"],
        "compute_jitter": config["compute_jitter"],
        "status": status,
        "makespan_s": makespan,
        "mean_job_jct_s": sum(jcts) / len(jcts) if jcts else None,
        "slowest_job_jct_s": max(jcts, default=None),
        "wall_time_s": record["wall_time_s"],
        "result_path": str(result_path),
        "error": record.get("error"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--workload")
    source.add_argument("--dag", type=Path)
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--seeds", default="0", help="comma-separated epoch/seed blocks")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--compute-jitter", type=float, default=0.0)
    parser.add_argument("--wait-budget-s", type=float, default=0.02)
    parser.add_argument("--poll-interval", type=float, default=0.001)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--comm-profile", type=Path)
    parser.add_argument("--include-old", action="store_true")
    parser.add_argument("--order-seed", type=int, default=0,
                        help="seed for deterministic strategy order within each seed/repeat block")
    args = parser.parse_args(argv)
    if args.repeats <= 0 or args.world_size < 2 or args.compute_jitter < 0 or args.compute_jitter >= 1:
        parser.error("repeats must be positive, world-size >= 2, and compute-jitter in [0, 1)")
    if args.wait_budget_s < 0 or args.poll_interval <= 0 or args.timeout <= 0:
        parser.error("wait-budget-s must be non-negative; poll interval and timeout must be positive")
    if args.include_old and args.dag:
        parser.error("--include-old currently supports built-in linear workloads only")
    if args.dag and args.comm_profile:
        parser.error("--comm-profile currently applies to linear workloads only")
    if not args.workload and not args.dag:
        args.workload = "balanced"
    if args.comm_profile:
        args.comm_profile = args.comm_profile.resolve()
    epochs = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
    if not epochs:
        parser.error("--seeds must contain at least one integer")

    output_dir = args.output_dir.resolve()
    raw_dir = output_dir / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "inputs").mkdir(exist_ok=True)
    config_snapshot = {
        "workload": args.workload, "dag": str(args.dag) if args.dag else None,
        "backend": args.backend, "world_size": args.world_size,
        "epochs": epochs, "repeats": args.repeats, "compute_jitter": args.compute_jitter,
        "wait_budget_s": args.wait_budget_s, "poll_interval": args.poll_interval,
        "order_seed": args.order_seed,
        "timeout": args.timeout, "comm_profile": str(args.comm_profile) if args.comm_profile else None,
    }
    (output_dir / "inputs" / "batch-config.json").write_text(json.dumps(config_snapshot, indent=2) + "\n")
    if args.comm_profile:
        (output_dir / "inputs" / "comm-profile.json").write_bytes(args.comm_profile.read_bytes())
    if args.dag:
        (output_dir / "inputs" / args.dag.name).write_bytes(args.dag.read_bytes())
    manifest = {
        "schema_version": 2,
        "batch_id": output_dir.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "torch": __import__("torch").__version__,
        },
        "git_head": _git("rev-parse", "HEAD"),
        "working_tree_dirty": bool(_git("status", "--porcelain")),
        "config": config_snapshot,
        "profile_digest": hashlib.sha256(args.comm_profile.read_bytes()).hexdigest()
        if args.comm_profile else None,
        "source_snapshot": _source_snapshot(),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    policies = [("new", policy, False, False) for policy in ("static_fifo", "static_ltf", "fifo", "ltf", "lookahead")]
    if args.include_old:
        policies = [("old-bare", "fifo", True, True), ("old", "fifo", True, False),
                    ("old", "ltf", True, False)] + policies
    records: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    run_index = 0
    runs_path = output_dir / "runs.jsonl"
    with runs_path.open("w") as runs_file:
        for epoch in epochs:
            for repeat in range(args.repeats):
                block_policies = list(policies)
                random.Random(args.order_seed + epoch * 1_000_003 + repeat).shuffle(block_policies)
                block_order = [f"{group}-{policy}" for group, policy, _old, _old_bare in block_policies]
                for group, policy, old, old_bare in block_policies:
                    run_index += 1
                    run_id = f"{run_index:04d}-{group}-{policy}-e{epoch}-r{repeat}"
                    result_path = raw_dir / f"{run_id}.json"
                    stdout_path = raw_dir / f"{run_id}.stdout"
                    stderr_path = raw_dir / f"{run_id}.stderr"
                    command = _command(args, policy, epoch, result_path, old=old, old_bare=old_bare)
                    started = time.time()
                    try:
                        completed = subprocess.run(
                            command,
                            cwd=ROOT,
                            env={
                                **os.environ,
                                "PYTHONPATH": os.pathsep.join(
                                    (str(ROOT / "src"), str(ROOT), os.environ.get("PYTHONPATH", ""))
                                ),
                            },
                            capture_output=True, text=True, timeout=args.timeout + 10,
                        )
                        error = None if completed.returncode == 0 else completed.stderr[-4000:]
                        stdout_path.write_text(completed.stdout)
                        stderr_path.write_text(completed.stderr)
                    except subprocess.TimeoutExpired as exc:
                        completed = None
                        error = f"timeout: {exc}"
                        stdout_path.write_text(exc.stdout or "")
                        stderr_path.write_text(exc.stderr or "")
                    wall_time_s = time.time() - started
                    result = _load(result_path)
                    record = {
                        "run_id": run_id, "group": group,
                        "config": {"policy": policy, "workload": args.workload or str(args.dag), "epoch": epoch,
                                   "repeat": repeat, "compute_jitter": args.compute_jitter,
                                   "order_seed": args.order_seed, "block_order": block_order},
                        "command": command, "started_at": datetime.now(timezone.utc).isoformat(),
                        "wall_time_s": wall_time_s,
                        "returncode": completed.returncode if completed is not None else None,
                        "result_path": str(result_path), "error": error,
                    }
                    records.append(record)
                    runs_file.write(json.dumps(record, sort_keys=True) + "\n")
                    runs_file.flush()
                    summary_rows.append(_measure_row(record, result, result_path))

    fields = list(summary_rows[0]) if summary_rows else []
    with (output_dir / "summary.csv").open("w", newline="") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    failed = sum(row["status"] != "ok" for row in summary_rows)
    print(json.dumps({"output_dir": str(output_dir), "runs": len(records), "failed": failed}, sort_keys=True))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
