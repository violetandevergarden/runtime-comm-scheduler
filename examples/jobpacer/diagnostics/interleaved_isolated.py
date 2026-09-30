"""Run the L0 shared and isolated denominator pilot in interleaved blocks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from examples.jobpacer.paths import repository_path, resolve_migrated_path
from examples.jobpacer.experiments import gloo_phase3_batch as batch
from examples.jobpacer.gloo.workloads import load_workload


ROOT = Path(__file__).resolve().parents[3]
REPLAY = ROOT / "examples/jobpacer/runtime/replay_launcher.py"
SEEDS = (0,)
REPEATS = 5
POLICIES = ("fifo", "ltf")
CONDITIONS = {
    "shared": ROOT / "benchmark/phase3/experiments/baseline/L0-balanced.json",
    "isolated-job-0": ROOT / "benchmark/phase3/experiments/isolated/job-0.json",
    "isolated-job-1": ROOT / "benchmark/phase3/experiments/isolated/job-1.json",
}


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    batch._write_csv(path, rows)


def _environment() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": __import__("platform").platform(),
        "cpu_count": os.cpu_count(),
        "cpu_model": batch._cpu_model(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "thread_environment": {key: os.environ.get(key) for key in
                               ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                "NUMEXPR_NUM_THREADS", "GLOO_SOCKET_IFNAME")},
        "torch": __import__("torch").__version__,
    }


def _batch_manifest(condition: str, workload: Path, profile: Path, output: Path,
                    environment: dict[str, Any], snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 3,
        "batch_id": output.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": environment,
        "git_head": batch._git("rev-parse", "HEAD"),
        "working_tree_dirty": bool(batch._git("status", "--porcelain")),
        "config": {
            "workload": str(workload),
            "dag": None,
            "backend": "gloo",
            "world_size": 2,
            "epochs": SEEDS,
            "repeats": REPEATS,
            "compute_jitter": 0.0,
            "wait_budget_s": 0.02,
            "poll_interval": 0.001,
            "warmup_iterations": 1,
            "order_seed": 20260923,
            "timeout": 60.0,
            "baseline": "new-static_fifo",
            "condition": condition,
        },
        "profile_digest": hashlib.sha256(profile.read_bytes()).hexdigest(),
        "workload_digest": hashlib.sha256(workload.read_bytes()).hexdigest(),
        "source_snapshot": snapshot,
        "orchestration_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--comm-profile", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    profile = resolve_migrated_path(
        repository_path(args.comm_profile), ROOT / "benchmark/phase3/results/migration-map.json"
    ).resolve(strict=True)
    output = repository_path(args.output_dir).resolve()
    if output.exists() and not args.resume:
        parser.error("output directory exists; use --resume")
    if args.resume and not output.exists():
        parser.error("--resume requires an existing output directory")
    if not output.exists():
        output.parent.mkdir(parents=True, exist_ok=True)
        output.mkdir(exist_ok=False)

    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    from examples.jobpacer.experiments.compact_suite import _validate_profile
    _validate_profile(profile, [{"workload": path.relative_to(ROOT / "benchmark/phase3/experiments").as_posix()}
                                for path in CONDITIONS.values()],
                      set(os.sched_getaffinity(0)),
                      {key: "1" for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                             "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
    environment = _environment()
    snapshot = batch._source_snapshot()
    manifests: dict[str, dict[str, Any]] = {}
    dirs: dict[str, Path] = {}
    for condition, workload in CONDITIONS.items():
        condition_dir = output / condition
        if not args.resume:
            condition_dir.mkdir()
            (condition_dir / "raw").mkdir()
            (condition_dir / "inputs").mkdir()
            (condition_dir / "inputs" / workload.name).write_bytes(workload.read_bytes())
            (condition_dir / "inputs" / "comm-profile.json").write_bytes(profile.read_bytes())
        manifest = _batch_manifest(condition, workload, profile, condition_dir, environment, snapshot)
        if args.resume:
            previous = batch._load(condition_dir / "manifest.json")
            if not previous or any(json.dumps(previous.get(key), sort_keys=True) !=
                                   json.dumps(manifest.get(key), sort_keys=True) for key in
                                   ("config", "environment", "profile_digest", "workload_digest",
                                    "source_snapshot", "orchestration_sha256")):
                parser.error(f"isolated resume rejected: {condition} manifest changed")
        else:
            (condition_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        manifests[condition] = manifest
        dirs[condition] = condition_dir

    for isolated in ("isolated-job-0", "isolated-job-1"):
        batch._check_isolated_batch(manifests["shared"], manifests[isolated],
                                    load_workload(str(CONDITIONS["shared"])))

    suite_manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seeds": SEEDS,
        "repeats": REPEATS,
        "compute_jitter": 0.0,
        "wait_budget_s": 0.02,
        "poll_interval": 0.001,
        "warmup_iterations": 1,
        "order_seed": 20260923,
        "condition_order": "randomized within each seed block",
        "policy_order": "randomized once per seed block and reused across conditions",
        "conditions": {key: str(value) for key, value in CONDITIONS.items()},
        "condition_manifests": {key: str(dirs[key] / "manifest.json") for key in CONDITIONS},
        "source_snapshot": snapshot,
        "orchestration_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": environment,
    }
    if args.resume:
        previous = batch._load(output / "manifest.json")
        if not previous or any(json.dumps(previous.get(key), sort_keys=True) !=
                               json.dumps(suite_manifest.get(key), sort_keys=True) for key in
                               ("seeds", "repeats", "compute_jitter", "environment",
                                "source_snapshot", "orchestration_sha256")):
            parser.error("isolated resume rejected: suite manifest changed")
    else:
        (output / "manifest.json").write_text(json.dumps(suite_manifest, indent=2, sort_keys=True) + "\n")

    summaries: dict[str, list[dict[str, Any]]] = {condition: [] for condition in CONDITIONS}
    jobs: dict[str, list[dict[str, Any]]] = {condition: [] for condition in CONDITIONS}
    mechanisms: dict[str, list[dict[str, Any]]] = {condition: [] for condition in CONDITIONS}
    order_rows: list[dict[str, Any]] = []
    run_index = 0
    latest = {key: batch._latest_records(dirs[key] / "runs.jsonl") if args.resume else {}
              for key in CONDITIONS}
    planned: dict[str, list[dict[str, Any]]] = {key: [] for key in CONDITIONS}
    runs_files = {key: (dirs[key] / "runs.jsonl").open("a") for key in CONDITIONS}
    failures = 0
    try:
        for repeat in range(REPEATS):
            seed = SEEDS[0]
            rng = random.Random(20260923 + repeat)
            condition_order = list(CONDITIONS)
            policy_order = list(POLICIES)
            rng.shuffle(condition_order)
            rng.shuffle(policy_order)
            order_rows.append({"seed": seed, "repeat": repeat, "condition_order": condition_order,
                               "policy_order": [f"new-{policy}" for policy in policy_order]})
            block_order = [f"{condition}-new-{policy}"
                           for condition in condition_order for policy in policy_order]
            for condition in condition_order:
                workload = CONDITIONS[condition]
                condition_dir = dirs[condition]
                for policy in policy_order:
                    run_index += 1
                    run_id = f"{run_index:04d}-{condition}-{policy}-e{seed}-r{repeat}"
                    item = {"run_id": run_id, "group": "new", "policy": policy,
                            "epoch": seed, "repeat": repeat, "compute_jitter": 0.0}
                    planned[condition].append(item)
                    if batch._successful_record(latest[condition].get(run_id), item):
                        continue
                    attempt, result_path, stdout_path, stderr_path = batch._attempt_paths(
                        condition_dir / "raw", run_id)
                    command = [sys.executable, str(REPLAY), "--policy", policy,
                               "--workload", str(workload), "--backend", "gloo",
                               "--world-size", "2", "--epoch", str(seed),
                               "--compute-jitter", "0.0", "--wait-budget-s", "0.02",
                               "--poll-interval", "0.001", "--timeout", "60",
                               "--warmup-iterations", "1", "--comm-profile", str(profile),
                               "--output", str(result_path)]
                    started = time.time()
                    started_at = datetime.fromtimestamp(started, timezone.utc).isoformat()
                    try:
                        completed = subprocess.run(
                            command, cwd=ROOT,
                            env={**os.environ, "PYTHONPATH": os.pathsep.join(
                                (str(ROOT / "src"), str(ROOT), os.environ.get("PYTHONPATH", "")))},
                            capture_output=True, text=True, timeout=70,
                        )
                        returncode = completed.returncode
                        error = None if returncode == 0 else completed.stderr[-4000:]
                        stdout_path.write_text(completed.stdout)
                        stderr_path.write_text(completed.stderr)
                    except subprocess.TimeoutExpired as exc:
                        returncode = None
                        error = f"timeout: {exc}"
                        stdout_path.write_text(str(exc.stdout or ""))
                        stderr_path.write_text(str(exc.stderr or ""))
                    wall_time = time.time() - started
                    record = {
                        "run_id": run_id, "group": "new",
                        "attempt": attempt,
                        "retry_of": latest[condition].get(run_id, {}).get("result_path"),
                        "config": {"policy": policy, "workload": str(workload), "epoch": seed,
                                   "repeat": repeat, "compute_jitter": 0.0, "order_seed": 20260923,
                                   "block_order": block_order, "condition": condition},
                        "command": command, "started_at": started_at,
                        "ended_at": datetime.now(timezone.utc).isoformat(),
                        "wall_time_s": wall_time, "returncode": returncode,
                        "result_path": str(result_path),
                        "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest()
                        if result_path.is_file() else None,
                        "error": error,
                    }
                    runs_files[condition].write(json.dumps(record, sort_keys=True) + "\n")
                    runs_files[condition].flush()
                    os.fsync(runs_files[condition].fileno())
                    latest[condition][run_id] = record
    finally:
        for file in runs_files.values():
            file.close()

    for condition in CONDITIONS:
        if set(latest[condition]) - {item["run_id"] for item in planned[condition]}:
            parser.error(f"isolated resume rejected: unexpected run in {condition}")
        for item in planned[condition]:
            record = latest[condition][item["run_id"]]
            result_path = batch._record_result_path(record)
            result = batch._load(result_path)
            summaries[condition].append(batch._measure_row(record, result, result_path))
            jobs[condition].extend(batch._job_rows(record, result))
            mechanisms[condition].append(batch._mechanism_row(record, result))
            failures += int(summaries[condition][-1]["status"] != "ok")
        _write_rows(dirs[condition] / "summary.csv", summaries[condition])
        _write_rows(dirs[condition] / "mechanisms.csv", mechanisms[condition])
        if condition != "shared":
            _write_rows(dirs[condition] / "jobs.csv", jobs[condition])
    batch._attach_isolated(
        jobs["shared"],
        [dirs["isolated-job-0"] / "jobs.csv", dirs["isolated-job-1"] / "jobs.csv"],
        manifests["shared"],
    )
    _write_rows(dirs["shared"] / "jobs.csv", jobs["shared"])
    (output / "interleaving-order.json").write_text(json.dumps(order_rows, indent=2) + "\n")
    def _slowdown_median(policy: str, job: str) -> float | None:
        import statistics
        values = [float(row["slowdown"]) for row in jobs["shared"]
                  if row["policy"] == policy and row["job_id"] == job and row["slowdown"] is not None]
        return statistics.median(values) if values else None

    (output / "summary.json").write_text(json.dumps({
        "runs": run_index, "failed": failures,
        "output_dir": str(output),
        "condition_order": order_rows,
        "slowdown_median_by_policy_job": {
            policy: {job: _slowdown_median(policy, job) for job in ("job-0", "job-1")}
            for policy in POLICIES
        },
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output_dir": str(output), "runs": run_index, "failed": failures}, sort_keys=True))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
