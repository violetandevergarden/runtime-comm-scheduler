"""Run a selected Phase 3 experiment block sequentially, with safe resume."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from examples.jobpacer.analysis.benchmark_paths import (
    is_formal_experiment_input, repository_path, resolve_migrated_path,
)
from examples.jobpacer.workloads import load_workload


ROOT = Path(__file__).resolve().parents[3]
RUNTIME_REPLAY = ROOT / "examples/jobpacer/scripts/run_phase3.py"
OLD_REPLAY = ROOT / "examples/jobpacer/scripts/run_phase2.py"
PHASE1_REPLAY = ROOT / "examples/jobpacer/scripts/run_phase1.py"
FORMAL_INPUT_ROOT = (ROOT / "benchmark/phase3/experiments").resolve()
MIGRATION_MAP = ROOT / "benchmark/phase3/results/migration-map.json"
ARM_SPECS = {
    "old-bare-fifo": ("old-bare", "fifo", True, True),
    "old-fifo": ("old", "fifo", True, False),
    "old-ltf": ("old", "ltf", True, False),
    **{f"new-{policy}": ("new", policy, False, False)
       for policy in ("static_fifo", "static_ltf", "fifo", "ltf", "lookahead")},
    "new-ltf-on-ready": ("new", "ltf", False, False),
    "new-ltf-precreate": ("new", "ltf", False, False),
    "new-ltf-poll-1ms": ("new", "ltf", False, False),
    "new-ltf-poll-0.2ms": ("new", "ltf", False, False),
}
ARM_BINDING_PREPARATION = {
    "new-ltf-on-ready": "on-ready",
    "new-ltf-precreate": "precreate",
    "new-ltf-poll-1ms": "precreate",
    "new-ltf-poll-0.2ms": "precreate",
}
ARM_POLL_INTERVAL = {"new-ltf-poll-1ms": 0.001, "new-ltf-poll-0.2ms": 0.0002}
SOURCE_SNAPSHOT_ROOTS = (
    ROOT / "src/runtime_comm_scheduler/dag",
    ROOT / "src/runtime_comm_scheduler/runtime",
    ROOT / "src/runtime_comm_scheduler/plan.py",
    ROOT / "src/runtime_comm_scheduler/scheduler.py",
    ROOT / "src/runtime_comm_scheduler/work.py",
    ROOT / "examples/jobpacer/runtime/replay_worker.py",
    ROOT / "examples/jobpacer/runtime/plan_builder.py",
    ROOT / "examples/jobpacer/scripts/run_phase3.py",
    ROOT / "examples/jobpacer/scripts/run_phase2.py",
    ROOT / "examples/jobpacer/scripts/run_phase1.py",
    ROOT / "examples/jobpacer/runtime/runtime_worker.py",
    ROOT / "examples/jobpacer/analysis/runtime_results.py",
    ROOT / "examples/jobpacer/analysis/visualize_phase3.py",
    ROOT / "examples/jobpacer/runtime/runtime_adapter.py",
    ROOT / "examples/jobpacer/workloads.py",
    ROOT / "examples/jobpacer/comm_profile.py",
    ROOT / "examples/jobpacer/scripts/run_experiments.py",
    ROOT / "examples/jobpacer/scripts/run_compact_suite.py",
    ROOT / "examples/jobpacer/scripts/run_interleaved_isolated.py",
)


def _git(*args: str) -> str | None:
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def _load(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(resolve_migrated_path(path, MIGRATION_MAP).read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _record_result_path(record: dict[str, Any]) -> Path:
    return resolve_migrated_path(record.get("result_path", ""), MIGRATION_MAP)


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
             old_bare: bool = False, binding_preparation: str | None = None,
             poll_interval: float | None = None) -> list[str]:
    source = ("--dag", str(args.dag)) if args.dag else ("--workload", args.workload)
    poll_interval = getattr(args, "poll_interval", 0.001) if poll_interval is None else poll_interval
    if old:
        command = [sys.executable, str(PHASE1_REPLAY if old_bare else OLD_REPLAY)]
        if not old_bare:
            command.extend(("--mode", "scheduler"))
        command.extend([
            "--policy", policy,
            "--selection", "runtime_arrival", *source,
            "--backend", args.backend, "--world-size", str(args.world_size),
            "--max-outstanding", "1", "--timeout", str(args.timeout),
            "--completion-poll-interval-s", str(poll_interval),
            "--compute-jitter", str(args.compute_jitter), "--epoch", str(epoch),
            "--warmup-iterations", str(args.warmup_iterations),
            "--output", str(output),
        ])
    else:
        command = [
            sys.executable, str(RUNTIME_REPLAY), "--policy", policy,
            *source, "--backend", args.backend,
            "--world-size", str(args.world_size), "--epoch", str(epoch),
            "--compute-jitter", str(args.compute_jitter), "--wait-budget-s", str(args.wait_budget_s),
            "--poll-interval", str(poll_interval), "--timeout", str(args.timeout),
            "--warmup-iterations", str(args.warmup_iterations),
            "--binding-preparation", binding_preparation or
            getattr(args, "binding_preparation", "precreate"),
            "--output", str(output),
        ]
    if args.comm_profile:
        command.extend(("--comm-profile", str(args.comm_profile)))
    if policy == "static_ltf" and args.static_ltf_order:
        command.extend(("--static-order", str(args.static_ltf_order)))
    return command


def _measure_row(record: dict[str, Any], result: dict[str, Any] | None, result_path: Path) -> dict[str, Any]:
    config = record["config"]
    status = (result.get("validation", {}).get("status")
              if result and record.get("returncode") == 0 else "process_failed")
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
        "arm": record.get("arm", f"{record['group']}-{config['policy']}"),
        "group": record["group"],
        "policy": config["policy"],
        "workload": config["workload"],
        "epoch": config["epoch"],
        "repeat": config["repeat"],
        "compute_jitter": config["compute_jitter"],
        "binding_preparation": config.get("binding_preparation"),
        "poll_interval_s": config.get("poll_interval_s", config.get("poll_interval")),
        "status": status,
        "makespan_s": makespan,
        "preparation_total_s": (_rank_max(result, "preparation_total_us") / 1_000_000
                                if result and _rank_max(result, "preparation_total_us") is not None else None),
        "binding_creation_total_s": (_rank_max(result, "binding_creation_total_us") / 1_000_000
                                     if result and _rank_max(result, "binding_creation_total_us") is not None else None),
        "mean_job_jct_s": sum(jcts) / len(jcts) if jcts else None,
        "slowest_job_jct_s": max(jcts, default=None),
        "wall_time_s": record["wall_time_s"],
        "result_path": str(result_path),
        "error": record.get("error"),
    }


def _job_rows(record: dict[str, Any], result: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not result or record.get("returncode") != 0:
        return []
    status = result.get("validation", {}).get("status")
    performance = result.get("performance", {})
    return [{
        "run_id": record["run_id"], "arm": record.get("arm"), "group": record["group"],
        "policy": record["config"]["policy"], "workload": record["config"]["workload"],
        "epoch": record["config"]["epoch"], "repeat": record["config"]["repeat"],
        "binding_preparation": record["config"].get("binding_preparation"),
        "poll_interval_s": record["config"].get("poll_interval_s"),
        "job_id": item["job_id"], "jct_s": item["makespan_us"] / 1_000_000,
        "status": status, "isolated_jct_s": None, "slowdown": None,
    } for item in performance.get("job_makespans", [])]


def _task_timing_rows(record: dict[str, Any], result: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not result or record.get("returncode") != 0:
        return []
    config = record["config"]
    output = []
    for rank in result.get("ranks", []):
        events = {}
        for event in rank.get("runtime_events", []):
            if event.get("task_id"):
                events.setdefault(event["task_id"], {})[event["kind"]] = event
        for job in rank.get("jobs", []):
            for task in job.get("tasks", []):
                task_id = task.get("task_id") or f"{job['job_id']}/comm-{task.get('ordinal', '?')}"
                task_events = events.get(task_id, {})
                def timestamp(event_name: str, *fields: str):
                    event = task_events.get(event_name)
                    if event is not None:
                        return event.get("time_us")
                    return next((task[field] for field in fields if task.get(field) is not None), None)
                ready = timestamp("", "ready_ts", "ready_record_ts")
                submit_call = timestamp("submit_call", "submit_call_ts", "submit_api_start_ts")
                submit_return = timestamp("submit_return", "submit_return_ts", "submit_api_return_ts")
                grant = timestamp("grant_received", "admit_ts")
                call_start = timestamp("collective_call_start", "collective_call_start_ts")
                call_return = timestamp("collective_call_return", "collective_call_return_ts")
                complete = timestamp("completion_observed", "completion_observed_ts")
                wait_start = timestamp("application_wait_start", "first_wait_ts", "application_wait_start_ts")
                wait_return = timestamp("application_wait_return", "consumer_end_ts", "wait_return_ts")
                def elapsed(start, end):
                    return (end - start) if start is not None and end is not None else None
                output.append({
                    "run_id": record["run_id"], "arm": record.get("arm"),
                    "rank": rank.get("rank"), "epoch": config.get("epoch"),
                    "repeat": config.get("repeat"), "job_id": job["job_id"],
                    "task_id": task_id,
                    "binding_preparation": config.get("binding_preparation"),
                    "poll_interval_s": config.get("poll_interval_s"),
                    "producer_ready_to_submit_call_us": elapsed(ready, submit_call),
                    "submit_api_us": elapsed(submit_call, submit_return),
                    "submit_call_to_grant_us": elapsed(submit_call, grant),
                    "grant_to_collective_start_us": elapsed(grant, call_start),
                    "collective_call_us": elapsed(call_start, call_return),
                    "call_return_to_completion_observation_us": elapsed(call_return, complete),
                    "application_wait_us": elapsed(wait_start, wait_return),
                    "binding_create_us": task.get("binding_create_duration_us"),
                    "tensor_create_us": task.get("tensor_create_duration_us"),
                    "completion_probe_count": task_events.get("completion_observed", {}).get(
                        "completion_probe_count"),
                    "correct": task.get("correct"),
                })
    return output


def _load_batch_manifest(jobs_path: Path) -> dict[str, Any]:
    resolved_jobs_path = resolve_migrated_path(jobs_path, MIGRATION_MAP)
    manifest_path = resolved_jobs_path.parent / "manifest.json"
    manifest = _load(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError(f"isolated jobs require batch manifest: {manifest_path}")
    return manifest


def _check_isolated_batch(shared: dict[str, Any], isolated: dict[str, Any],
                          shared_workload: Any) -> str:
    for manifest in (shared, isolated):
        source = resolve_migrated_path(manifest["config"]["workload"], MIGRATION_MAP)
        recorded = manifest.get("workload_digest")
        if not source.is_file() or recorded != hashlib.sha256(source.read_bytes()).hexdigest():
            raise ValueError(f"workload file changed or digest missing: {source}")
    for key in ("profile_digest",):
        if shared.get(key) != isolated.get(key) or shared.get(key) is None:
            raise ValueError(f"isolated {key} differs or is missing")
    shared_config, isolated_config = shared["config"], isolated["config"]
    for key in ("backend", "world_size", "compute_jitter", "wait_budget_s",
                "poll_interval", "warmup_iterations", "repeats"):
        if shared_config.get(key) != isolated_config.get(key):
            raise ValueError(f"isolated {key} differs from shared batch")
    if tuple(shared_config["epochs"]) != tuple(isolated_config["epochs"]):
        raise ValueError("isolated seed blocks differ from shared batch")
    for key in ("cpu_model", "cpu_affinity", "thread_environment", "torch", "platform"):
        if shared["environment"].get(key) != isolated["environment"].get(key):
            raise ValueError(f"isolated environment {key} differs from shared batch")
    if shared.get("source_snapshot", {}).get("digest") != isolated.get("source_snapshot", {}).get("digest"):
        raise ValueError("isolated source snapshot differs from shared batch")
    isolated_workload = load_workload(
        resolve_migrated_path(isolated_config["workload"], MIGRATION_MAP)
    )
    if len(isolated_workload.jobs) != 1:
        raise ValueError("isolated denominator workload must contain exactly one job")
    job = isolated_workload.jobs[0]
    shared_job = next((item for item in shared_workload.jobs if item.job_id == job.job_id), None)
    shared_overrides = {key: value for key, value in
                        (shared_workload.execution_compute_duration_s or {}).items()
                        if key.startswith(job.job_id + "/")}
    if (shared_job != job or shared_workload.seed != isolated_workload.seed
            or shared_overrides != (isolated_workload.execution_compute_duration_s or {})):
        raise ValueError(f"isolated job {job.job_id} has different task or compute samples")
    return job.job_id


def _attach_isolated(job_rows: list[dict[str, Any]], paths: list[Path],
                     shared_manifest: dict[str, Any]) -> None:
    shared_config = shared_manifest["config"]
    if shared_config.get("dag"):
        raise ValueError("isolated denominator matching currently supports linear workloads")
    shared_workload = load_workload(
        resolve_migrated_path(shared_config["workload"], MIGRATION_MAP)
    )
    denominators: dict[tuple[str, str, int, int, str], float] = {}
    for path in paths:
        isolated_job_id = _check_isolated_batch(
            shared_manifest, _load_batch_manifest(path), shared_workload)
        with path.open(newline="") as input_file:
            for row in csv.DictReader(input_file):
                if row.get("job_id") != isolated_job_id:
                    raise ValueError(f"isolated jobs.csv contains unexpected job {row.get('job_id')!r}")
                if row.get("status") != "ok" or not row.get("jct_s"):
                    continue
                key = (row["group"], row["policy"], int(row["epoch"]),
                       int(row["repeat"]), row["job_id"])
                if key in denominators:
                    raise ValueError(f"duplicate isolated denominator for {key}")
                denominators[key] = float(row["jct_s"])
    for row in job_rows:
        key = (row["group"], row["policy"], int(row["epoch"]), int(row["repeat"]), row["job_id"])
        denominator = denominators.get(key)
        if denominator is not None:
            row["isolated_jct_s"] = denominator
            row["slowdown"] = float(row["jct_s"]) / denominator


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _paired_rows(rows: list[dict[str, Any]], *, baseline: str, tie_threshold: float,
                 bootstrap_samples: int, bootstrap_seed: int) -> list[dict[str, Any]]:
    """Median repeats within seed, then compare policies using paired seed blocks."""
    successful = [row for row in rows if row["status"] == "ok" and row["makespan_s"] is not None]
    by_arm_block: dict[tuple[str, int, int], float] = {}
    for row in successful:
        arm = row.get("arm") or f"{row['group']}-{row['policy']}"
        key = (arm, int(row["epoch"]), int(row["repeat"]))
        if key in by_arm_block:
            raise ValueError(f"duplicate performance block {key}")
        by_arm_block[key] = float(row["makespan_s"])
    arms = sorted({arm for arm, _seed, _repeat in by_arm_block})
    all_seeds = sorted({int(row["epoch"]) for row in rows})
    output = []
    for arm in arms:
        if arm == baseline:
            continue
        paired = []
        for seed in all_seeds:
            repeats = sorted({repeat for candidate_arm, candidate_seed, repeat in by_arm_block
                              if candidate_seed == seed and candidate_arm in {baseline, arm}})
            common = [repeat for repeat in repeats
                      if (baseline, seed, repeat) in by_arm_block and (arm, seed, repeat) in by_arm_block]
            if common:
                paired.append((
                    seed,
                    statistics.median(by_arm_block[(baseline, seed, repeat)] for repeat in common),
                    statistics.median(by_arm_block[(arm, seed, repeat)] for repeat in common),
                ))
        ratios = [base / candidate for _seed, base, candidate in paired]
        differences = [candidate - base for _seed, base, candidate in paired]
        bootstrap = []
        if paired and bootstrap_samples:
            rng = random.Random(f"{bootstrap_seed}:{baseline}:{arm}")
            for _ in range(bootstrap_samples):
                sample = [ratios[rng.randrange(len(ratios))] for _index in range(len(ratios))]
                bootstrap.append(statistics.median(sample))
        output.append({
            "baseline": baseline, "candidate": arm, "paired_seeds": len(paired),
            "missing_seed_blocks": len(all_seeds) - len(paired),
            "median_difference_s": statistics.median(differences) if differences else None,
            "median_speedup": statistics.median(ratios) if ratios else None,
            "p10_speedup": _percentile(ratios, 0.1) if ratios else None,
            "p90_speedup": _percentile(ratios, 0.9) if ratios else None,
            "wins": sum(ratio > 1 + tie_threshold for ratio in ratios),
            "ties": sum(abs(ratio - 1) <= tie_threshold for ratio in ratios),
            "losses": sum(ratio < 1 - tie_threshold for ratio in ratios),
            "bootstrap_speedup_low": _percentile(bootstrap, 0.025) if bootstrap else None,
            "bootstrap_speedup_high": _percentile(bootstrap, 0.975) if bootstrap else None,
        })
    return output


def _mechanism_row(record: dict[str, Any], result: dict[str, Any] | None) -> dict[str, Any]:
    decision_records = []
    coordinator_rank: dict[str, Any] | None = None
    if result:
        coordinator_rank = next((rank for rank in result.get("ranks", [])
                                 if rank.get("decision_records")), None)
        decision_records = coordinator_rank.get("decision_records", []) if coordinator_rank else []
    snapshots = [item for item in decision_records if item.get("kind") == "policy_snapshot"]
    decisions = [item for item in decision_records if item.get("kind") == "decision"]
    offers: dict[str, list[float]] = {}
    for item in decision_records:
        if item.get("kind") == "offer":
            offers.setdefault(item["task_id"], []).append(float(item["now"]))
    max_offer_spread = max((max(times) - min(times) for times in offers.values() if len(times) > 1), default=0.0)
    workload_name = Path(str(record["config"]["workload"])).stem
    policy = record["config"]["policy"]
    max_eligible = max((len(item.get("eligible", [])) for item in snapshots), default=0)
    waits = sum(item.get("decision") == "wait" for item in decisions)
    deadlines = sum(item.get("kind") == "lookahead_deadline" for item in decision_records)
    deadline_fallbacks = sum(item.get("decision") == "dispatch"
                             and item.get("reason") == "LOOKAHEAD_DEADLINE_FALLBACK"
                             for item in decisions)
    wait_records = [item for item in decisions if item.get("decision") == "wait"]
    world_size = int(result.get("config", {}).get("world_size", 2)) if result else 2
    offer_by_task: dict[str, dict[int, float]] = {}
    for item in decision_records:
        if item.get("kind") == "offer":
            offer_by_task.setdefault(item["task_id"], {})[int(item["endpoint"])] = float(item["now"])
    ontime_waits = sum(
        set(offer_by_task.get(item.get("target", ""), {})) == set(range(world_size))
        and max(offer_by_task[item["target"]].values()) <= float(item["deadline"])
        for item in wait_records
    )
    head_blocked = sum(float(item.get("duration", 0)) for item in decision_records
                       if item.get("kind") == "idle_interval" and item.get("reason") == "STATIC_HEAD_BLOCKED")
    unsafe_anticipated = False
    if coordinator_rank:
        # Coordinator ``now`` and rank-local DAG event timestamps use the same
        # monotonic clock in the coordinator rank.  G4 is invalid only when
        # unsafe is anticipated before current has physically completed; an
        # anticipation after that completion is a safe frontier declaration.
        current_completion_us = [
            int(event["time_us"])
            for event in coordinator_rank.get("dag_events", [])
            if event.get("kind") == "comm_completed_observed"
            and event.get("task_id") == "job-1/current"
            and event.get("time_us") is not None
        ]
        unsafe_anticipated = any(
            any(entry.get("task_id", "").endswith("/unsafe") for entry in item.get("anticipated", []))
            and (
                not current_completion_us
                or float(item.get("now", 0.0)) * 1_000_000 < min(current_completion_us)
            )
            for item in snapshots
        )
    dispatches = [item.get("task_id", "") for item in decisions if item.get("decision") == "dispatch"]
    gate_id = "job-gate/occupy" if workload_name.startswith("G2") else "job-gate/comm-0"
    research_ids = ({"job-0/comm-a", "job-0/comm-b"} if workload_name.startswith("G2")
                    else {"job-0/comm-0", "job-1/comm-0"})
    gate_dispatched_first = bool(dispatches and dispatches[0] == gate_id)
    gate_completion = max((float(item["now"]) for item in decision_records
                           if item.get("kind") == "completed" and item.get("task_id") == gate_id),
                          default=None)
    # The coordinator records first eligible only after capacity is released.
    # For these first-in-group candidates, complete member OFFER coverage while
    # gate is inflight reconstructs readiness before the gate completes.
    member_offers = {task_id: {} for task_id in research_ids}
    for item in decision_records:
        if item.get("kind") == "offer" and item.get("task_id") in research_ids:
            member_offers[item["task_id"]][item.get("endpoint")] = float(item["now"])
    candidates_ready_before_gate_completion = (gate_completion is not None
        and all(set(member_offers[task_id]) == set(range(world_size))
                and max(member_offers[task_id].values()) < gate_completion
                for task_id in research_ids))
    first_research_decision = next((item for item in decisions
        if item.get("decision") == "dispatch" and item.get("task_id") in research_ids), None)
    candidates_compete_at_choice = bool(first_research_decision and research_ids.issubset(
        set(first_research_decision.get("eligible", []))))
    triggered: bool | None = None
    criterion = "validation only"
    if workload_name.startswith(("L2", "G2")):
        criterion = "gate first; both named candidates eligible before gate completion and at first research choice"
        triggered = (gate_dispatched_first and candidates_ready_before_gate_completion
                     and candidates_compete_at_choice)
    elif workload_name.startswith("L1") and policy.startswith("static"):
        criterion, triggered = "static head blocked while another task can progress", head_blocked > 0
    elif workload_name.startswith("L3") and policy == "lookahead":
        criterion, triggered = "lookahead wait target offered by all members by deadline", ontime_waits > 0
    elif workload_name.startswith("L4") and policy == "lookahead":
        criterion = "lookahead wait reached fixed deadline and dispatched by fallback"
        triggered = waits > 0 and deadlines > 0 and deadline_fallbacks > 0
    elif workload_name.startswith("L5"):
        target_offers = offer_by_task.get("job-0/comm-0", {})
        criterion = "job-0/comm-0 rank 1 OFFER follows rank 0 by at least 5 ms"
        triggered = 0 in target_offers and 1 in target_offers and target_offers[1] - target_offers[0] >= 0.005
    elif workload_name.startswith("G4") and policy == "lookahead":
        criterion = "lookahead waits for target without anticipating unsafe frontier"
        triggered = waits > 0 and all(item.get("target") == "job-0/target" for item in wait_records) and not unsafe_anticipated
    elif workload_name.startswith("G1"):
        criterion = "diamond validates with no competing eligible communication"
        triggered = bool(result and result.get("validation", {}).get("status") == "ok" and max_eligible <= 1)
    elif workload_name.startswith(("L0", "G0", "G3")):
        triggered = bool(result and result.get("validation", {}).get("status") == "ok")
    status = ("process_failed" if "returncode" in record and record["returncode"] != 0
              else result.get("validation", {}).get("status") if result else "failed")
    return {
        "run_id": record["run_id"], "status": status,
        "scenario": workload_name, "criterion": criterion, "mechanism_triggered": triggered,
        "max_simultaneous_eligible": max_eligible,
        "lookahead_waits": waits,
        "lookahead_ontime_waits": ontime_waits,
        "lookahead_deadlines": deadlines,
        "lookahead_deadline_fallbacks": deadline_fallbacks,
        "static_head_blocked_s": head_blocked,
        "max_member_offer_spread_s": max_offer_spread,
        "unsafe_frontier_anticipated": unsafe_anticipated,
        "gate_dispatched_first": gate_dispatched_first,
        "research_candidates_ready_before_gate_completion": candidates_ready_before_gate_completion,
        "research_candidates_compete_at_choice": candidates_compete_at_choice,
    }


def _cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def _rank_max(result: dict[str, Any], field: str) -> float | None:
    values = [rank[field] for rank in result.get("ranks", [])
              if isinstance(rank.get(field), (int, float))]
    return max(values) if values else None


def _selected_arms(args: argparse.Namespace, parser: argparse.ArgumentParser) -> tuple[str, ...]:
    if args.arms:
        if args.include_old:
            parser.error("--arms and --include-old cannot be combined")
        names = tuple(part.strip() for part in args.arms.split(","))
    else:
        names = tuple(ARM_SPECS) if args.include_old else tuple(
            name for name in ARM_SPECS if name.startswith("new-"))
    if not names or any(name not in ARM_SPECS for name in names) or len(names) != len(set(names)):
        parser.error("--arms must be unique comma-separated names from: " + ", ".join(ARM_SPECS))
    if args.dag and any(name.startswith("old-") for name in names):
        parser.error("old Phase 1/2 arms require a linear workload")
    if args.baseline not in names or any(name not in names for name in args.secondary_baseline):
        parser.error("baseline and secondary baselines must be selected arms")
    return names


def _plan(epochs: tuple[int, ...], repeats: int, arms: tuple[str, ...], order_seed: int,
          binding_preparation: str = "precreate", poll_interval: float = 0.001) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for epoch in epochs:
        for repeat in range(repeats):
            ordered = list(arms)
            random.Random(order_seed + epoch * 1_000_003 + repeat).shuffle(ordered)
            for arm in ordered:
                group, policy, old, old_bare = ARM_SPECS[arm]
                effective_binding = ("legacy_tensor_precreated" if old else
                                     ARM_BINDING_PREPARATION.get(arm, binding_preparation))
                effective_poll = ARM_POLL_INTERVAL.get(arm, poll_interval)
                plan.append({"run_id": f"{len(plan) + 1:04d}-{arm}-e{epoch}-r{repeat}",
                             "arm": arm, "group": group, "policy": policy,
                             "old": old, "old_bare": old_bare,
                             "binding_preparation": effective_binding,
                             "poll_interval": effective_poll,
                             "epoch": epoch, "repeat": repeat, "block_order": ordered})
    return plan


def _historical_walls(paths: list[Path], scenario: str, arms: tuple[str, ...]) -> dict[str, float]:
    samples: dict[str, list[float]] = {arm: [] for arm in arms}
    for path in paths:
        csv_paths = ([path / "summary.csv"] if (path / "summary.csv").exists()
                     else sorted(path.rglob("summary.csv"))) if path.is_dir() else [path]
        for csv_path in csv_paths:
            with csv_path.open(newline="") as input_file:
                for row in csv.DictReader(input_file):
                    arm = row.get("arm") or f"{row.get('group')}-{row.get('policy')}"
                    targets = ([arm] if arm in samples else
                               [candidate for candidate in samples
                                if arm == "new-ltf" and candidate.startswith("new-ltf-")])
                    if (targets and Path(row.get("workload", "")).stem == scenario
                            and row.get("wall_time_s")):
                        for target in targets:
                            samples[target].append(float(row["wall_time_s"]))
    return {arm: statistics.median(values) for arm, values in samples.items() if values}


def _latest_records(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return latest
    with path.open() as input_file:
        for line in input_file:
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid runs.jsonl record: {exc}") from exc
            latest[record["run_id"]] = record
    return latest


def _successful_record(record: dict[str, Any] | None, planned: dict[str, Any]) -> bool:
    if not record or record.get("returncode") != 0:
        return False
    config = record.get("config", {})
    if (record.get("run_id") != planned["run_id"]
            or (planned.get("arm") is not None and record.get("arm") != planned["arm"])
            or record.get("group") != planned["group"]
            or config.get("policy") != planned["policy"] or config.get("epoch") != planned["epoch"]
            or config.get("repeat") != planned["repeat"]
            or (planned.get("binding_preparation") is not None
                and config.get("binding_preparation") != planned["binding_preparation"])
            or (planned.get("poll_interval") is not None
                and config.get("poll_interval_s") != planned["poll_interval"])):
        return False
    result_path = _record_result_path(record)
    if (not result_path.is_file() or not record.get("result_sha256")
            or hashlib.sha256(result_path.read_bytes()).hexdigest() != record["result_sha256"]):
        return False
    result = _load(result_path)
    result_config = result.get("config", {}) if result else {}
    try:
        jitter_matches = float(result_config.get("compute_jitter", -1)) == float(config.get("compute_jitter", -2))
    except (TypeError, ValueError):
        jitter_matches = False
    return bool(result and result.get("validation", {}).get("status") == "ok"
                and result_config.get("policy") == planned["policy"]
                and result_config.get("epoch") == planned["epoch"]
                and (planned.get("old", False)
                     or planned.get("binding_preparation") is None
                     or result_config.get("binding_preparation") == planned["binding_preparation"])
                and jitter_matches)


def _attempt_paths(raw_dir: Path, run_id: str) -> tuple[int, Path, Path, Path]:
    attempt = 1
    while True:
        stem = run_id if attempt == 1 else f"{run_id}.attempt-{attempt}"
        paths = tuple(raw_dir / f"{stem}.{suffix}" for suffix in ("json", "stdout", "stderr"))
        if not any(path.exists() for path in paths):
            return (attempt, *paths)
        attempt += 1


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as output_file:
        if fields:
            writer = csv.DictWriter(output_file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    temporary.replace(path)


def _mechanism_summary(rows: list[dict[str, Any]], records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row, record in zip(rows, records):
        grouped.setdefault(record["arm"], []).append(row)
    return [{
        "arm": arm, "runs": len(items),
        "ok": sum(item["status"] == "ok" for item in items),
        "mechanism_triggered": sum(item["mechanism_triggered"] is True for item in items),
        "gate_first": sum(item["gate_dispatched_first"] is True for item in items),
        "research_ready_before_gate_completion": sum(
            item["research_candidates_ready_before_gate_completion"] is True for item in items),
        "research_competition_at_choice": sum(
            item["research_candidates_compete_at_choice"] is True for item in items),
        "lookahead_ontime_runs": sum(item["lookahead_ontime_waits"] > 0 for item in items),
    } for arm, items in sorted(grouped.items())]


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
    parser.add_argument("--binding-preparation", choices=("precreate", "on-ready"), default="precreate")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--comm-profile", type=Path)
    parser.add_argument("--static-ltf-order", type=Path,
                        help="explicit DAG order for the static_ltf arm; static_fifo remains unchanged")
    parser.add_argument("--include-old", action="store_true")
    parser.add_argument("--arms", help="comma-separated arm names; overrides the legacy five/eight-arm default")
    parser.add_argument("--preview", action="store_true", help="show count and estimated wall time without writing")
    parser.add_argument("--resume", action="store_true", help="continue an identical batch in its output directory")
    parser.add_argument("--history-summary", type=Path, action="append", default=[],
                        help="prior summary.csv or batch directory for wall-time estimates")
    parser.add_argument("--order-seed", type=int, default=0,
                        help="seed for deterministic strategy order within each seed/repeat block")
    parser.add_argument("--baseline", default="new-static_fifo")
    parser.add_argument("--secondary-baseline", action="append", default=[],
                        help="additional arm for paired comparison within the same batch")
    parser.add_argument("--tie-threshold", type=float, default=0.01)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--isolated-jobs", type=Path, action="append", default=[],
                        help="jobs.csv from a mode-matched isolated batch; may be repeated")
    args = parser.parse_args(argv)
    if args.repeats <= 0 or args.world_size < 2 or args.compute_jitter < 0 or args.compute_jitter >= 1:
        parser.error("repeats must be positive, world-size >= 2, and compute-jitter in [0, 1)")
    if args.wait_budget_s < 0 or args.poll_interval <= 0 or args.timeout <= 0 or args.warmup_iterations < 0:
        parser.error("wait-budget-s must be non-negative; poll interval and timeout must be positive")
    if args.tie_threshold < 0 or args.bootstrap_samples < 0:
        parser.error("tie-threshold and bootstrap-samples must be non-negative")
    if args.static_ltf_order and not args.dag:
        parser.error("--static-ltf-order requires --dag")
    if not args.workload and not args.dag:
        args.workload = "balanced"
    source_path = args.dag or Path(args.workload)
    if args.dag:
        args.dag = resolve_migrated_path(repository_path(args.dag), MIGRATION_MAP)
        source_path = args.dag
    elif source_path.exists() or source_path.suffix.lower() == ".json":
        source_path = resolve_migrated_path(repository_path(source_path), MIGRATION_MAP)
        args.workload = str(source_path)
    if args.static_ltf_order:
        args.static_ltf_order = resolve_migrated_path(repository_path(args.static_ltf_order), MIGRATION_MAP)
    if args.comm_profile:
        args.comm_profile = resolve_migrated_path(repository_path(args.comm_profile), MIGRATION_MAP)
    args.output_dir = repository_path(args.output_dir)
    args.history_summary = [repository_path(path) for path in args.history_summary]
    if (source_path.exists() and is_formal_experiment_input(source_path, FORMAL_INPUT_ROOT)
            and not args.comm_profile):
        parser.error("formal Phase 3 experiment inputs require --comm-profile")
    if args.comm_profile:
        args.comm_profile = args.comm_profile.resolve()
    if args.static_ltf_order:
        from examples.jobpacer.runtime.runtime_adapter import load_dag, load_static_order
        try:
            args.static_ltf_order = args.static_ltf_order.resolve(strict=True)
            load_static_order(args.static_ltf_order, load_dag(args.dag, world_size=args.world_size).graph)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
    args.isolated_jobs = [resolve_migrated_path(repository_path(path), MIGRATION_MAP)
                          for path in args.isolated_jobs]
    epochs = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
    if not epochs:
        parser.error("--seeds must contain at least one integer")
    if len(set(epochs)) != len(epochs):
        parser.error("--seeds must not contain duplicates")
    arms = _selected_arms(args, parser)
    plan = _plan(epochs, args.repeats, arms, args.order_seed,
                 args.binding_preparation, args.poll_interval)

    output_dir = args.output_dir.resolve()
    raw_dir = output_dir / "raw"
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    config_snapshot = {
        "workload": str(source_path.resolve()) if not args.dag and source_path.is_file() else args.workload,
        "dag": str(args.dag.resolve()) if args.dag else None,
        "backend": args.backend, "world_size": args.world_size,
        "epochs": epochs, "repeats": args.repeats, "compute_jitter": args.compute_jitter,
        "wait_budget_s": args.wait_budget_s, "poll_interval": args.poll_interval,
        "binding_preparation": args.binding_preparation,
        "order_seed": args.order_seed,
        "arms": arms,
        "arm_configuration": {item["arm"]: {
            "binding_preparation": item["binding_preparation"],
            "poll_interval_s": item["poll_interval"],
        } for item in plan[:len(arms)]},
        "arm_configuration": {item["arm"]: {
            "binding_preparation": item["binding_preparation"],
            "poll_interval_s": item["poll_interval"],
        } for item in plan[:len(arms)]},
        "baseline": args.baseline, "secondary_baselines": args.secondary_baseline,
        "timeout": args.timeout, "comm_profile": str(args.comm_profile) if args.comm_profile else None,
        "warmup_iterations": args.warmup_iterations,
        "static_ltf_order": str(args.static_ltf_order) if args.static_ltf_order else None,
        "isolated_jobs": [str(path) for path in args.isolated_jobs],
    }
    manifest = {
        "schema_version": 5,
        "batch_id": output_dir.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "cpu_model": _cpu_model(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "thread_environment": {key: os.environ.get(key) for key in
                                   ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                    "NUMEXPR_NUM_THREADS", "GLOO_SOCKET_IFNAME")},
            "torch": __import__("torch").__version__,
        },
        "git_head": _git("rev-parse", "HEAD"),
        "working_tree_dirty": bool(_git("status", "--porcelain")),
        "config": config_snapshot,
        "profile_digest": hashlib.sha256(args.comm_profile.read_bytes()).hexdigest()
        if args.comm_profile else None,
        "workload_digest": hashlib.sha256(source_path.read_bytes()).hexdigest()
        if source_path.is_file() else None,
        "static_order_digest": hashlib.sha256(args.static_ltf_order.read_bytes()).hexdigest()
        if args.static_ltf_order else None,
        "source_snapshot": _source_snapshot(),
    }
    existing_manifest = _load(output_dir / "manifest.json") if output_dir.exists() else None
    if output_dir.exists() and not args.resume and not args.preview:
        parser.error(f"output directory already exists; use --resume or a fresh path: {output_dir}")
    if args.resume and output_dir.exists():
        if not existing_manifest or existing_manifest.get("schema_version") != 5:
            parser.error("resume requires a schema-5 batch manifest")
        for key in ("config", "profile_digest", "workload_digest", "static_order_digest",
                    "environment", "git_head", "source_snapshot"):
            if json.dumps(existing_manifest.get(key), sort_keys=True) != json.dumps(manifest.get(key), sort_keys=True):
                parser.error(f"resume rejected: {key} changed")
        if not raw_dir.is_dir():
            parser.error("resume rejected: raw result directory is missing")
        saved_config = _load(output_dir / "inputs" / "batch-config.json")
        if json.dumps(saved_config, sort_keys=True) != json.dumps(config_snapshot, sort_keys=True):
            parser.error("resume rejected: saved batch configuration changed")
        snapshots = [
            ("comm-profile.json", manifest["profile_digest"]) if args.comm_profile else None,
            (source_path.name, manifest["workload_digest"]) if source_path.is_file() else None,
            (args.static_ltf_order.name, manifest["static_order_digest"]) if args.static_ltf_order else None,
        ]
        for snapshot in snapshots:
            if snapshot is None:
                continue
            saved_path = output_dir / "inputs" / snapshot[0]
            if not saved_path.is_file() or hashlib.sha256(saved_path.read_bytes()).hexdigest() != snapshot[1]:
                parser.error(f"resume rejected: saved input {snapshot[0]} changed")
    elif args.resume and not args.preview:
        parser.error(f"resume requires an existing batch directory: {output_dir}")
    previous = _latest_records(output_dir / "runs.jsonl") if args.resume and output_dir.exists() else {}
    planned_ids = {item["run_id"] for item in plan}
    if set(previous) - planned_ids:
        parser.error("resume rejected: runs.jsonl contains unplanned run IDs")
    complete = sum(_successful_record(previous.get(item["run_id"]), item) for item in plan)
    if args.preview:
        try:
            history_sources = args.history_summary or ([output_dir] if output_dir.exists() else [])
            history = _historical_walls(history_sources, source_path.stem, arms)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        known_total = sum(history.get(item["arm"], 0.0) for item in plan)
        known_runs = sum(item["arm"] in history for item in plan)
        remaining_plan = [item for item in plan if not _successful_record(previous.get(item["run_id"]), item)]
        print(json.dumps({"scenario": source_path.stem, "arms": arms, "seeds": len(epochs),
                          "repeats": args.repeats, "planned": len(plan), "complete": complete,
                          "remaining": len(plan) - complete,
                          "estimated_wall_time_s": known_total if known_runs == len(plan) else None,
                          "remaining_estimated_wall_time_s": sum(history.get(item["arm"], 0.0)
                                                                 for item in remaining_plan)
                          if all(item["arm"] in history for item in remaining_plan) else None,
                          "history_coverage": f"{known_runs}/{len(plan)}"}, sort_keys=True))
        return 0
    if not output_dir.exists():
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(exist_ok=False)
        raw_dir.mkdir()
        (output_dir / "inputs").mkdir()
        tables_dir.mkdir()
        figures_dir.mkdir()
        (output_dir / "inputs" / "batch-config.json").write_text(json.dumps(config_snapshot, indent=2) + "\n")
        if args.comm_profile:
            (output_dir / "inputs" / "comm-profile.json").write_bytes(args.comm_profile.read_bytes())
        if source_path.is_file():
            (output_dir / "inputs" / source_path.name).write_bytes(source_path.read_bytes())
        if args.static_ltf_order:
            (output_dir / "inputs" / args.static_ltf_order.name).write_bytes(args.static_ltf_order.read_bytes())
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    else:
        manifest = existing_manifest
    if args.isolated_jobs:
        if args.dag:
            parser.error("isolated denominators currently support linear workloads only")
        shared_workload = load_workload(args.workload)
        for path in args.isolated_jobs:
            _check_isolated_batch(manifest, _load_batch_manifest(path), shared_workload)

    runs_path = output_dir / "runs.jsonl"
    latest = dict(previous)
    with runs_path.open("a") as runs_file:
        for item in plan:
            run_id = item["run_id"]
            if _successful_record(latest.get(run_id), item):
                continue
            attempt, result_path, stdout_path, stderr_path = _attempt_paths(raw_dir, run_id)
            command = _command(args, item["policy"], item["epoch"], result_path,
                               old=item["old"], old_bare=item["old_bare"],
                               binding_preparation=item["binding_preparation"],
                               poll_interval=item["poll_interval"])
            started = time.time()
            started_at = datetime.fromtimestamp(started, timezone.utc).isoformat()
            try:
                completed = subprocess.run(
                    command, cwd=ROOT,
                    env={**os.environ, "PYTHONPATH": os.pathsep.join(
                        (str(ROOT / "src"), str(ROOT), os.environ.get("PYTHONPATH", "")))},
                    capture_output=True, text=True, timeout=args.timeout + 10,
                )
                error = None if completed.returncode == 0 else completed.stderr[-4000:]
                stdout_path.write_text(completed.stdout)
                stderr_path.write_text(completed.stderr)
            except subprocess.TimeoutExpired as exc:
                completed = None
                error = f"timeout: {exc}"
                stdout_path.write_text(str(exc.stdout or ""))
                stderr_path.write_text(str(exc.stderr or ""))
            record = {
                "run_id": run_id, "group": item["group"], "arm": item["arm"],
                "attempt": attempt, "retry_of": latest.get(run_id, {}).get("result_path"),
                "config": {"policy": item["policy"], "arm": item["arm"],
                           "binding_preparation": item["binding_preparation"],
                           "poll_interval_s": item["poll_interval"],
                           "workload": args.workload or str(args.dag),
                           "epoch": item["epoch"], "repeat": item["repeat"],
                           "compute_jitter": args.compute_jitter, "order_seed": args.order_seed,
                           "block_order": item["block_order"]},
                "command": command, "started_at": started_at,
                "ended_at": datetime.now(timezone.utc).isoformat(),
                "wall_time_s": time.time() - started,
                "returncode": completed.returncode if completed is not None else None,
                "result_path": str(result_path),
                "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest()
                if result_path.is_file() else None,
                "error": error,
            }
            runs_file.write(json.dumps(record, sort_keys=True) + "\n")
            runs_file.flush()
            os.fsync(runs_file.fileno())
            latest[run_id] = record

    records = [latest[item["run_id"]] for item in plan]
    summary_rows = [_measure_row(record, _load(_record_result_path(record)),
                                 _record_result_path(record)) for record in records]
    job_rows = [row for record in records
                for row in _job_rows(record, _load(_record_result_path(record)))]
    task_rows = [row for record in records
                 for row in _task_timing_rows(record, _load(_record_result_path(record)))]
    mechanism_rows = [_mechanism_row(record, _load(_record_result_path(record))) for record in records]
    mechanism_summary = _mechanism_summary(mechanism_rows, records)

    _write_csv(output_dir / "summary.csv", summary_rows)
    for filename, table in (("jobs.csv", job_rows), ("task-timings.csv", task_rows),
                            ("mechanisms.csv", mechanism_rows)):
        if filename == "jobs.csv" and args.isolated_jobs:
            _attach_isolated(job_rows, args.isolated_jobs, manifest)
        _write_csv(output_dir / filename, table)
        _write_csv(tables_dir / filename, table)
    _write_csv(tables_dir / "summary.csv", summary_rows)
    _write_csv(tables_dir / "mechanism-summary.csv", mechanism_summary)
    _write_csv(output_dir / "mechanism-summary.csv", mechanism_summary)
    paired = []
    for baseline in dict.fromkeys((args.baseline, *args.secondary_baseline)):
        paired.extend(_paired_rows(summary_rows, baseline=baseline, tie_threshold=args.tie_threshold,
                                   bootstrap_samples=args.bootstrap_samples, bootstrap_seed=args.order_seed))
    _write_csv(output_dir / "paired-summary.csv", paired)
    (output_dir / "analysis.json").write_text(json.dumps({
        "unit": "seed-block after within-seed repeat median",
        "baselines": list(dict.fromkeys((args.baseline, *args.secondary_baseline))),
        "compute_jitter": args.compute_jitter,
        "interpretation": "fixed-input system-noise control" if args.compute_jitter == 0
                          else "compute-duration perturbation across seed blocks",
        "tie_threshold": args.tie_threshold,
        "bootstrap_samples": args.bootstrap_samples, "comparisons": paired,
        "mechanism_summary": mechanism_summary,
        "failed_runs": sum(row["status"] != "ok" for row in summary_rows),
    }, indent=2, sort_keys=True) + "\n")
    (tables_dir / "paired-summary.csv").write_bytes((output_dir / "paired-summary.csv").read_bytes())
    (tables_dir / "analysis.json").write_bytes((output_dir / "analysis.json").read_bytes())
    failed = sum(row["status"] != "ok" for row in summary_rows)
    print(json.dumps({"output_dir": str(output_dir), "runs": len(records), "failed": failed}, sort_keys=True))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
