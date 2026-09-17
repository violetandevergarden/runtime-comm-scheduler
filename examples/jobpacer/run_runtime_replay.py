"""Launch the independent Stage 3.1 runtime replay."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
WORKER = HERE / "runtime_worker.py"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _start(rank: int, args: argparse.Namespace, rendezvous_port: int, control_port: int) -> subprocess.Popen[str]:
    env = dict(os.environ)
    env.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(rendezvous_port), RANK=str(rank), WORLD_SIZE=str(args.world_size), LOCAL_RANK="0")
    env["PYTHONPATH"] = str(HERE.parents[1] / "src") + os.pathsep + env.get("PYTHONPATH", "")
    if args.backend == "nccl":
        env["CUDA_VISIBLE_DEVICES"] = str(rank)
    return subprocess.Popen(
        [sys.executable, str(WORKER), "--policy", args.policy, "--workload", args.workload, "--backend", args.backend, "--timeout", str(args.timeout), "--control-port", str(control_port), "--fault", args.fault],
        cwd=str(HERE.parents[1]), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )


def _collect(rank: int, process: subprocess.Popen[str], timeout: float) -> tuple[dict[str, Any] | None, str | None]:
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        return None, f"rank {rank} timed out: {stderr[-2000:]}"
    if process.returncode != 0:
        return None, f"rank {rank} exited {process.returncode}: {stderr[-3000:]} {stdout[-1000:]}"
    try:
        return json.loads(stdout.strip().splitlines()[-1]), None
    except (IndexError, json.JSONDecodeError) as exc:
        return None, f"rank {rank} emitted invalid JSON ({exc}): {stdout[-2000:]}"


def _validate_results(results: list[dict[str, Any]], world_size: int) -> dict[str, Any]:
    all_correct = all(
        task["correct"]
        for result in results
        for job in result.get("jobs", [])
        for task in job.get("tasks", [])
    )
    errors: list[str] = []
    rank_task_sequences = []
    for result in results:
        grants = tuple(result.get("grant_sequence", ()))
        launches = tuple(result.get("launch_sequence", ()))
        rank_task_sequences.append(grants)
        if grants != launches:
            errors.append(f"rank {result.get('rank')} grant/launch projection mismatch")
        for job in result.get("jobs", []):
            sequence = [task["group_seq"] for task in job.get("tasks", [])]
            if sequence != sorted(sequence):
                errors.append(f"rank {result.get('rank')} group {job.get('job_id')} sequence regressed")
    group_sequences: dict[str, list[tuple[str, ...]]] = {}
    for result in results:
        by_task = {
            task["task_id"]: task
            for job in result.get("jobs", [])
            for task in job.get("tasks", [])
        }
        per_group: dict[str, list[str]] = {}
        for task_id in result.get("launch_sequence", ()):
            task = by_task.get(task_id)
            if task is not None:
                per_group.setdefault(task["group_id"], []).append(task_id)
        for group_id, sequence in per_group.items():
            group_sequences.setdefault(group_id, []).append(tuple(sequence))
    for group_id, sequences in group_sequences.items():
        if len(set(sequences)) != 1:
            errors.append(f"group {group_id} launch projection mismatch")
    return {
        "status": "ok" if len(results) == world_size and all_correct and not errors else "failed",
        "all_collectives_correct": all_correct,
        "rank_count": len(results),
        "errors": errors,
        "rank_task_sequences": rank_task_sequences,
    }


def _metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {}
    coordinator_records = next(
        (result.get("decision_records", []) for result in results if result.get("decision_records")),
        [],
    )
    dispatch_times = {
        item["task_id"]: item["now"]
        for item in coordinator_records
        if item.get("kind") == "decision" and item.get("decision") == "dispatch"
    }
    eligible_times = {
        item["task_id"]: item["now"]
        for item in coordinator_records
        if item.get("kind") == "eligible"
    }
    progress: dict[str, dict[str, list[float]]] = {}
    for item in coordinator_records:
        if item.get("kind") not in {"submitted", "completed"}:
            continue
        task = progress.setdefault(item["task_id"], {"submitted": [], "completed": []})
        task[item["kind"]].append(item["now"])
    coordinator_tasks = {}
    for task_id, grant_time in dispatch_times.items():
        submitted = progress.get(task_id, {}).get("submitted", [])
        completed = progress.get(task_id, {}).get("completed", [])
        if submitted and completed:
            coordinator_tasks[task_id] = {
                "eligible_to_grant_s": (
                    grant_time - eligible_times[task_id]
                    if task_id in eligible_times
                    else None
                ),
                "grant_to_all_submitted_s": max(submitted) - grant_time,
                "all_submitted_to_all_completed_s": max(completed) - max(submitted),
            }
    job_durations = {}
    for result in results:
        for job in result.get("jobs", []):
            if "job_start_ts" in job and "job_end_ts" in job:
                job_durations.setdefault(job["job_id"], []).append(
                    (job["job_end_ts"] - job["job_start_ts"]) / 1_000_000.0
                )
    finished = [item["now"] for item in coordinator_records if item.get("kind") == "finished" and "now" in item]
    started = [item["now"] for item in coordinator_records if "now" in item]
    idle_by_reason: dict[str, float] = {}
    for item in coordinator_records:
        if item.get("kind") == "idle_interval":
            idle_by_reason[item["reason"]] = idle_by_reason.get(item["reason"], 0.0) + item["duration"]
    local_waits: dict[str, dict[str, dict[str, float]]] = {}
    for result in results:
        event_times: dict[str, dict[str, int]] = {}
        for event in result.get("runtime_events", []):
            if event.get("task_id"):
                event_times.setdefault(event["task_id"], {})[event["kind"]] = event["time_us"]
        for job in result.get("jobs", []):
            for task in job.get("tasks", []):
                times = event_times.get(task["task_id"], {})
                if "grant_received" in times and "submit_call_ts" in task:
                    local_waits.setdefault(str(result.get("rank")), {})[task["task_id"]] = {
                        "submit_call_to_grant_s": (times["grant_received"] - task["submit_call_ts"]) / 1_000_000.0,
                        "grant_to_launch_s": (times.get("launch_start", times["grant_received"]) - times["grant_received"]) / 1_000_000.0,
                        "consumer_wait_s": (task["consumer_end_ts"] - task["first_wait_ts"]) / 1_000_000.0,
                    }
    return {
        "coordinator_task_timings": coordinator_tasks,
        "coordinator_idle_s": idle_by_reason,
        "coordinator_makespan_s": max(finished) - min(started) if finished and started else None,
        "job_duration_s": {job: max(values) for job, values in job_durations.items()},
        "rank_task_timings": local_waits,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("static_fifo", "static_ltf", "fifo", "ltf", "lookahead"), default="fifo")
    parser.add_argument("--workload", default="balanced")
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--fault",
        choices=("none", "missing_task", "metadata_mismatch", "launch_failure", "completion_probe_failure"),
        default="none",
    )
    args = parser.parse_args()
    rendezvous_port, control_port = _free_port(), _free_port()
    processes = [_start(rank, args, rendezvous_port, control_port) for rank in range(args.world_size)]
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(processes)) as pool:
        futures = [pool.submit(_collect, rank, process, args.timeout) for rank, process in enumerate(processes)]
        for future in futures:
            result, error = future.result()
            if error:
                errors.append(error)
            elif result is not None:
                results.append(result)
    results.sort(key=lambda item: item.get("rank", -1))
    validation = _validate_results(results, args.world_size)
    validation["errors"] = errors + validation["errors"]
    if errors:
        validation["status"] = "failed"
    config = vars(args).copy()
    config["output"] = str(args.output) if args.output else None
    payload = {"config": config, "validation": validation, "metrics": _metrics(results), "ranks": results}
    if args.output:
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if validation["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
