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

from runtime_comm_scheduler.dag import build_static_order, load_dag, load_static_order


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
    command = [sys.executable, str(WORKER), "--policy", args.policy, "--backend", args.backend,
               "--epoch", str(args.epoch), "--timeout", str(args.timeout),
               "--poll-interval", str(args.poll_interval), "--dag-poll-interval", str(args.dag_poll_interval),
               "--compute-jitter", str(args.compute_jitter), "--control-port", str(control_port),
               "--fault", args.fault]
    if args.dag:
        command.extend(("--dag", str(args.dag)))
    else:
        command.extend(("--workload", args.workload or "balanced"))
    if args.static_order:
        command.extend(("--static-order", str(args.static_order)))
    return subprocess.Popen(
        command,
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


def _validate_results(results: list[dict[str, Any]], world_size: int, *,
                       expected_tasks: dict[str, dict[str, Any]],
                       expected_nodes: dict[int, set[str]] | None = None,
                       digests: set[str] | None = None,
                       expected_config: dict[str, Any] | None = None) -> dict[str, Any]:
    errors: list[str] = []
    rank_task_sequences = []
    by_rank = {int(result.get("rank", -1)): result for result in results}
    actual_task_ids: set[str] = set()
    all_correct = True
    for result in results:
        rank = int(result.get("rank", -1))
        grants = tuple(result.get("grant_sequence", ()))
        launches = tuple(result.get("launch_sequence", ()))
        rank_task_sequences.append(grants)
        if grants != launches:
            errors.append(f"rank {rank} grant/launch projection mismatch")
        if result.get("status") != "ok":
            errors.append(f"rank {rank} status is {result.get('status')!r}")
        if expected_config is not None:
            for key, value in expected_config.items():
                if result.get(key) != value:
                    errors.append(f"rank {rank} run config mismatch for {key}: expected {value}, got {result.get(key)}")
        if digests is not None and result.get("manifest_digest") not in digests:
            errors.append(f"rank {rank} manifest digest mismatch")
        if digests is not None:
            actual_groups = tuple(sorted((item["group_id"], tuple(item["ranks"]))
                                         for item in result.get("groups", ())))
            if actual_groups != expected_tasks.get("__group_defs__"):
                errors.append(f"rank {rank} group manifest mismatch")
        task_list = [task for job in result.get("jobs", ()) for task in job.get("tasks", ())]
        task_ids = [task.get("task_id") for task in task_list]
        if len(task_ids) != len(set(task_ids)):
            errors.append(f"rank {rank} has duplicate task results")
        expected = set(expected_tasks.get(str(rank), {}))
        if set(task_ids) != expected or len(task_ids) != len(expected):
            errors.append(f"rank {rank} task set mismatch: missing={sorted(expected - set(task_ids))}, extra={sorted(set(task_ids) - expected)}")
        if set(result.get("expected_task_ids", task_ids)) != expected:
            errors.append(f"rank {rank} reported expected task set mismatch")
        for task in task_list:
            task_id = task.get("task_id")
            if task_id in expected_tasks.get(str(rank), {}):
                expected_meta = expected_tasks[str(rank)][task_id]
                if task.get("group_id") != expected_meta["group_id"] or task.get("group_seq") != expected_meta["group_seq"]:
                    errors.append(f"rank {rank} task metadata mismatch for {task_id}")
            all_correct &= task.get("correct") is True
            actual_task_ids.add(task_id)
        for job in result.get("jobs", ()):
            if job.get("status") != "ok":
                errors.append(f"rank {rank} job {job.get('job_id')} status is {job.get('status')!r}")
        if expected_nodes is not None:
            completed = [node for job in result.get("jobs", ()) for node in job.get("completed_node_ids", ())]
            expected_node_ids = expected_nodes.get(rank, set())
            if len(completed) != len(set(completed)) or set(completed) != expected_node_ids:
                errors.append(f"rank {rank} completed DAG node set mismatch")

    if set(by_rank) != set(range(world_size)):
        errors.append(f"rank set mismatch: expected={list(range(world_size))}, actual={sorted(by_rank)}")

    group_sequences: dict[str, list[str]] = {}
    for task_id in expected_tasks.get("__all__", {}):
        meta = expected_tasks["__all__"][task_id]
        group_sequences.setdefault(meta["group_id"], []).append(task_id)
    for group_id, task_ids in group_sequences.items():
        expected = tuple(sorted(task_ids, key=lambda task_id: expected_tasks["__all__"][task_id]["group_seq"]))
        members = expected_tasks["__groups__"][group_id]
        for rank in members:
            result = by_rank.get(rank)
            if result is None:
                continue
            actual = tuple(task_id for task_id in result.get("launch_sequence", ())
                           if task_id in expected_tasks[str(rank)] and expected_tasks[str(rank)][task_id]["group_id"] == group_id)
            if actual != expected:
                errors.append(f"group {group_id} rank {rank} launch projection mismatch")

    coordinator_records = next((result.get("decision_records", []) for result in results
                                if result.get("decision_records")), [])
    dispatches = {item.get("task_id") for item in coordinator_records
                  if item.get("kind") == "decision" and item.get("decision") == "dispatch"}
    expected_global = set(expected_tasks.get("__all__", {}))
    if not coordinator_records:
        errors.append("coordinator decision records are missing")
    elif dispatches != expected_global:
        errors.append(f"coordinator dispatch task set mismatch: missing={sorted(expected_global - dispatches)}, extra={sorted(dispatches - expected_global)}")
    if actual_task_ids != expected_global:
        errors.append(f"global task result set mismatch: missing={sorted(expected_global - actual_task_ids)}, extra={sorted(actual_task_ids - expected_global)}")
        all_correct = False
    return {
        "status": "ok" if len(results) == world_size and bool(expected_global) and all_correct and not errors else "failed",
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
    dag_rank_task_timings: dict[str, dict[str, dict[str, float]]] = {}
    predicted_ready_error_s: dict[str, dict[str, float]] = {}
    dag_node_ready_wait_s: dict[str, dict[str, float]] = {}
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
        if result.get("input_mode") == "dag":
            rank = str(result.get("rank"))
            dag_events = result.get("dag_events", [])
            ready_at = {f"{event['job_id']}/{event['node_id']}": event["time_us"]
                        for event in dag_events if event.get("kind") == "node_ready" and event.get("node_kind") == "comm"}
            ready_by_node = {f"{event['job_id']}/{event['node_id']}": event["time_us"]
                             for event in dag_events if event.get("kind") == "node_ready"}
            compute_started = {f"{event['job_id']}/{event['node_id']}": event["time_us"]
                               for event in dag_events if event.get("kind") == "compute_started"}
            for node_id, ready_time in ready_by_node.items():
                started_time = compute_started.get(node_id)
                if started_time is not None:
                    dag_node_ready_wait_s.setdefault(rank, {})[node_id] = (started_time - ready_time) / 1_000_000.0
            declared = {event["task_id"]: event for event in dag_events if event.get("kind") == "comm_declared"}
            dag_by_task: dict[str, dict[str, int]] = {}
            for event in dag_events:
                if event.get("kind") == "node_ready" and event.get("node_kind") == "comm":
                    task_id = f"{event['job_id']}/{event['node_id']}"
                    dag_by_task.setdefault(task_id, {})["node_ready"] = event["time_us"]
                if event.get("task_id"):
                    dag_by_task.setdefault(event["task_id"], {})[event["kind"]] = event["time_us"]
            for task_id, prediction in declared.items():
                actual_ready = ready_at.get(task_id)
                predicted_ready_at = prediction.get("predicted_ready_at_us")
                if actual_ready is not None and isinstance(predicted_ready_at, int):
                    predicted_ready_error_s.setdefault(rank, {})[task_id] = (
                        actual_ready - predicted_ready_at
                    ) / 1_000_000.0
            for task_id, times in dag_by_task.items():
                runtime_times = event_times.get(task_id, {})
                values = {}
                if "node_ready" in times and "comm_submit_call" in times:
                    values["ready_to_submit_s"] = (times["comm_submit_call"] - times["node_ready"]) / 1_000_000.0
                if "offered" in runtime_times and "grant_received" in runtime_times:
                    values["offer_to_grant_s"] = (runtime_times["grant_received"] - runtime_times["offered"]) / 1_000_000.0
                if "grant_received" in runtime_times and "launch_start" in runtime_times:
                    values["grant_to_launch_s"] = (runtime_times["launch_start"] - runtime_times["grant_received"]) / 1_000_000.0
                if values:
                    dag_rank_task_timings.setdefault(rank, {})[task_id] = values
    return {
        "coordinator_task_timings": coordinator_tasks,
        "coordinator_idle_s": idle_by_reason,
        "coordinator_makespan_s": max(finished) - min(started) if finished and started else None,
        "job_duration_s": {job: max(values) for job, values in job_durations.items()},
        "rank_task_timings": local_waits,
        "dag_rank_task_timings": dag_rank_task_timings,
        "dag_node_ready_wait_s": dag_node_ready_wait_s,
        "predicted_ready_error_s": predicted_ready_error_s,
        "dag_timing_clock": "rank-local monotonic; never compare values across ranks",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("static_fifo", "static_ltf", "fifo", "ltf", "lookahead"), default="fifo")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--workload")
    source.add_argument("--dag", type=Path)
    parser.add_argument("--static-order", type=Path)
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--poll-interval", type=float, default=0.001)
    parser.add_argument("--dag-poll-interval", type=float, default=0.001)
    parser.add_argument("--compute-jitter", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--fault",
        choices=("none", "missing_task", "metadata_mismatch", "launch_failure", "completion_probe_failure",
                 "compute_failure", "binding_failure"),
        default="none",
    )
    args = parser.parse_args()
    if args.timeout <= 0 or args.poll_interval <= 0 or args.dag_poll_interval <= 0:
        parser.error("timeout and poll intervals must be positive")
    if not 0 <= args.compute_jitter < 1:
        parser.error("compute-jitter must be in [0, 1)")
    if args.epoch < 0 or args.world_size <= 0:
        parser.error("epoch must be non-negative and world-size positive")
    expected: dict[str, dict[str, Any]] = {str(rank): {} for rank in range(args.world_size)}
    expected["__all__"] = {}
    expected["__groups__"] = {}
    expected_nodes: dict[int, set[str]] | None = None
    digests: set[str] | None = None
    if args.dag:
        try:
            dag = load_dag(args.dag, epoch=args.epoch, world_size=args.world_size)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        if args.static_order and args.policy not in {"static_fifo", "static_ltf"}:
            parser.error("--static-order requires a static policy")
        if args.static_order:
            try:
                load_static_order(args.static_order, dag)
            except (OSError, ValueError) as exc:
                parser.error(str(exc))
        elif args.policy in {"static_fifo", "static_ltf"}:
            build_static_order(dag, args.policy)
        digests = {dag.manifest_digest}
        expected["__group_defs__"] = tuple(sorted((group.group_id, group.ranks) for group in dag.groups))
        expected_nodes = {rank: set() for rank in range(args.world_size)}
        for group in dag.groups:
            expected["__groups__"][group.group_id] = group.ranks
        for job in dag.jobs:
            job_groups = {node.group_id for node in job.nodes if hasattr(node, "group_id")}
            ranks = dag.group_ranks[next(iter(job_groups))]
            for rank in ranks:
                expected_nodes[rank].update(f"{job.job_id}/{node.node_id}" for node in job.nodes)
            for node in job.nodes:
                if not hasattr(node, "group_id"):
                    continue
                task_id = f"{job.job_id}/{node.node_id}"
                meta = {"group_id": node.group_id, "group_seq": node.group_seq}
                expected["__all__"][task_id] = meta
                for rank in dag.group_ranks[node.group_id]:
                    expected[str(rank)][task_id] = meta
    else:
        try:
            from .workloads import load_workload, ranks_for_job
            from .runtime_adapter import all_specs
        except ImportError:  # pragma: no cover
            from workloads import load_workload, ranks_for_job
            from runtime_adapter import all_specs
        workload = load_workload(args.workload or "balanced")
        for job, _index, spec, _hint in all_specs(workload, epoch=args.epoch):
            meta = {"group_id": spec.group_id, "group_seq": spec.group_seq}
            expected["__all__"][spec.task_id] = meta
            expected["__groups__"].setdefault(spec.group_id, ranks_for_job(job, args.world_size))
            for rank in ranks_for_job(job, args.world_size):
                expected[str(rank)][spec.task_id] = meta
        if args.static_order:
            parser.error("--static-order requires --dag")
    rendezvous_port, control_port = _free_port(), _free_port()
    processes = [_start(rank, args, rendezvous_port, control_port) for rank in range(args.world_size)]
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(processes)) as pool:
        futures = [pool.submit(_collect, rank, process, args.timeout + 5) for rank, process in enumerate(processes)]
        for future in futures:
            result, error = future.result()
            if error:
                errors.append(error)
            elif result is not None:
                results.append(result)
    results.sort(key=lambda item: item.get("rank", -1))
    validation = _validate_results(results, args.world_size, expected_tasks=expected,
                                   expected_nodes=expected_nodes, digests=digests,
                                   expected_config={
                                       "epoch": args.epoch, "world_size": args.world_size,
                                       "max_inflight": 1, "backend": args.backend,
                                       "completion_poll_interval_s": args.poll_interval,
                                       "dag_poll_interval_s": args.dag_poll_interval,
                                       "compute_jitter": args.compute_jitter,
                                   })
    validation["errors"] = errors + validation["errors"]
    if errors:
        validation["status"] = "failed"
    config = vars(args).copy()
    config["output"] = str(args.output) if args.output else None
    config["dag"] = str(args.dag) if args.dag else None
    config["static_order"] = str(args.static_order) if args.static_order else None
    git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE.parents[1],
                              capture_output=True, text=True, check=False)
    git_status = subprocess.run(["git", "status", "--porcelain"], cwd=HERE.parents[1],
                                capture_output=True, text=True, check=False)
    config["code_revision"] = git_head.stdout.strip() if git_head.returncode == 0 else None
    config["working_tree_dirty"] = bool(git_status.stdout.strip())
    payload = {"config": config, "validation": validation, "metrics": _metrics(results), "ranks": results}
    if args.output:
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if validation["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
