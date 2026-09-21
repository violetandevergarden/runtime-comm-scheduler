"""Pure validation and metrics for completed JobPacer runtime replays."""

from __future__ import annotations

from typing import Any

from runtime_comm_scheduler.dag import CommNode, DagGraph


def expected_dag_results(graph: DagGraph, world_size: int) -> dict[str, Any]:
    expected: dict[str, Any] = {str(rank): {} for rank in range(world_size)}
    expected["__all__"] = {}
    expected["__groups__"] = {}
    expected["__group_defs__"] = tuple(sorted((group.group_id, group.ranks) for group in graph.groups))
    expected_nodes = {rank: set() for rank in range(world_size)}
    group_ranks = {group.group_id: group.ranks for group in graph.groups}
    for group in graph.groups:
        expected["__groups__"][group.group_id] = group.ranks
    for job in graph.jobs:
        job_groups = {node.group_id for node in job.nodes if isinstance(node, CommNode)}
        ranks = group_ranks[next(iter(job_groups))]
        for rank in ranks:
            expected_nodes[rank].update(f"{job.job_id}/{node.node_id}" for node in job.nodes)
        for node in job.nodes:
            if not isinstance(node, CommNode):
                continue
            task_id = f"{job.job_id}/{node.node_id}"
            meta = {"group_id": node.group_id, "group_seq": node.group_seq}
            expected["__all__"][task_id] = meta
            for rank in group_ranks[node.group_id]:
                expected[str(rank)][task_id] = meta
    return {"expected": expected, "expected_nodes": expected_nodes}


def validate_results(results: list[dict[str, Any]], world_size: int, *,
                     expected: dict[str, dict[str, Any]],
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
            if actual_groups != expected.get("__group_defs__"):
                errors.append(f"rank {rank} group manifest mismatch")
        task_list = [task for job in result.get("jobs", ()) for task in job.get("tasks", ())]
        task_ids = [task.get("task_id") for task in task_list]
        if len(task_ids) != len(set(task_ids)):
            errors.append(f"rank {rank} has duplicate task results")
        rank_expected = set(expected.get(str(rank), {}))
        if set(task_ids) != rank_expected or len(task_ids) != len(rank_expected):
            errors.append(f"rank {rank} task set mismatch: missing={sorted(rank_expected - set(task_ids))}, extra={sorted(set(task_ids) - rank_expected)}")
        if set(result.get("expected_task_ids", task_ids)) != rank_expected:
            errors.append(f"rank {rank} reported expected task set mismatch")
        for task in task_list:
            task_id = task.get("task_id")
            if task_id in expected.get(str(rank), {}):
                expected_meta = expected[str(rank)][task_id]
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
    for task_id in expected.get("__all__", {}):
        meta = expected["__all__"][task_id]
        group_sequences.setdefault(meta["group_id"], []).append(task_id)
    for group_id, task_ids in group_sequences.items():
        group_expected = tuple(sorted(task_ids, key=lambda task_id: expected["__all__"][task_id]["group_seq"]))
        members = expected["__groups__"][group_id]
        for rank in members:
            result = by_rank.get(rank)
            if result is None:
                continue
            actual = tuple(task_id for task_id in result.get("launch_sequence", ())
                           if task_id in expected[str(rank)] and expected[str(rank)][task_id]["group_id"] == group_id)
            if actual != group_expected:
                errors.append(f"group {group_id} rank {rank} launch projection mismatch")

    coordinator_records = next((result.get("decision_records", []) for result in results
                                if result.get("decision_records")), [])
    dispatches = {item.get("task_id") for item in coordinator_records
                  if item.get("kind") == "decision" and item.get("decision") == "dispatch"}
    expected_global = set(expected.get("__all__", {}))
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


def metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
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
                "eligible_to_grant_s": grant_time - eligible_times[task_id] if task_id in eligible_times else None,
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
        "coordinator_epoch_duration_s": max(finished) - min(started) if finished and started else None,
        "job_duration_s": {job: max(values) for job, values in job_durations.items()},
        "rank_task_timings": local_waits,
        "dag_rank_task_timings": dag_rank_task_timings,
        "dag_node_ready_wait_s": dag_node_ready_wait_s,
        "predicted_ready_error_s": predicted_ready_error_s,
        "dag_timing_clock": "rank-local monotonic; never compare values across ranks",
    }
