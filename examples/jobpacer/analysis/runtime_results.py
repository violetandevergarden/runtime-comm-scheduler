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
    coordinator_rank = by_rank.get(0, {})
    protocol_counts = coordinator_rank.get("protocol_transition_counts")
    if protocol_counts is not None:
        expected_reports = sum(
            len(expected.get("__groups__", {}).get(meta["group_id"], ()))
            for meta in expected.get("__all__", {}).values()
        )
        if protocol_counts.get("submitted_reports") != expected_reports:
            errors.append("coordinator SUBMITTED report count mismatch")
        if protocol_counts.get("completed_reports") != expected_reports:
            errors.append("coordinator COMPLETED report count mismatch")
        if protocol_counts.get("submitted_before_completed_enforced") is not True:
            errors.append("coordinator did not confirm SUBMITTED-before-COMPLETED enforcement")
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
    coordinator_instrumentation = next(
        (result.get("coordinator_instrumentation", []) for result in results
         if result.get("coordinator_instrumentation")),
        [],
    )
    minimal_observation = any(result.get("observation_mode") == "minimal" for result in results)
    dispatch_times = {
        item["task_id"]: item["now"]
        for item in coordinator_records
        if item.get("kind") == "decision" and item.get("decision") == "dispatch"
        and item.get("now") is not None
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
    dispatch_records = [item for item in coordinator_records
                        if item.get("kind") == "decision" and item.get("decision") == "dispatch"]
    snapshots = [item for item in coordinator_records if item.get("kind") == "policy_snapshot"]
    coordinator_tasks = {}
    for dispatch_index, dispatch in enumerate(dispatch_records):
        if dispatch.get("now") is None:
            continue
        task_id = dispatch["task_id"]
        grant_time = dispatch["now"]
        submitted = progress.get(task_id, {}).get("submitted", [])
        completed = progress.get(task_id, {}).get("completed", [])
        if submitted and completed:
            submitted_all = max(submitted)
            completed_all = max(completed)
            next_grant = (dispatch_records[dispatch_index + 1]
                          if dispatch_index + 1 < len(dispatch_records) else None)
            next_grant_time = next_grant.get("now") if next_grant else None
            gap_snapshots = [item for item in snapshots
                             if completed_all <= item.get("now", float("-inf"))
                             and next_grant_time is not None
                             and item.get("now", float("inf")) <= next_grant_time]
            eligible_in_gap = any(item.get("eligible") for item in gap_snapshots)
            eligible_before_dispatch = any(
                item.get("eligible") and item.get("now", float("inf")) < next_grant_time
                for item in gap_snapshots
            ) if next_grant_time is not None else None
            coordinator_tasks[task_id] = {
                "eligible_to_grant_s": grant_time - eligible_times[task_id] if task_id in eligible_times else None,
                "grant_to_all_submitted_s": submitted_all - grant_time,
                "all_submitted_to_all_completed_s": completed_all - submitted_all,
                "all_completed_to_next_grant_s": (next_grant_time - completed_all
                                                   if next_grant_time is not None else None),
                "next_task_id": next_grant.get("task_id") if next_grant else None,
                "legal_candidate_present_during_gap": (eligible_in_gap
                                                        if next_grant_time is not None else None),
                "legal_candidate_present_before_next_dispatch": eligible_before_dispatch,
                "legacy_gap_metric_includes_next_dispatch_snapshot": True,
                "coordinator_clock": "single coordinator monotonic clock",
            }
    coordinator_diag_by_kind: dict[str, list[dict[str, Any]]] = {}
    coordinator_diag_by_task: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for item in coordinator_instrumentation:
        coordinator_diag_by_kind.setdefault(item.get("kind", "unknown"), []).append(item)
        if item.get("task_id") is not None:
            coordinator_diag_by_task.setdefault(item["task_id"], {}).setdefault(
                item.get("kind", "unknown"), []).append(item)
    coordinator_diagnostic_tasks: dict[str, dict[str, Any]] = {}
    first_eligible_us = {
        task_id: min(item["time_us"] for item in kinds.get("first_eligible", []))
        for task_id, kinds in coordinator_diag_by_task.items() if kinds.get("first_eligible")
    }
    capacity_release_us = {
        task_id: max(item["time_us"] for item in kinds.get("capacity_released", []))
        for task_id, kinds in coordinator_diag_by_task.items() if kinds.get("capacity_released")
    }
    for index, dispatch in enumerate(dispatch_records):
        task_id = dispatch["task_id"]
        kinds = coordinator_diag_by_task.get(task_id, {})
        decision = next((item for item in kinds.get("decision_processing", [])), None)
        committed = next((item for item in kinds.get("grant_committed", [])), None)
        queue_starts = kinds.get("grant_writer_queue_put_start", [])
        queue_ends = kinds.get("grant_writer_queue_put_end", [])
        socket_starts = kinds.get("control_socket_sendall_start", [])
        socket_ends = kinds.get("control_socket_sendall_end", [])
        offer_enqueues = [item for item in kinds.get("message_enqueued", [])
                          if item.get("message_kind") == "OFFER"]
        last_offer_enqueued = max((item["time_us"] for item in offer_enqueues), default=None)
        previous_id = dispatch_records[index - 1]["task_id"] if index else None
        release = capacity_release_us.get(previous_id) if previous_id else None
        eligible = first_eligible_us.get(task_id)
        decision_start = decision.get("time_us") if decision else None
        decision_end = decision.get("end_time_us") if decision else None
        ready = max(release, eligible) if release is not None and eligible is not None else None
        coordinator_diagnostic_tasks[task_id] = {
            "previous_task_id": previous_id,
            "previous_capacity_release_us": release,
            "last_offer_enqueued_us": last_offer_enqueued,
            "capacity_release_to_last_offer_enqueued_us": (
                last_offer_enqueued - release
                if last_offer_enqueued is not None and release is not None else None),
            "first_eligible_us": eligible,
            "decision_start_us": decision_start,
            "decision_end_us": decision_end,
            "decision_processing_us": (decision_end - decision_start
                                        if decision_start is not None and decision_end is not None else None),
            "ready_after_capacity_and_offer_us": ready,
            "both_conditions_to_decision_start_us": (max(0, decision_start - ready)
                if decision_start is not None and ready is not None else None),
            "candidate_ready_after_capacity_release_us": (max(0, eligible - release)
                if eligible is not None and release is not None else None),
            "capacity_held_after_candidate_ready_us": (max(0, release - eligible)
                if eligible is not None and release is not None else None),
            "grant_commit_us": committed.get("time_us") if committed else None,
            "grant_writer_queue_put_start_us": min((item["time_us"] for item in queue_starts), default=None),
            "grant_writer_queue_put_end_us": max((item["time_us"] for item in queue_ends), default=None),
            "grant_socket_sendall_start_us": min((item["time_us"] for item in socket_starts), default=None),
            "grant_socket_sendall_end_us": max((item["time_us"] for item in socket_ends), default=None),
            "post_decision_to_queue_us": (min(item["time_us"] for item in queue_starts) - decision_end
                if queue_starts and decision_end is not None else None),
            "writer_queue_to_sendall_end_us": (max(item["time_us"] for item in socket_ends)
                - min(item["time_us"] for item in queue_ends)
                if socket_ends and queue_ends else None),
            "interval_classification": (
                "capacity_and_candidate_ready_processing"
                if release is not None and eligible is not None and decision_start is not None
                else "diagnostic_boundaries_missing"),
            "coordinator_clock": "single coordinator monotonic clock",
        }
    message_queue_wait_us: dict[str, list[int]] = {}
    for item in coordinator_diag_by_kind.get("event_loop_dequeued", []):
        if item.get("queue_wait_us") is not None:
            message_queue_wait_us.setdefault(item["message_kind"], []).append(item["queue_wait_us"])
    message_processing_us: dict[str, list[int]] = {}
    processing_starts = [item for item in coordinator_diag_by_kind.get("event_processing_start", [])]
    processing_ends = [item for item in coordinator_diag_by_kind.get("event_processing_end", [])]
    end_by_key = {(item.get("endpoint"), item.get("event_seq")): item for item in processing_ends}
    for item in processing_starts:
        ended = end_by_key.get((item.get("endpoint"), item.get("event_seq")))
        if ended is not None:
            message_processing_us.setdefault(item["message_kind"], []).append(
                ended["time_us"] - item["time_us"])
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
    rank_process_cpu_time_s: dict[str, float] = {}
    rank_context_switches: dict[str, dict[str, int]] = {}
    dag_rank_task_timings: dict[str, dict[str, dict[str, float]]] = {}
    predicted_ready_error_s: dict[str, dict[str, float]] = {}
    dag_node_ready_wait_s: dict[str, dict[str, float]] = {}
    for result in results:
        rank_key = str(result.get("rank"))
        if isinstance(result.get("process_cpu_time_s"), (int, float)):
            rank_process_cpu_time_s[rank_key] = float(result["process_cpu_time_s"])
        if "voluntary_context_switches" in result and "involuntary_context_switches" in result:
            rank_context_switches[rank_key] = {
                "voluntary": int(result["voluntary_context_switches"]),
                "involuntary": int(result["involuntary_context_switches"]),
            }
        event_times: dict[str, dict[str, int]] = {}
        event_records: dict[str, dict[str, dict[str, Any]]] = {}
        for event in result.get("runtime_events", []):
            if event.get("task_id"):
                event_times.setdefault(event["task_id"], {})[event["kind"]] = event["time_us"]
                event_records.setdefault(event["task_id"], {})[event["kind"]] = event
        for job in result.get("jobs", []):
            for task in job.get("tasks", []):
                times = event_times.get(task["task_id"], {})
                if "grant_received" in times and "submit_call_ts" in task:
                    values: dict[str, float] = {
                        "submit_call_to_grant_s": (times["grant_received"] - task["submit_call_ts"]) / 1_000_000.0,
                        "grant_to_collective_start_s": (times.get("collective_call_start", times.get("launch_start", times["grant_received"]))
                                                         - times["grant_received"]) / 1_000_000.0,
                    }
                    def interval(start: str, end: str) -> float | None:
                        if start in times and end in times:
                            return (times[end] - times[start]) / 1_000_000.0
                        return None

                    for name, start, end in (
                        ("submit_api_duration_s", "submit_call_start", "submit_call_return"),
                        ("offer_send_duration_s", "offer_send_start", "offer_send_end"),
                        ("grant_receive_to_launch_dequeue_s", "grant_received", "launch_worker_dequeued"),
                        ("launch_dequeue_to_collective_start_s", "launch_worker_dequeued", "collective_call_start"),
                        ("collective_return_to_submitted_send_end_s", "collective_call_return", "submitted_send_end"),
                        ("completion_observed_to_application_wait_return_s", "completion_observed", "application_wait_return"),
                        ("completion_observed_to_completed_send_end_s", "completion_observed", "completed_send_end"),
                        ("first_probe_to_completion_observed_s", "completion_probe_first", "completion_observed"),
                    ):
                        duration = interval(start, end)
                        if duration is not None:
                            values[name] = duration
                    if task.get("ready_ts") is not None:
                        values["producer_ready_to_submit_call_s"] = (
                            task["submit_call_ts"] - task["ready_ts"]
                        ) / 1_000_000.0
                    if task.get("binding_create_duration_us") is not None:
                        values["binding_create_s"] = task["binding_create_duration_us"] / 1_000_000.0
                    if task.get("tensor_create_duration_us") is not None:
                        values["tensor_create_s"] = task["tensor_create_duration_us"] / 1_000_000.0
                    call_start = times.get("collective_call_start", task.get("collective_call_start_ts"))
                    call_return = times.get("collective_call_return", task.get("collective_call_return_ts"))
                    completion = times.get("completion_observed", task.get("completion_observed_ts"))
                    if call_start is not None and call_return is not None:
                        values["collective_call_duration_s"] = (call_return - call_start) / 1_000_000.0
                    if call_return is not None and completion is not None:
                        values["call_return_to_completion_observation_s"] = (
                            completion - call_return
                        ) / 1_000_000.0
                    app_wait_start = times.get("application_wait_start", task.get("first_wait_ts"))
                    app_wait_return = times.get("application_wait_return", task.get("consumer_end_ts"))
                    if app_wait_start is not None and app_wait_return is not None:
                        values["application_wait_s"] = (app_wait_return - app_wait_start) / 1_000_000.0
                    if completion is not None and app_wait_return is not None:
                        values["completion_observed_to_application_continue_s"] = (
                            app_wait_return - completion
                        ) / 1_000_000.0
                    completion_event = event_records.get(task["task_id"], {}).get("completion_observed", {})
                    if completion_event.get("completion_probe_count") is not None:
                        values["completion_probe_count"] = float(completion_event["completion_probe_count"])
                    if completion_event.get("completion_probe_total_us") is not None:
                        values["completion_probe_total_s"] = (
                            completion_event["completion_probe_total_us"] / 1_000_000.0)
                    first_probe_us = completion_event.get("completion_first_probe_us")
                    last_probe_us = completion_event.get("completion_last_probe_us")
                    probe_count = completion_event.get("completion_probe_count")
                    if (isinstance(first_probe_us, int) and isinstance(last_probe_us, int)
                            and isinstance(probe_count, int) and probe_count > 1):
                        values["mean_probe_interval_s"] = (
                            (last_probe_us - first_probe_us) / (probe_count - 1) / 1_000_000.0)
                    local_waits.setdefault(rank_key, {})[task["task_id"]] = values
        for job in result.get("jobs", []):
            tasks = job.get("tasks", [])
            for current, following in zip(tasks, tasks[1:]):
                current_times = event_times.get(current.get("task_id"), {})
                next_times = event_times.get(following.get("task_id"), {})
                wait_return = current_times.get("application_wait_return")
                next_producer = next_times.get("producer_start")
                if wait_return is not None and next_producer is not None:
                    local_waits.setdefault(rank_key, {}).setdefault(current["task_id"], {})[
                        "application_wait_return_to_next_producer_s"] = (
                            next_producer - wait_return) / 1_000_000.0
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
        "coordinator_diagnostic_task_timings": coordinator_diagnostic_tasks,
        "coordinator_message_queue_wait_us": message_queue_wait_us,
        "coordinator_message_processing_us": message_processing_us,
        "coordinator_idle_s": idle_by_reason,
        "coordinator_epoch_duration_s": (max(finished) - min(started)
                                          if (not minimal_observation or coordinator_instrumentation)
                                          and finished and started else None),
        "job_duration_s": {job: max(values) for job, values in job_durations.items()},
        "rank_task_timings": local_waits,
        "rank_process_cpu_time_s": rank_process_cpu_time_s,
        "rank_context_switches": rank_context_switches,
        "dag_rank_task_timings": dag_rank_task_timings,
        "dag_node_ready_wait_s": dag_node_ready_wait_s,
        "predicted_ready_error_s": predicted_ready_error_s,
        "dag_timing_clock": "rank-local monotonic; never compare values across ranks",
    }


def performance(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate common application boundaries without subtracting rank clocks."""
    if not results or any("application_makespan_us" not in result for result in results):
        return {}
    ordered = sorted(results, key=lambda result: int(result.get("rank", -1)))
    job_records: dict[str, list[float]] = {}
    for result in ordered:
        release = result["application_release_ts"]
        for job in result.get("jobs", ()):
            if "job_end_ts" not in job:
                continue
            job_records.setdefault(job["job_id"], []).append(job["job_end_ts"] - release)
    job_makespans = [
        {"job_id": job_id, "makespan_us": max(values)}
        for job_id, values in sorted(job_records.items())
    ]
    application_values = [result["application_makespan_us"] for result in ordered]
    drain_values = [
        result.get("communication_drain_makespan_us", result["application_makespan_us"])
        for result in ordered
    ]
    return {
        "job_makespans": job_makespans,
        "workload_makespan_us": max(application_values),
        "rank_application_makespans_us": application_values,
        "communication_drain_makespan_us": max(drain_values),
        "rank_communication_drain_makespans_us": drain_values,
        "clock_semantics": "rank-local durations; aggregate max, never subtract rank timestamps",
    }
