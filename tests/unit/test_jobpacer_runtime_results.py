from __future__ import annotations

from pathlib import Path

import pytest

from examples.jobpacer.runtime.runtime_adapter import load_dag
from examples.jobpacer.analysis.runtime_results import expected_dag_results, metrics, performance, validate_results


ROOT = Path(__file__).resolve().parents[2]


def _valid_results(graph, world_size):
    expected_data = expected_dag_results(graph, world_size)
    expected = expected_data["expected"]
    records = [{"kind": "decision", "decision": "dispatch", "task_id": task_id}
               for task_id in expected["__all__"]]
    results = []
    for rank in range(world_size):
        task_ids = list(expected[str(rank)])
        tasks = [{"task_id": task_id, **expected[str(rank)][task_id], "correct": True}
                 for task_id in task_ids]
        result = {
            "rank": rank,
            "status": "ok",
            "grant_sequence": task_ids,
            "launch_sequence": task_ids,
            "expected_task_ids": task_ids,
            "groups": [{"group_id": group.group_id, "ranks": list(group.ranks)}
                       for group in graph.groups],
            "manifest_digest": "known",
            "jobs": [{"job_id": "job", "status": "ok", "tasks": tasks,
                      "completed_node_ids": sorted(expected_data["expected_nodes"][rank])}],
        }
        result["decision_records"] = records if rank == 0 else []
        results.append(result)
    return expected_data, results


def test_expected_dag_results_and_validation_cover_tasks_nodes_members_and_launches():
    dag = load_dag(ROOT / "benchmark/phase3/experiments/dag-semantics/smoke/linear.json", world_size=2)
    expected_data, results = _valid_results(dag.graph, 2)
    validation = validate_results(results, 2, expected=expected_data["expected"],
                                  expected_nodes=expected_data["expected_nodes"], digests={"known"})
    assert validation == {
        "status": "ok", "all_collectives_correct": True, "rank_count": 2,
        "errors": [], "rank_task_sequences": [tuple(results[0]["grant_sequence"]),
                                                tuple(results[1]["grant_sequence"])],
    }

    missing = [dict(result) for result in results]
    missing[0] = {**missing[0], "jobs": [dict(missing[0]["jobs"][0])]}
    missing[0]["jobs"][0]["tasks"] = missing[0]["jobs"][0]["tasks"][:-1]
    assert validate_results(missing, 2, expected=expected_data["expected"],
                            expected_nodes=expected_data["expected_nodes"], digests={"known"})["status"] == "failed"

    bad_projection = [dict(result) for result in results]
    bad_projection[0] = {**bad_projection[0], "launch_sequence": []}
    assert validate_results(bad_projection, 2, expected=expected_data["expected"],
                            expected_nodes=expected_data["expected_nodes"], digests={"known"})["status"] == "failed"

    for mutation in ("duplicate", "missing-rank", "missing-member-task", "bad-digest",
                     "bad-group-manifest", "bad-group-seq", "incorrect-tensor"):
        expected_data, mutated = _valid_results(dag.graph, 2)
        if mutation == "duplicate":
            task = mutated[0]["jobs"][0]["tasks"][0]
            mutated[0]["jobs"][0]["tasks"].append(task)
        elif mutation == "missing-rank":
            mutated.pop()
        elif mutation == "missing-member-task":
            mutated[1]["jobs"][0]["tasks"].pop()
        elif mutation == "bad-digest":
            mutated[1]["manifest_digest"] = "other"
        elif mutation == "bad-group-manifest":
            mutated[0]["groups"][0]["ranks"] = [0]
        elif mutation == "bad-group-seq":
            mutated[0]["jobs"][0]["tasks"][0]["group_seq"] += 1
        else:
            mutated[0]["jobs"][0]["tasks"][0]["correct"] = False
        assert validate_results(mutated, 2, expected=expected_data["expected"],
                                expected_nodes=expected_data["expected_nodes"], digests={"known"})["status"] == "failed"


def test_non_coordinator_engines_validate_launch_projection_without_grant_records():
    dag = load_dag(ROOT / "benchmark/phase3/experiments/dag-semantics/smoke/linear.json", world_size=2)
    expected_data, results = _valid_results(dag.graph, 2)
    for result in results:
        result["comm_engine"] = "old"
        result["grant_sequence"] = []
        result["decision_records"] = []
    validation = validate_results(
        results, 2, expected=expected_data["expected"],
        expected_nodes=expected_data["expected_nodes"], digests={"known"},
    )
    assert validation["status"] == "ok"
    assert validation["rank_task_sequences"] == [
        tuple(item["launch_sequence"]) for item in results
    ]


def test_metrics_label_coordinator_epoch_duration_and_prediction_time_anchor():
    task_id = "job/comm"
    results = [{
        "rank": 0,
        "input_mode": "dag",
        "jobs": [{"job_id": "job", "job_start_ts": 10, "job_end_ts": 20,
                  "tasks": [{"task_id": task_id}]}],
        "runtime_events": [{"task_id": task_id, "kind": "offered", "time_us": 90},
                           {"task_id": task_id, "kind": "grant_received", "time_us": 100},
                           {"task_id": task_id, "kind": "launch_start", "time_us": 110}],
        "dag_events": [
            {"kind": "node_ready", "node_kind": "comm", "job_id": "job", "node_id": "comm", "time_us": 180},
            {"kind": "comm_declared", "task_id": task_id, "time_us": 150,
             "predicted_ready_at_us": 170},
        ],
        "decision_records": [
            {"kind": "group_registered", "now": 1.0},
            {"kind": "eligible", "task_id": task_id, "now": 1.5},
            {"kind": "decision", "decision": "dispatch", "task_id": task_id, "now": 2.0},
            {"kind": "submitted", "task_id": task_id, "now": 2.2},
            {"kind": "completed", "task_id": task_id, "now": 2.5},
            {"kind": "finished", "now": 3.0},
        ],
    }]
    result = metrics(results)
    assert set(result) == {
        "coordinator_task_timings", "coordinator_diagnostic_task_timings",
        "coordinator_message_queue_wait_us", "coordinator_message_processing_us",
        "coordinator_idle_s", "coordinator_epoch_duration_s",
        "job_duration_s", "rank_task_timings", "rank_process_cpu_time_s", "rank_context_switches",
        "dag_rank_task_timings",
        "dag_node_ready_wait_s", "predicted_ready_error_s", "dag_timing_clock",
    }
    assert result["predicted_ready_error_s"] == {"0": {task_id: 10 / 1_000_000}}
    assert result["coordinator_task_timings"][task_id]["grant_to_all_submitted_s"] == pytest.approx(0.2)
    assert result["coordinator_epoch_duration_s"] == pytest.approx(2.0)
    assert metrics([]) == {}


def test_linear_metrics_report_completion_feedback_and_probe_events():
    task_id = "job-0/comm-0"
    records = [
        {"kind": "decision", "decision": "dispatch", "task_id": task_id, "now": 1.0},
        {"kind": "submitted", "task_id": task_id, "now": 1.1},
        {"kind": "completed", "task_id": task_id, "now": 1.4},
        {"kind": "policy_snapshot", "now": 1.45, "eligible": [{"task_id": "job-1/comm-0"}]},
        {"kind": "decision", "decision": "dispatch", "task_id": "job-1/comm-0", "now": 1.5},
        {"kind": "submitted", "task_id": "job-1/comm-0", "now": 1.6},
        {"kind": "completed", "task_id": "job-1/comm-0", "now": 1.7},
    ]
    result = metrics([{
        "rank": 0,
        "process_cpu_time_s": 0.03,
        "voluntary_context_switches": 7,
        "involuntary_context_switches": 2,
        "decision_records": records,
        "runtime_events": [
            {"task_id": task_id, "kind": "submit_call", "time_us": 100},
            {"task_id": task_id, "kind": "grant_received", "time_us": 150},
            {"task_id": task_id, "kind": "collective_call_start", "time_us": 160},
            {"task_id": task_id, "kind": "collective_call_return", "time_us": 170},
            {"task_id": task_id, "kind": "completion_observed", "time_us": 240,
             "completion_probe_count": 4},
            {"task_id": task_id, "kind": "application_wait_start", "time_us": 200},
            {"task_id": task_id, "kind": "application_wait_return", "time_us": 250},
        ],
        "jobs": [{"job_id": "job-0", "tasks": [{
            "task_id": task_id, "ready_ts": 80, "submit_call_ts": 100,
            "consumer_end_ts": 250, "first_wait_ts": 200,
        }]}],
    }])
    timing = result["coordinator_task_timings"][task_id]
    assert timing["all_completed_to_next_grant_s"] == pytest.approx(0.1)
    assert timing["legal_candidate_present_during_gap"] is True
    local = result["rank_task_timings"]["0"][task_id]
    assert local["producer_ready_to_submit_call_s"] == pytest.approx(20 / 1_000_000)
    assert local["completion_probe_count"] == 4
    assert local["completion_observed_to_application_continue_s"] == pytest.approx(10 / 1_000_000)
    assert result["rank_process_cpu_time_s"] == {"0": 0.03}
    assert result["rank_context_switches"]["0"] == {"voluntary": 7, "involuntary": 2}


def test_coordinator_gap_classification_excludes_next_dispatch_snapshot():
    first, second = "job-0/comm-0", "job-0/comm-1"
    result = metrics([{
        "rank": 0,
        "decision_records": [
            {"kind": "decision", "decision": "dispatch", "task_id": first, "now": 1.0},
            {"kind": "submitted", "task_id": first, "now": 1.1},
            {"kind": "completed", "task_id": first, "now": 2.0},
            {"kind": "policy_snapshot", "now": 3.0, "eligible": [{"task_id": second}]},
            {"kind": "decision", "decision": "dispatch", "task_id": second, "now": 3.0},
        ],
        "coordinator_instrumentation": [
            {"kind": "capacity_released", "task_id": first, "time_us": 2_000_000},
            {"kind": "message_enqueued", "task_id": second, "message_kind": "OFFER",
             "endpoint": 0, "time_us": 2_500_000},
            {"kind": "message_enqueued", "task_id": second, "message_kind": "OFFER",
             "endpoint": 1, "time_us": 2_600_000},
            {"kind": "first_eligible", "task_id": second, "time_us": 2_900_000},
            {"kind": "decision_processing", "task_id": second,
             "time_us": 3_000_000, "end_time_us": 3_002_000},
            {"kind": "grant_committed", "task_id": second, "time_us": 3_002_000},
            {"kind": "grant_writer_queue_put_start", "task_id": second, "time_us": 3_003_000},
            {"kind": "grant_writer_queue_put_end", "task_id": second, "time_us": 3_004_000},
        ],
    }])
    legacy = result["coordinator_task_timings"][first]
    assert legacy["legal_candidate_present_during_gap"] is True
    assert legacy["legacy_gap_metric_includes_next_dispatch_snapshot"] is True
    assert legacy["legal_candidate_present_before_next_dispatch"] is False
    split = result["coordinator_diagnostic_task_timings"][second]
    assert split["last_offer_enqueued_us"] == 2_600_000
    assert split["capacity_release_to_last_offer_enqueued_us"] == 600_000
    assert split["candidate_ready_after_capacity_release_us"] == 900_000
    assert split["both_conditions_to_decision_start_us"] == 100_000
    assert split["decision_processing_us"] == 2_000


def test_performance_uses_rank_local_release_durations():
    results = [
        {
            "rank": 0,
            "application_release_ts": 100,
            "application_makespan_us": 30,
            "communication_drain_makespan_us": 40,
            "jobs": [{"job_id": "job", "job_end_ts": 125}],
        },
        {
            "rank": 1,
            "application_release_ts": 1000,
            "application_makespan_us": 35,
            "communication_drain_makespan_us": 45,
            "jobs": [{"job_id": "job", "job_end_ts": 1040}],
        },
    ]
    result = performance(results)
    assert result["workload_makespan_us"] == 35
    assert result["communication_drain_makespan_us"] == 45
    assert result["job_makespans"] == [{"job_id": "job", "makespan_us": 40}]
    assert "never subtract rank timestamps" in result["clock_semantics"]
    assert performance([]) == {}
