from __future__ import annotations

from pathlib import Path

import pytest

from examples.jobpacer.runtime_adapter import load_dag
from examples.jobpacer.runtime_results import expected_dag_results, metrics, validate_results


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
    dag = load_dag(ROOT / "benchmark/phase3/linear.json", world_size=2)
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
        "coordinator_task_timings", "coordinator_idle_s", "coordinator_epoch_duration_s",
        "job_duration_s", "rank_task_timings", "dag_rank_task_timings",
        "dag_node_ready_wait_s", "predicted_ready_error_s", "dag_timing_clock",
    }
    assert result["predicted_ready_error_s"] == {"0": {task_id: 10 / 1_000_000}}
    assert result["coordinator_task_timings"][task_id]["grant_to_all_submitted_s"] == pytest.approx(0.2)
    assert result["coordinator_epoch_duration_s"] == pytest.approx(2.0)
    assert metrics([]) == {}
