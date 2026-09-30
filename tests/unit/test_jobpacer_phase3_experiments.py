from __future__ import annotations

import csv
import json
import os
import socket
from argparse import Namespace
from pathlib import Path
import subprocess
import sys

import pytest

from examples.jobpacer.comm_profile import CommunicationProfile, ProfileRecord
from examples.jobpacer.runtime.runtime_adapter import apply_dag_profile, load_dag, make_replay_compute
from examples.jobpacer.scripts.run_experiments import _attach_isolated, _command, _mechanism_row, _paired_rows
from examples.jobpacer.scripts import run_experiments as experiment_batch
from examples.jobpacer.scripts import run_phase3 as phase3_replay
from examples.jobpacer.diagnostics import interleaved_isolated as isolated_batch
from examples.jobpacer.scripts import run_compact_suite as compact_suite
from examples.jobpacer.scripts.run_compact_suite import (
    _check_lookahead_evidence, _check_mechanism_evidence, _suite, _validate_profile,
)
from examples.jobpacer.workloads import linear_execution_duration, load_workload
from runtime_comm_scheduler.dag.model import compute_tails
from runtime_comm_scheduler.runtime.policy import Candidate, select_fifo, select_ltf


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "benchmark/phase3/experiments"


def test_bare_failure_gives_peer_a_bounded_chance_to_observe_shared_failure():
    class Child:
        def __init__(self, returncode=None, *, wait_error=None):
            self.returncode = returncode
            self.wait_error = wait_error
            self.wait_timeouts = []
            self.killed = False

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.wait_timeouts.append(timeout)
            if self.wait_error is not None:
                raise self.wait_error
            self.returncode = 1
            return self.returncode

        def kill(self):
            self.killed = True
            self.returncode = -9

    failed = Child(returncode=1)
    peer = Child()
    phase3_replay._stop_failed_siblings([failed, peer], comm_engine="bare")
    assert peer.wait_timeouts == [phase3_replay.BARE_FAILURE_CLEANUP_GRACE_S]
    assert not peer.killed

    timed_out_peer = Child(wait_error=subprocess.TimeoutExpired("worker", 5.0))
    phase3_replay._stop_failed_siblings([failed, timed_out_peer], comm_engine="bare")
    assert timed_out_peer.killed

    raw_peer = Child()
    phase3_replay._stop_failed_siblings([failed, raw_peer], comm_engine="raw-ordered")
    assert raw_peer.wait_timeouts == []
    assert raw_peer.killed


def test_rendezvous_retry_is_bounded_to_pre_task_tcpstore_bind_conflicts():
    conflict = (
        "rank 0 exited 1: DistNetworkError: The server socket has failed to listen "
        "on any local network address. code: -98, name: EADDRINUSE, "
        "message: address already in use"
    )
    assert phase3_replay._is_rendezvous_bind_conflict([conflict])
    assert phase3_replay._should_retry_rendezvous_startup([], [conflict], 0)
    assert not phase3_replay._should_retry_rendezvous_startup(
        [], [conflict], phase3_replay.RENDEZVOUS_STARTUP_MAX_ATTEMPTS - 1)
    assert not phase3_replay._should_retry_rendezvous_startup(
        [{"rank": 0}], [conflict], 0)
    assert not phase3_replay._should_retry_rendezvous_startup(
        [], ["application failed: address already in use"], 0)


def _profile(*sizes: int) -> CommunicationProfile:
    records = tuple(ProfileRecord(
        op="all_reduce", num_bytes=size, dtype="float32", group_size=2,
        backend="gloo", device_type="cpu", reduction="sum", p50_s=size / 1e10,
        p10_s=size / 2e10, p90_s=size / 5e9, mean_s=size / 1e10,
        stdev_s=0.0, samples=30,
    ) for size in sizes)
    return CommunicationProfile(1, {
        "backend": "gloo", "device_type": "cpu", "world_size": 2,
        "group_ranks": [[0, 1]],
        "hostname": socket.gethostname(), "torch_version": __import__("torch").__version__,
    }, {"warmup": 5, "iterations": 30}, records)


def test_formal_linear_inputs_parse_and_execution_override_does_not_change_hint():
    paths = sorted((EXPERIMENTS / "baseline").glob("L[0-5]-*.json"))
    paths += sorted((EXPERIMENTS / "readiness").glob("L[0-5]-*.json"))
    paths += sorted((EXPERIMENTS / "priority").glob("L[0-5]-*.json"))
    paths += sorted((EXPERIMENTS / "lookahead").glob("L[0-5]-*.json"))
    paths.sort(key=lambda path: int(path.name[1]))
    paths = [path for path in paths if path.name != "L0-no-overlap-bridge.json"]
    assert [path.name[:2] for path in paths] == [f"L{index}" for index in range(6)]
    l1 = load_workload(paths[1])
    comm = l1.jobs[0].communications[0]
    assert comm.producer_compute_s == pytest.approx(0.002)
    assert linear_execution_duration(l1, 0, "job-0", 0, 0, "producer",
                                     comm.producer_compute_s, 0.0) == pytest.approx(0.015)
    assert linear_execution_duration(l1, 0, "job-0", 0, 0, "consumer",
                                     comm.consumer_compute_s, 0.0) == pytest.approx(0.001)
    l5 = load_workload(EXPERIMENTS / "readiness/L5-member-skew.json")
    comm = l5.jobs[0].communications[0]
    assert linear_execution_duration(l5, 0, "job-0", 0, 0, "producer",
                                     comm.producer_compute_s, 0.0) == pytest.approx(0.002)
    assert linear_execution_duration(l5, 0, "job-0", 0, 1, "producer",
                                     comm.producer_compute_s, 0.0) == pytest.approx(0.015)


def test_formal_dag_inputs_parse_and_strict_profile_rewrites_every_signature():
    dag_directories = ("bridge", "dag-semantics", "priority", "lookahead")
    paths = sorted((path for directory in dag_directories
                    for path in (EXPERIMENTS / directory).glob("G[0-4]-*.json")
                    if path.name != "G0-interleaved-order.json"),
                   key=lambda path: int(path.name[1]))
    assert [path.name[:2] for path in paths] == [f"G{index}" for index in range(5)]
    profile = _profile(1048576, 16777216)
    for path in paths:
        dag = load_dag(path, world_size=2)
        applied = apply_dag_profile(
            dag, profile, {"backend": "gloo", "device_type": "cpu", "world_size": 2}
        )
        assert applied.manifest_digest != dag.manifest_digest
        assert json.loads(applied.canonical_json)["name"] == dag.name
    with pytest.raises(ValueError, match="missing DAG"):
        apply_dag_profile(
            load_dag(paths[0], world_size=2), _profile(4096),
            {"backend": "gloo", "device_type": "cpu", "world_size": 2},
        )


def test_replay_entry_rejects_formal_input_without_profile_before_opening_sockets(tmp_path):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(ROOT)))
    completed = subprocess.run([
        sys.executable, "-m", "examples.jobpacer.scripts.run_phase3",
        "--policy", "fifo", "--workload", str(EXPERIMENTS / "baseline/L0-balanced.json"),
        "--output", str(tmp_path / "result.json"),
    ], cwd=ROOT, env=env, capture_output=True, text=True, timeout=10)
    assert completed.returncode == 2
    assert "require --comm-profile" in completed.stderr
    assert not (tmp_path / "result.json").exists()


def test_seed_pairing_uses_repeat_medians_and_reports_missing_blocks():
    rows = []
    for policy, values in {"static_fifo": ((10.0, 12.0), (20.0, 22.0)),
                           "fifo": ((8.0, 10.0), (18.0, 20.0))}.items():
        for epoch, repeats in enumerate(values):
            for repeat, value in enumerate(repeats):
                rows.append({"group": "new", "policy": policy, "epoch": epoch,
                             "repeat": repeat, "status": "ok", "makespan_s": value})
    paired = _paired_rows(rows, baseline="new-static_fifo", tie_threshold=0.01,
                          bootstrap_samples=100, bootstrap_seed=7)
    assert len(paired) == 1
    assert paired[0]["paired_seeds"] == 2
    assert paired[0]["median_difference_s"] == pytest.approx(-2.0)
    assert paired[0]["seed_block_median_delta_s"] == pytest.approx(-2.0)
    assert paired[0]["paired_ratio"] == pytest.approx(1.1652777777777779)
    assert paired[0]["wins"] == 2
    assert paired[0]["missing_seed_blocks"] == 0


def test_primary_estimator_is_median_of_paired_repeat_deltas_not_arm_medians():
    rows = []
    for policy, values in {"static_fifo": (1.0, 100.0, 101.0),
                           "fifo": (2.0, 3.0, 102.0)}.items():
        for repeat, value in enumerate(values):
            rows.append({"group": "new", "policy": policy, "epoch": 1,
                         "repeat": repeat, "status": "ok", "makespan_s": value})
    paired = _paired_rows(rows, baseline="new-static_fifo", tie_threshold=0.01,
                          bootstrap_samples=0, bootstrap_seed=7)[0]
    assert paired["arm_median_difference_s"] == pytest.approx(-97.0)
    assert paired["median_difference_s"] == pytest.approx(-97.0)  # retained historical alias
    assert paired["paired_delta_median_within_seed"] == [
        {"seed": 1, "median_delta_s": 1.0},
    ]
    assert paired["seed_block_median_delta_s"] == pytest.approx(1.0)


def test_seed_pairing_never_combines_different_successful_repeats():
    rows = [
        {"group": "new", "policy": "static_fifo", "epoch": 0, "repeat": 0,
         "status": "ok", "makespan_s": 2.0},
        {"group": "new", "policy": "fifo", "epoch": 0, "repeat": 0,
         "status": "failed", "makespan_s": None},
        {"group": "new", "policy": "static_fifo", "epoch": 0, "repeat": 1,
         "status": "failed", "makespan_s": None},
        {"group": "new", "policy": "fifo", "epoch": 0, "repeat": 1,
         "status": "ok", "makespan_s": 1.0},
    ]
    paired = _paired_rows(rows, baseline="new-static_fifo", tie_threshold=0.01,
                          bootstrap_samples=0, bootstrap_seed=0)
    assert paired[0]["paired_seeds"] == 0
    assert paired[0]["missing_seed_blocks"] == 1


def test_binding_and_poll_arms_are_paired_inside_randomized_blocks():
    arms = ("old-ltf", "new-ltf-on-ready", "new-ltf-precreate")
    plan = experiment_batch._plan((5,), 2, arms, 17)
    by_repeat = {repeat: [item for item in plan if item["repeat"] == repeat]
                 for repeat in (0, 1)}
    for items in by_repeat.values():
        assert {item["arm"] for item in items} == set(arms)
        assert len({tuple(item["block_order"]) for item in items}) == 1
        assert len(set(items[0]["block_order"])) == len(arms)
    assert {item["binding_preparation"] for item in by_repeat[0]} == {
        "legacy_tensor_precreated", "on-ready", "precreate",
    }

    polling = experiment_batch._plan((5,), 1,
                                     ("new-ltf-poll-1ms", "new-ltf-poll-0.2ms"), 3)
    assert {item["poll_interval"] for item in polling} == {0.001, 0.0002}
    assert all(item["binding_preparation"] == "precreate" for item in polling)

    observation = experiment_batch._plan(
        (6200,), 5,
        ("new-static_fifo-minimal", "new-static_fifo-diagnostic"), 91,
        binding_preparation="precreate", poll_interval=0.001)
    for repeat in range(5):
        block = [item for item in observation if item["repeat"] == repeat]
        assert {item["observation_mode"] for item in block} == {"minimal", "diagnostic"}
        assert len({tuple(item["block_order"]) for item in block}) == 1
        assert {item["binding_preparation"] for item in block} == {"precreate"}


def test_schema_v2_dag_arms_share_runtime_worker_and_keep_raw_ordered_explicit():
    arms = experiment_batch.DAG_SUPPORTED_ARMS
    planned = experiment_batch._plan((91,), 1, arms, 13)
    assert {item["comm_engine"] for item in planned} == {"old", "new"}
    assert len(planned) == 6
    raw = experiment_batch._plan((91,), 1, ("raw-ordered-static-fifo",), 13)[0]
    assert raw["comm_engine"] == "raw-ordered"
    assert raw["arm"] not in arms

    args = Namespace(
        dag="fork-join.json", workload=None, backend="nccl", world_size=2,
        compute_jitter=0.0, wait_budget_s=0.02, poll_interval=0.001,
        timeout=45, warmup_iterations=1, binding_preparation="precreate",
        comm_profile="comm.json", compute_profile="compute.json",
        static_ltf_order=None,
    )
    command = _command(args, "static_fifo", 91, Path("old.json"), old=True,
                       comm_engine="old")
    assert str(experiment_batch.RUNTIME_REPLAY) in command
    assert command[command.index("--comm-engine") + 1] == "old"
    assert command[command.index("--dag") + 1] == "fork-join.json"
    assert command[command.index("--compute-profile") + 1] == "compute.json"

    command = _command(
        Namespace(dag=None, workload="chain.json", backend="gloo", world_size=2,
                  epoch=6300, compute_jitter=0.0, wait_budget_s=0.02,
                  poll_interval=0.001, timeout=20, warmup_iterations=1,
                  binding_preparation="precreate", comm_profile=None,
                  static_ltf_order=None),
        "static_fifo", 6300, Path("out.json"), observation_mode="minimal",
        declaration_mode="on-submit")
    assert command[command.index("--declaration-mode") + 1] == "on-submit"


def test_task_timing_rows_export_common_admission_and_completion_boundaries():
    record = {
        "run_id": "run", "arm": "new-ltf-precreate", "group": "new",
        "config": {"epoch": 1, "repeat": 2, "binding_preparation": "precreate",
                   "poll_interval_s": 0.001}, "returncode": 0,
    }
    trace = {"ranks": [{
        "rank": 0,
        "runtime_events": [
            {"task_id": "job-0/comm-0", "kind": "declare_call_start", "time_us": 70},
            {"task_id": "job-0/comm-0", "kind": "declare_call_end", "time_us": 78},
            {"task_id": "job-0/comm-0", "kind": "control_send_interval", "time_us": 77,
             "message_kind": "DECLARE", "lock_wait_us": 2, "socket_write_flush_us": 5},
            {"task_id": "job-0/comm-0", "kind": "control_send_interval", "time_us": 98,
             "message_kind": "OFFER", "lock_wait_us": 3, "socket_write_flush_us": 12},
            {"task_id": "job-0/comm-0", "kind": "grant_received", "time_us": 150},
            {"task_id": "job-0/comm-0", "kind": "collective_call_start", "time_us": 160},
            {"task_id": "job-0/comm-0", "kind": "collective_call_return", "time_us": 170},
            {"task_id": "job-0/comm-0", "kind": "completion_observed", "time_us": 240,
             "completion_probe_count": 4, "completion_first_probe_us": 235,
             "completion_last_probe_us": 239},
        ],
        "jobs": [{"job_id": "job-0", "tasks": [{
            "task_id": "job-0/comm-0", "ordinal": 0, "ready_ts": 80,
            "submit_call_ts": 100, "submit_return_ts": 110,
            "binding_create_duration_us": 5, "correct": True,
        }]}],
    }]}

    rows = experiment_batch._task_timing_rows(record, trace)

    assert rows[0]["producer_ready_to_submit_call_us"] == 20
    assert rows[0]["declare_call_us"] == 8
    assert rows[0]["declare_control_send_lock_wait_us"] == 2
    assert rows[0]["declare_socket_write_flush_us"] == 5
    assert rows[0]["offer_control_send_lock_wait_us"] == 3
    assert rows[0]["offer_socket_write_flush_us"] == 12
    assert rows[0]["submit_call_to_grant_us"] == 50
    assert rows[0]["grant_to_collective_start_us"] == 10
    assert rows[0]["collective_call_us"] == 10
    assert rows[0]["call_return_to_completion_observation_us"] == 70
    assert rows[0]["call_return_to_first_probe_us"] == 65
    assert rows[0]["mean_completion_probe_interval_us"] == 4 / 3
    assert rows[0]["binding_create_us"] == 5
    assert rows[0]["completion_probe_count"] == 4
    assert rows[0]["declaration_mode"] == "before-producer"


def test_coordinator_instrumentation_exports_measured_policy_duration():
    record = {
        "run_id": "control", "arm": "new-static_fifo", "returncode": 0,
        "config": {"epoch": 1, "repeat": 0, "observation_mode": "diagnostic"},
    }
    trace = {"ranks": [{
        "rank": 0,
        "coordinator_instrumentation": [{
            "kind": "policy_call", "time_us": 100, "end_time_us": 107,
            "eligible_count": 2,
        }],
        "decision_records": [],
    }], "metrics": {}}
    tasks, events = experiment_batch._coordinator_diagnostic_rows(record, trace)
    assert tasks == []
    assert events[0]["duration_us"] == 7


def test_isolated_denominator_is_joined_by_mode_policy_seed_repeat_and_job(tmp_path):
    path = tmp_path / "jobs.csv"
    fields = ("group", "policy", "epoch", "repeat", "job_id", "jct_s", "status")
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"group": "new", "policy": "fifo", "epoch": 1, "repeat": 0,
                         "job_id": "job-0", "jct_s": 2.0, "status": "ok"})
    import hashlib
    shared_path = EXPERIMENTS / "baseline/L0-balanced.json"
    isolated_path = EXPERIMENTS / "isolated/job-0.json"
    environment = {"cpu_model": "test", "cpu_affinity": [0], "thread_environment": {},
                   "torch": "test", "platform": "test"}
    config = {"backend": "gloo", "world_size": 2, "compute_jitter": 0.3,
              "wait_budget_s": 0.02, "poll_interval": 0.001, "warmup_iterations": 1,
              "repeats": 1, "epochs": [1]}
    shared_manifest = {"config": {**config, "workload": str(shared_path), "dag": None},
                       "environment": environment, "profile_digest": "same",
                       "source_snapshot": {"digest": "same"},
                       "workload_digest": hashlib.sha256(shared_path.read_bytes()).hexdigest()}
    isolated_manifest = {**shared_manifest,
                         "config": {**config, "workload": str(isolated_path)},
                         "workload_digest": hashlib.sha256(isolated_path.read_bytes()).hexdigest()}
    (tmp_path / "manifest.json").write_text(json.dumps(isolated_manifest))
    rows = [{"group": "new", "policy": "fifo", "epoch": 1, "repeat": 0,
             "job_id": "job-0", "jct_s": 3.0, "isolated_jct_s": None, "slowdown": None}]
    _attach_isolated(rows, [path], shared_manifest)
    assert rows[0]["isolated_jct_s"] == 2.0
    assert rows[0]["slowdown"] == 1.5
    isolated_manifest["config"]["compute_jitter"] = 0.0
    (tmp_path / "manifest.json").write_text(json.dumps(isolated_manifest))
    with pytest.raises(ValueError, match="compute_jitter"):
        _attach_isolated(rows, [path], shared_manifest)


def test_g0_bridge_reuses_linear_compute_samples():
    dag = load_dag(EXPERIMENTS / "bridge/G0-linear-bridge.json", world_size=2)
    linear = load_workload(EXPERIMENTS / "bridge/L0-no-overlap-bridge.json")
    assert dag.seed == linear.seed
    class Recorder:
        def record(self, *args, **kwargs):
            pass
    class Stop:
        def wait(self, duration):
            self.duration = duration
    for job in dag.graph.jobs:
        samples = {}
        compute = make_replay_compute(job, dag.execution, seed=dag.seed, epoch=201,
                                      rank=1, jitter=0.3, samples=samples, event_log=Recorder())
        for index in range(2):
            node = next(node for node in job.nodes if node.node_id == f"c{index}")
            stop = Stop()
            compute(node, stop)
            expected = linear_execution_duration(linear, 201, job.job_id, index, 1,
                                                 "producer", 0.004, 0.3)
            assert stop.duration == pytest.approx(expected)
            assert samples[node.node_id] == pytest.approx(expected)


def test_interleaved_static_order_only_applies_to_static_ltf_arm(tmp_path):
    dag_path = EXPERIMENTS / "bridge/G0-linear-bridge.json"
    order_path = EXPERIMENTS / "bridge/G0-interleaved-order.json"
    args = Namespace(dag=dag_path, workload=None, backend="gloo", world_size=2,
                     timeout=20, poll_interval=0.001, compute_jitter=0.0,
                     wait_budget_s=0.02, warmup_iterations=1, comm_profile=None,
                     static_ltf_order=order_path)
    static_command = _command(args, "static_ltf", 0, tmp_path / "static.json")
    fifo_command = _command(args, "static_fifo", 0, tmp_path / "fifo.json")
    assert static_command[-2:] == ["--static-order", str(order_path)]
    assert "--static-order" not in fifo_command


@pytest.mark.parametrize("scenario,gate,candidates", [
    ("L2-tail-asymmetry", "job-gate/comm-0", ("job-0/comm-0", "job-1/comm-0")),
    ("G2-multi-frontier", "job-gate/occupy", ("job-0/comm-a", "job-0/comm-b")),
])
def test_gate_mechanism_requires_named_candidates_before_completion(scenario, gate, candidates):
    record = {"run_id": "run", "config": {"workload": scenario + ".json", "policy": "ltf"}}
    trace = [
        {"kind": "decision", "decision": "dispatch", "task_id": gate, "now": 1.0},
        {"kind": "offer", "task_id": candidates[0], "endpoint": 0, "now": 2.0},
        {"kind": "offer", "task_id": candidates[0], "endpoint": 1, "now": 2.1},
        {"kind": "offer", "task_id": candidates[1], "endpoint": 0, "now": 3.0},
        {"kind": "offer", "task_id": candidates[1], "endpoint": 1, "now": 3.1},
        {"kind": "completed", "task_id": gate, "now": 4.0},
        {"kind": "decision", "decision": "dispatch", "task_id": candidates[0],
         "eligible": list(candidates), "now": 4.1},
    ]
    result = {"validation": {"status": "ok"}, "ranks": [{"decision_records": trace}]}
    assert _mechanism_row(record, result)["mechanism_triggered"] is True
    trace[4]["now"] = 4.05
    assert _mechanism_row(record, result)["mechanism_triggered"] is False


def test_controlled_priority_choice_uses_identical_candidate_snapshot():
    linear = load_workload(EXPERIMENTS / "priority/L2-tail-asymmetry.json")
    tails = {job.job_id: sum(comm.producer_compute_s + comm.consumer_compute_s
                              + comm.estimated_comm_s for comm in job.communications[1:])
             for job in linear.jobs if job.job_id != "job-gate"}
    dag = load_dag(EXPERIMENTS / "priority/G2-multi-frontier.json", world_size=2)
    dag_tails = compute_tails(dag.graph)
    for candidates in (
        (Candidate("job-1/comm-0", 0.003, tails["job-1"], 1),
         Candidate("job-0/comm-0", 0.003, tails["job-0"], 2)),
        (Candidate("job-0/comm-b", 0.003, dag_tails["job-0/comm-b"], 1),
         Candidate("job-0/comm-a", 0.003, dag_tails["job-0/comm-a"], 2)),
    ):
        assert select_fifo(candidates).task_id == candidates[0].task_id
        assert select_ltf(candidates).task_id == candidates[1].task_id


def test_g4_mechanism_only_flags_unsafe_frontier_before_current_completion():
    record = {"run_id": "run", "config": {"workload": "G4-prediction-frontier.json",
                                                "policy": "lookahead"}}
    result = {"validation": {"status": "ok"}, "ranks": [{
        "decision_records": [{
            "kind": "policy_snapshot", "now": 12.0,
            "anticipated": [{"task_id": "job-1/unsafe"}],
        }],
        "dag_events": [{"kind": "comm_completed_observed", "task_id": "job-1/current",
                         "time_us": 12_100_000}],
    }]}
    assert _mechanism_row(record, result)["unsafe_frontier_anticipated"] is True

    result["ranks"][0]["dag_events"][0]["time_us"] = 11_900_000
    assert _mechanism_row(record, result)["unsafe_frontier_anticipated"] is False


def test_l4_requires_actual_deadline_fallback_dispatch():
    record = {"run_id": "run", "config": {"workload": "L4-late-prediction.json",
                                                "policy": "lookahead"}}
    trace = [{"kind": "decision", "decision": "wait", "target": "job-0/comm-0",
              "deadline": 2.0, "now": 1.0},
             {"kind": "lookahead_deadline", "now": 2.0},
             {"kind": "decision", "decision": "dispatch", "task_id": "job-1/comm-0",
              "reason": "dynamic_ltf", "now": 2.0}]
    result = {"validation": {"status": "ok"}, "ranks": [{"decision_records": trace}]}
    assert _mechanism_row(record, result)["mechanism_triggered"] is False
    trace[-1]["reason"] = "LOOKAHEAD_DEADLINE_FALLBACK"
    assert _mechanism_row(record, result)["mechanism_triggered"] is True


def test_compact_suite_budget_matches_selected_arms():
    suite = _suite(EXPERIMENTS / "suites/compact.json")
    assert suite["expected_replays"] == {
        "main": 660, "mechanism": 90, "isolated": 30,
        "base_total": 780, "lookahead_optional": 120, "with_optional": 900,
    }
    assert suite["sections"]["main"]["cases"][0]["arms"] == [
        "old-bare-fifo", "old-fifo", "old-ltf", "new-static_fifo", "new-static_ltf",
        "new-fifo", "new-ltf",
    ]


def test_batch_resume_skips_verified_results_and_preserves_failed_attempt(tmp_path, monkeypatch):
    calls = []
    original_run = subprocess.run

    def fake_run(command, **kwargs):
        if "--output" not in command:
            return original_run(command, **kwargs)
        calls.append(command)
        output = Path(command[command.index("--output") + 1])
        payload = {"config": {"policy": command[command.index("--policy") + 1],
                              "epoch": int(command[command.index("--epoch") + 1]),
                                  "compute_jitter": float(command[command.index("--compute-jitter") + 1]),
                                  "binding_preparation": command[command.index("--binding-preparation") + 1],
                                  "observation_mode": command[command.index("--observation-mode") + 1]},
                   "validation": {"status": "ok"},
                   "performance": {"workload_makespan_us": 1000, "job_makespans": []},
                   "ranks": []}
        output.write_text(json.dumps(payload))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(experiment_batch.subprocess, "run", fake_run)
    destination = tmp_path / "batch"
    argv = ["--output-dir", str(destination), "--workload", "balanced",
            "--arms", "new-fifo,new-ltf", "--baseline", "new-fifo",
            "--seeds", "5", "--repeats", "1", "--bootstrap-samples", "0"]
    assert experiment_batch.main(argv) == 0
    assert len(calls) == 2
    assert experiment_batch.main([*argv, "--resume"]) == 0
    assert len(calls) == 2
    records = experiment_batch._latest_records(destination / "runs.jsonl")
    failed_path = Path(next(iter(records.values()))["result_path"])
    failed_path.unlink()
    assert experiment_batch.main([*argv, "--resume"]) == 0
    assert len(calls) == 3
    with (destination / "runs.jsonl").open() as input_file:
        attempts = [json.loads(line) for line in input_file]
    assert len(attempts) == 3
    assert attempts[-1]["attempt"] == 2
    assert attempts[-1]["retry_of"] == str(failed_path)
    assert Path(attempts[-1]["result_path"]).exists()
    with pytest.raises(SystemExit):
        experiment_batch.main([*argv, "--resume", "--compute-jitter", "0.3"])
    assert len(calls) == 3


def test_compact_suite_rejects_mechanism_expansion_without_pre_registered_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(compact_suite, "RESULTS_ROOT", tmp_path)
    suite_id = "pilot"
    for scenario in ("L2", "L3"):
        directory = compact_suite._case_batch_dir(scenario, suite_id, "mechanism")
        directory.mkdir(parents=True)
        rows = []
        for policy in (("fifo", "ltf") if scenario == "L2" else ("lookahead",)):
            for index in range(5):
                rows.append({"run_id": f"{index}-new-{policy}", "status": "ok",
                             "mechanism_triggered": str(index < (3 if scenario == "L2" else 4))})
        with (directory / "mechanisms.csv").open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
    _check_mechanism_evidence(suite_id, {"id": "L2", "requires_mechanism_case": "L2",
                                         "minimum_triggered_per_arm": 3})
    _check_lookahead_evidence(suite_id, "L3", 4)
    with pytest.raises(ValueError, match="require 5/5"):
        _check_lookahead_evidence(suite_id, "L3", 5)


def test_compact_profile_requires_recorded_affinity_and_thread_environment(tmp_path):
    profile = _profile(1048576)
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile.to_dict()))
    with pytest.raises(ValueError, match="affinity differs or is unrecorded"):
        _validate_profile(path, [{"workload": "linear/L0-balanced.json"}], {0, 1},
                          {"OMP_NUM_THREADS": "1"})


def test_isolated_diagnostic_runs_thirty_interleaved_conditions_and_resumes(tmp_path, monkeypatch):
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.setenv(key, "1")
    monkeypatch.setattr(compact_suite, "_validate_profile", lambda *args: None)
    calls = []
    original_run = subprocess.run

    def fake_run(command, **kwargs):
        if "--output" not in command:
            return original_run(command, **kwargs)
        calls.append(command)
        workload = load_workload(command[command.index("--workload") + 1])
        output = Path(command[command.index("--output") + 1])
        output.write_text(json.dumps({
            "config": {"policy": command[command.index("--policy") + 1],
                       "epoch": int(command[command.index("--epoch") + 1]),
                       "compute_jitter": float(command[command.index("--compute-jitter") + 1])},
            "validation": {"status": "ok"},
            "performance": {"workload_makespan_us": 2000, "job_makespans": [
                {"job_id": job.job_id, "makespan_us": 2000} for job in workload.jobs]},
            "ranks": [],
        }))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(isolated_batch.subprocess, "run", fake_run)
    profile = tmp_path / "profile.json"
    profile.write_text("{}")
    output = tmp_path / "isolated"
    argv = ["--output-dir", str(output), "--comm-profile", str(profile)]
    assert isolated_batch.main(argv) == 0
    assert len(calls) == 30
    assert isolated_batch.main([*argv, "--resume"]) == 0
    assert len(calls) == 30
