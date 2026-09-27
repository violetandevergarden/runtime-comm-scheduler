"""Opt-in two-rank Gloo replay coverage for the independent runtime."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


LINEAR_CASES = (
    ("fifo", "balanced", None, 0.0),
    ("static_fifo", "delayed", None, 0.0),
    ("static_ltf", "tail", None, 0.0),
    ("ltf", "tail", None, 0.0),
    ("lookahead", "delayed", None, 0.0),
)
POLICIES = ("fifo", "static_fifo", "static_ltf", "ltf", "lookahead")
DAG_CASES = tuple((policy, None, name, 0.0)
                  for name in ("linear", "diamond", "multi-group") for policy in POLICIES)
CASES = LINEAR_CASES + DAG_CASES + (("fifo", None, "multi-group", 0.8),)


def _run_dag_replay(root, policy, dag_name, output, *, compute_jitter=0.0, static_order=None):
    env = dict(os.environ)
    src = str(root / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    command = [
        sys.executable,
        "-m", "examples.jobpacer.scripts.run_phase3",
        "--policy", policy,
        "--backend", "gloo",
        "--world-size", "2",
        "--timeout", "20",
        "--epoch", "7",
        "--compute-jitter", str(compute_jitter),
        "--output", str(output),
        "--dag", str(root / "benchmark/phase3/experiments/dag-semantics/smoke" / f"{dag_name}.json"),
    ]
    if static_order is not None:
        command.extend(("--static-order", str(static_order)))
    completed = subprocess.run(
        command,
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=35,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout[-4000:]
    return json.loads(output.read_text())


def _decision_records(payload):
    return next(rank["decision_records"] for rank in payload["ranks"] if rank.get("decision_records"))


@pytest.mark.parametrize(("policy", "workload", "dag_name", "compute_jitter"), CASES)
def test_two_rank_runtime_replay(policy, workload, dag_name, compute_jitter, tmp_path):
    if os.environ.get("RUN_JOBPACER_RUNTIME_REPLAY") != "1":
        pytest.skip("set RUN_JOBPACER_RUNTIME_REPLAY=1 to run the local TCP replay")
    pytest.importorskip("torch")

    root = Path(__file__).resolve().parents[2]
    output = tmp_path / f"{policy}-{workload or dag_name}.json"
    if dag_name:
        payload = _run_dag_replay(root, policy, dag_name, output, compute_jitter=compute_jitter)
    else:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
        command = [sys.executable, "-m", "examples.jobpacer.scripts.run_phase3",
                   "--policy", policy, "--workload", workload, "--backend", "gloo",
                   "--world-size", "2", "--timeout", "20", "--epoch", "7",
                   "--compute-jitter", str(compute_jitter), "--output", str(output)]
        completed = subprocess.run(command, cwd=root, env=env, capture_output=True,
                                   text=True, timeout=35)
        assert completed.returncode == 0, completed.stderr + completed.stdout[-4000:]
        payload = json.loads(output.read_text())
    assert payload["validation"]["status"] == "ok"
    assert payload["validation"]["rank_count"] == 2
    assert payload["validation"]["all_collectives_correct"] is True
    assert payload["validation"]["errors"] == []
    assert payload["config"]["code_revision"]
    assert isinstance(payload["config"]["working_tree_dirty"], bool)
    if dag_name:
        assert all(rank["manifest_digest"] for rank in payload["ranks"])
        assert all(rank["dag_events"] for rank in payload["ranks"])
        assert all(rank["canonical_dag"]["name"] == dag_name for rank in payload["ranks"])
        assert all(rank["dag_tails_s"] for rank in payload["ranks"])
        assert "dag_node_ready_wait_s" in payload["metrics"]
    if dag_name == "multi-group":
        assert payload["ranks"][0]["dag_tails_s"]["job-0/comm-a0"] == pytest.approx(0.021)
        assert payload["ranks"][0]["dag_tails_s"]["job-0/comm-b0"] == pytest.approx(0.003)
    if dag_name and policy != "lookahead":
        assert not any(event["kind"] == "comm_declared"
                       for rank in payload["ranks"] for event in rank["dag_events"])
    if policy == "static_fifo" and dag_name == "multi-group":
        assert payload["metrics"]["coordinator_idle_s"].get("STATIC_HEAD_BLOCKED", 0) > 0
    if policy == "static_ltf" and dag_name == "multi-group":
        dispatches = [record["task_id"] for record in _decision_records(payload)
                      if record.get("kind") == "decision" and record.get("decision") == "dispatch"]
        assert dispatches[0] == "job-1/comm-c0"
    if policy == "lookahead" and dag_name == "multi-group":
        target = "job-1/comm-c0"
        for rank in payload["ranks"]:
            declaration = next(event for event in rank["dag_events"]
                               if event.get("kind") == "comm_declared" and event.get("task_id") == target)
            ready = next(event for event in rank["dag_events"]
                         if event.get("kind") == "node_ready" and event.get("job_id") == "job-1"
                         and event.get("node_id") == "comm-c0")
            assert declaration["ready_after_s"] > 0
            assert declaration["predicted_ready_at_us"] > declaration["prediction_base_us"]
            assert payload["metrics"]["predicted_ready_error_s"][str(rank["rank"])][target] == pytest.approx(
                (ready["time_us"] - declaration["predicted_ready_at_us"]) / 1_000_000.0
            )
    if dag_name == "multi-group" and compute_jitter:
        samples = [next(job["compute_samples_s"]["skew"] for job in rank["jobs"] if job["job_id"] == "job-1")
                   for rank in payload["ranks"]]
        assert samples[0] != samples[1]


def test_static_ltf_loads_and_executes_external_order(tmp_path):
    if os.environ.get("RUN_JOBPACER_RUNTIME_REPLAY") != "1":
        pytest.skip("set RUN_JOBPACER_RUNTIME_REPLAY=1 to run the local TCP replay")
    pytest.importorskip("torch")
    root = Path(__file__).resolve().parents[2]
    order = ["job-0/comm-b0", "job-0/comm-a0", "job-0/comm-a1",
             "job-0/comm-b1", "job-1/comm-c0", "job-1/comm-c1"]
    order_file = tmp_path / "static-ltf-order.json"
    order_file.write_text(json.dumps(order))
    payload = _run_dag_replay(root, "static_ltf", "multi-group",
                              tmp_path / "static-ltf-external.json", static_order=order_file)
    dispatches = [record["task_id"] for record in _decision_records(payload)
                  if record.get("kind") == "decision" and record.get("decision") == "dispatch"]
    assert dispatches[0] == order[0]


@pytest.mark.parametrize("declaration_mode", ("before-producer", "on-submit"))
def test_linear_metadata_mismatch_fails_before_launch(declaration_mode, tmp_path):
    if os.environ.get("RUN_JOBPACER_RUNTIME_REPLAY") != "1":
        pytest.skip("set RUN_JOBPACER_RUNTIME_REPLAY=1 to run the local TCP replay")
    pytest.importorskip("torch")
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / f"metadata-mismatch-{declaration_mode}.json"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(root / "src"), str(root), env.get("PYTHONPATH", "")))
    command = [sys.executable, "-m", "examples.jobpacer.scripts.run_phase3",
               "--policy", "fifo", "--workload", "balanced", "--backend", "gloo",
               "--world-size", "2", "--timeout", "3", "--declaration-mode", declaration_mode,
               "--fault", "metadata_mismatch", "--output", str(output)]
    completed = subprocess.run(command, cwd=root, env=env, capture_output=True,
                               text=True, timeout=15)
    assert completed.returncode != 0
    payload = json.loads(output.read_text())
    assert payload["validation"]["status"] == "failed"
    assert payload["validation"]["errors"]


@pytest.mark.parametrize("declaration_mode", ("before-producer", "on-submit"))
def test_linear_missing_task_is_bounded_without_declaration(declaration_mode, tmp_path):
    if os.environ.get("RUN_JOBPACER_RUNTIME_REPLAY") != "1":
        pytest.skip("set RUN_JOBPACER_RUNTIME_REPLAY=1 to run the local TCP replay")
    pytest.importorskip("torch")
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / f"missing-task-{declaration_mode}.json"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(root / "src"), str(root), env.get("PYTHONPATH", "")))
    command = [sys.executable, "-m", "examples.jobpacer.scripts.run_phase3",
               "--policy", "static_fifo", "--workload", "balanced", "--backend", "gloo",
               "--world-size", "2", "--setup-timeout", "5", "--timeout", "3",
               "--declaration-mode", declaration_mode, "--fault", "missing_task",
               "--output", str(output)]
    completed = subprocess.run(command, cwd=root, env=env, capture_output=True,
                               text=True, timeout=15)
    assert completed.returncode != 0
    payload = json.loads(output.read_text())
    assert payload["validation"]["status"] == "failed"
    assert payload["validation"]["errors"]


@pytest.mark.parametrize("declaration_mode", ("before-producer", "on-submit"))
@pytest.mark.parametrize("fault", ("launch_failure", "completion_probe_failure"))
def test_linear_launch_and_probe_failures_are_bounded(fault, declaration_mode, tmp_path):
    if os.environ.get("RUN_JOBPACER_RUNTIME_REPLAY") != "1":
        pytest.skip("set RUN_JOBPACER_RUNTIME_REPLAY=1 to run the local TCP replay")
    pytest.importorskip("torch")
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / f"linear-{fault}.json"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    command = [sys.executable, "-m", "examples.jobpacer.scripts.run_phase3",
               "--policy", "fifo", "--workload", "balanced", "--backend", "gloo",
               "--world-size", "2", "--setup-timeout", "5", "--timeout", "3",
               "--declaration-mode", declaration_mode, "--fault", fault,
               "--output", str(output)]
    completed = subprocess.run(command, cwd=root, env=env, capture_output=True,
                               text=True, timeout=15)
    assert completed.returncode != 0
    payload = json.loads(output.read_text())
    assert payload["validation"]["status"] == "failed"
    assert payload["validation"]["errors"]


def test_declaration_modes_preserve_linear_specs_hints_order_and_readiness(tmp_path):
    if os.environ.get("RUN_JOBPACER_RUNTIME_REPLAY") != "1":
        pytest.skip("set RUN_JOBPACER_RUNTIME_REPLAY=1 to run the local TCP replay")
    pytest.importorskip("torch")
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(root / "src"), str(root), env.get("PYTHONPATH", "")))
    payloads = {}
    for declaration_mode in ("before-producer", "on-submit"):
        output = tmp_path / f"declaration-{declaration_mode}.json"
        command = [sys.executable, "-m", "examples.jobpacer.scripts.run_phase3",
                   "--policy", "static_fifo", "--workload", "delayed", "--backend", "gloo",
                   "--world-size", "2", "--epoch", "85", "--binding-preparation", "precreate",
                   "--declaration-mode", declaration_mode, "--observation-mode", "diagnostic",
                   "--timeout", "20", "--output", str(output)]
        completed = subprocess.run(command, cwd=root, env=env, capture_output=True,
                                   text=True, timeout=35)
        assert completed.returncode == 0, completed.stderr + completed.stdout[-4000:]
        payloads[declaration_mode] = json.loads(output.read_text())

    before = payloads["before-producer"]
    direct = payloads["on-submit"]
    for payload, mode in ((before, "before-producer"), (direct, "on-submit")):
        assert payload["validation"]["status"] == "ok"
        assert payload["validation"]["all_collectives_correct"] is True
        assert payload["config"]["declaration_mode"] == mode
        ranks = payload["ranks"]
        assert ranks[0]["launch_sequence"] == ranks[1]["launch_sequence"]
        assert all(rank["grant_sequence"] == rank["launch_sequence"] for rank in ranks)
        for rank in ranks:
            events_by_task = {}
            for event in rank["runtime_events"]:
                if event.get("task_id"):
                    events_by_task.setdefault(event["task_id"], []).append(event)
            tasks = [task for job in rank["jobs"] for task in job["tasks"]]
            assert len(tasks) == sum(len(job["tasks"]) for job in rank["jobs"])
            assert all(task["collective_spec"] and task["task_hint"] for task in tasks)
            for task in tasks:
                task_events = events_by_task[task["task_id"]]
                offer_start = next(event["time_us"] for event in task_events
                                   if event["kind"] == "offer_send_start")
                assert task["ready_ts"] <= offer_start
                declare_events = [event for event in task_events
                                  if event["kind"].startswith("declare_")
                                  or event["kind"] == "declared"]
                assert bool(declare_events) is (mode == "before-producer")
    normalized = lambda payload: [
        (rank["rank"], tuple((task["task_id"], task["group_id"], task["group_seq"],
                              task["collective_spec"], task["task_hint"])
                             for job in rank["jobs"] for task in job["tasks"]))
        for rank in payload["ranks"]
    ]
    assert normalized(before) == normalized(direct)


def test_on_submit_declaration_mode_rejects_lookahead_and_dag_before_launch(tmp_path):
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(root / "src"), str(root), env.get("PYTHONPATH", "")))
    common = [sys.executable, "-m", "examples.jobpacer.scripts.run_phase3",
              "--declaration-mode", "on-submit", "--backend", "gloo"]
    commands = [common + ["--policy", "lookahead", "--workload", "balanced"],
                common + ["--policy", "fifo", "--dag",
                          str(root / "benchmark/phase3/experiments/dag-semantics/smoke/linear.json")]]
    for command in commands:
        completed = subprocess.run(command, cwd=root, env=env, capture_output=True,
                                   text=True, timeout=10)
        assert completed.returncode == 2
        assert "linear non-Lookahead" in completed.stderr


@pytest.mark.parametrize("policy", ("fifo", "ltf", "static_fifo"))
@pytest.mark.parametrize("binding_preparation", ("precreate", "on-ready"))
def test_linear_binding_preparation_preserves_readiness_and_collective_semantics(
    policy, binding_preparation, tmp_path
):
    if os.environ.get("RUN_JOBPACER_RUNTIME_REPLAY") != "1":
        pytest.skip("set RUN_JOBPACER_RUNTIME_REPLAY=1 to run the local TCP replay")
    pytest.importorskip("torch")
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / f"linear-{policy}-{binding_preparation}.json"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(root / "src"), str(root), env.get("PYTHONPATH", "")))
    command = [sys.executable, "-m", "examples.jobpacer.scripts.run_phase3",
               "--policy", policy, "--workload", "balanced", "--backend", "gloo",
               "--world-size", "2", "--epoch", "83", "--compute-jitter", "0.3",
               "--binding-preparation", binding_preparation, "--timeout", "20",
               "--output", str(output)]
    completed = subprocess.run(command, cwd=root, env=env, capture_output=True,
                               text=True, timeout=35)
    assert completed.returncode == 0, completed.stderr + completed.stdout[-4000:]
    payload = json.loads(output.read_text())
    assert payload["validation"]["status"] == "ok"
    assert payload["validation"]["all_collectives_correct"] is True
    for rank in payload["ranks"]:
        assert rank["binding_preparation"] == binding_preparation
        assert rank["grant_sequence"] == rank["launch_sequence"]
        events = {event["task_id"]: event["time_us"] for event in rank["runtime_events"]
                  if event["kind"] == "offered"}
        tasks = [task for job in rank["jobs"] for task in job["tasks"]]
        assert set(events) == {task["task_id"] for task in tasks}
        assert all(task["ready_ts"] <= events[task["task_id"]] for task in tasks)
        if binding_preparation == "precreate":
            assert rank["preparation_end_ts"] < rank["application_release_ts"]
            assert all(event["binding_create_end_ts"] <= rank["preparation_end_ts"]
                       for event in rank["binding_creation_events"])
        else:
            assert rank["preparation_total_us"] is None
            assert all(task["ready_ts"] <= task["binding_create_start_ts"]
                       <= task["binding_create_end_ts"] <= task["submit_call_ts"]
                       for task in tasks)


@pytest.mark.parametrize("observation_mode", ("minimal", "diagnostic"))
def test_linear_observation_modes_preserve_collective_semantics(observation_mode, tmp_path):
    if os.environ.get("RUN_JOBPACER_RUNTIME_REPLAY") != "1":
        pytest.skip("set RUN_JOBPACER_RUNTIME_REPLAY=1 to run the local TCP replay")
    pytest.importorskip("torch")
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / f"observation-{observation_mode}.json"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(root / "src"), str(root), env.get("PYTHONPATH", "")))
    command = [sys.executable, "-m", "examples.jobpacer.scripts.run_phase3",
               "--policy", "static_fifo", "--workload", "balanced", "--backend", "gloo",
               "--world-size", "2", "--epoch", "84", "--binding-preparation", "precreate",
               "--observation-mode", observation_mode, "--timeout", "20", "--output", str(output)]
    completed = subprocess.run(command, cwd=root, env=env, capture_output=True,
                               text=True, timeout=35)
    assert completed.returncode == 0, completed.stderr + completed.stdout[-4000:]
    payload = json.loads(output.read_text())
    assert payload["validation"]["status"] == "ok"
    assert payload["validation"]["observation_mode"] == observation_mode
    assert payload["config"]["binding_preparation"] == "precreate"
    assert payload["validation"]["all_collectives_correct"] is True
    task_sequences = [rank["launch_sequence"] for rank in payload["ranks"]]
    assert task_sequences[0] == task_sequences[1]
    assert all(rank["grant_sequence"] == rank["launch_sequence"] for rank in payload["ranks"])
    coordinator = next(rank for rank in payload["ranks"] if rank["rank"] == 0)
    expected_reports = sum(len(job["tasks"]) for rank in payload["ranks"] for job in rank["jobs"])
    assert coordinator["protocol_transition_counts"] == {
        "submitted_reports": expected_reports,
        "completed_reports": expected_reports,
        "submitted_before_completed_enforced": True,
    }
    assert payload["validation"]["diagnostic_timestamps_present"] is (observation_mode == "diagnostic")
    if observation_mode == "minimal":
        assert all(rank["runtime_events"] == [] for rank in payload["ranks"])
        assert coordinator["coordinator_instrumentation"] == []
    else:
        for rank in payload["ranks"]:
            assert any(event["kind"] == "offer_send_start" for event in rank["runtime_events"])
            assert any(event["kind"] == "collective_call_return" for event in rank["runtime_events"])
        assert any(event["kind"] == "event_processing_start"
                   for event in coordinator["coordinator_instrumentation"])


@pytest.mark.parametrize("binding_preparation", ("precreate", "on-ready"))
def test_linear_binding_preparation_failures_exit_bounded(binding_preparation, tmp_path):
    if os.environ.get("RUN_JOBPACER_RUNTIME_REPLAY") != "1":
        pytest.skip("set RUN_JOBPACER_RUNTIME_REPLAY=1 to run the local TCP replay")
    pytest.importorskip("torch")
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / f"binding-failure-{binding_preparation}.json"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(root / "src"), str(root), env.get("PYTHONPATH", "")))
    command = [sys.executable, "-m", "examples.jobpacer.scripts.run_phase3",
               "--policy", "ltf", "--workload", "balanced", "--backend", "gloo",
               "--world-size", "2", "--binding-preparation", binding_preparation,
               "--setup-timeout", "5", "--timeout", "3", "--fault", "binding_failure",
               "--output", str(output)]
    completed = subprocess.run(command, cwd=root, env=env, capture_output=True,
                               text=True, timeout=15)
    assert completed.returncode != 0
    payload = json.loads(output.read_text())
    assert payload["validation"]["status"] == "failed"
    assert any("binding failure" in error for error in payload["validation"]["errors"])


@pytest.mark.parametrize("fault", ("compute_failure", "binding_failure", "missing_task"))
def test_dag_failures_are_bounded(fault, tmp_path):
    if os.environ.get("RUN_JOBPACER_RUNTIME_REPLAY") != "1":
        pytest.skip("set RUN_JOBPACER_RUNTIME_REPLAY=1 to run the local TCP replay")
    pytest.importorskip("torch")
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / f"failure-{fault}.json"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    command = [sys.executable, "-m", "examples.jobpacer.scripts.run_phase3",
               "--policy", "fifo", "--dag", str(root / "benchmark/phase3/experiments/dag-semantics/smoke/diamond.json"),
               "--backend", "gloo", "--world-size", "2", "--timeout", "3",
               "--fault", fault, "--output", str(output)]
    completed = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, timeout=15)
    assert completed.returncode != 0
    payload = json.loads(output.read_text())
    assert payload["validation"]["status"] == "failed"
    assert payload["validation"]["errors"]


def test_invalid_dag_manifest_is_rejected_before_rank_workers(tmp_path):
    root = Path(__file__).resolve().parents[2]
    invalid = tmp_path / "invalid.json"
    data = json.loads((root / "benchmark/phase3/experiments/dag-semantics/smoke/linear.json").read_text())
    data["schema_version"] = 99
    invalid.write_text(json.dumps(data))
    command = [sys.executable, "-m", "examples.jobpacer.scripts.run_phase3",
               "--dag", str(invalid), "--world-size", "2"]
    completed = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=5)
    assert completed.returncode == 2
    assert "schema_version" in completed.stderr


def test_unsupported_reduction_is_rejected_before_rank_workers(tmp_path):
    root = Path(__file__).resolve().parents[2]
    invalid = tmp_path / "unsupported-reduction.json"
    data = json.loads((root / "benchmark/phase3/experiments/dag-semantics/smoke/linear.json").read_text())
    comm = next(node for job in data["jobs"] for node in job["nodes"] if node["kind"] == "comm")
    comm["collective"]["reduction"] = "max"
    invalid.write_text(json.dumps(data))
    output = tmp_path / "should-not-start.json"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    command = [sys.executable, "-m", "examples.jobpacer.scripts.run_phase3",
               "--dag", str(invalid), "--world-size", "2", "--output", str(output)]
    completed = subprocess.run(command, cwd=root, env=env, capture_output=True,
                               text=True, timeout=5)
    assert completed.returncode == 2
    assert "reduction" in completed.stderr and "unsupported" in completed.stderr
    assert not output.exists()
