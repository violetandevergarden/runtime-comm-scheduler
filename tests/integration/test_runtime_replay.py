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
        str(root / "examples/jobpacer/run_runtime_replay.py"),
        "--policy", policy,
        "--backend", "gloo",
        "--world-size", "2",
        "--timeout", "20",
        "--epoch", "7",
        "--compute-jitter", str(compute_jitter),
        "--output", str(output),
        "--dag", str(root / "benchmark/phase3" / f"{dag_name}.json"),
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
        command = [sys.executable, str(root / "examples/jobpacer/run_runtime_replay.py"),
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


@pytest.mark.parametrize("fault", ("compute_failure", "binding_failure", "missing_task"))
def test_dag_failures_are_bounded(fault, tmp_path):
    if os.environ.get("RUN_JOBPACER_RUNTIME_REPLAY") != "1":
        pytest.skip("set RUN_JOBPACER_RUNTIME_REPLAY=1 to run the local TCP replay")
    pytest.importorskip("torch")
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / f"failure-{fault}.json"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    command = [sys.executable, str(root / "examples/jobpacer/run_runtime_replay.py"),
               "--policy", "fifo", "--dag", str(root / "benchmark/phase3/diamond.json"),
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
    data = json.loads((root / "benchmark/phase3/linear.json").read_text())
    data["schema_version"] = 99
    invalid.write_text(json.dumps(data))
    command = [sys.executable, str(root / "examples/jobpacer/run_runtime_replay.py"),
               "--dag", str(invalid), "--world-size", "2"]
    completed = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=5)
    assert completed.returncode == 2
    assert "schema_version" in completed.stderr
