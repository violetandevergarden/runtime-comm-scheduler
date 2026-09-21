"""Two-rank Gloo profiling and replay acceptance path."""

from __future__ import annotations

import json
import socket
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]


def _require_local_socket() -> None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM):
            pass
    except PermissionError:
        pytest.skip("sandbox does not permit local Gloo sockets")


def test_profile_then_replay_two_message_sizes(tmp_path):
    _require_local_socket()
    manifest = tmp_path / "workload.json"
    profile_path = tmp_path / "profile.json"
    trace_path = tmp_path / "trace.json"
    manifest.write_text(json.dumps({
        "name": "profile-integration",
        "jobs": [
            {"job_id": "a", "communications": [
                {"id": 0, "num_bytes": 4096},
                {"id": 1, "num_bytes": 1048576},
            ]},
            {"job_id": "b", "communications": [
                {"id": 0, "num_bytes": 4096},
            ]},
        ],
    }))
    subprocess.run([
        sys.executable, "examples/jobpacer/profile_communication.py",
        "--workload", str(manifest), "--backend", "gloo", "--world-size", "2",
        "--warmup", "1", "--iterations", "3", "--timeout", "20",
        "--output", str(profile_path),
    ], cwd=ROOT, check=True, timeout=30)
    profile = json.loads(profile_path.read_text())
    assert len(profile["records"]) == 2
    assert all(record["samples"] == 3 and record["p50_s"] > 0 for record in profile["records"])
    assert all(record["p10_s"] <= record["p50_s"] <= record["p90_s"] for record in profile["records"])

    subprocess.run([
        sys.executable, "examples/jobpacer/run_replay.py", "--mode", "scheduler",
        "--policy", "ltf", "--workload", str(manifest), "--backend", "gloo",
        "--world-size", "2", "--max-outstanding", "1", "--timeout", "20",
        "--comm-profile", str(profile_path), "--output", str(trace_path),
    ], cwd=ROOT, check=True, timeout=30)
    trace = json.loads(trace_path.read_text())
    assert trace["validation"]["status"] == "ok"
    assert len({rank["workload_digest"] for rank in trace["ranks"]}) == 1
    assert all(
        task["estimate_source"] == "offline_profile" and task["estimated_comm_s"] > 0
        for rank in trace["ranks"] for job in rank["jobs"] for task in job["tasks"]
    )

    observed_parallelism = False
    for policy, capacity in (("fifo", 2), ("ltf", 3), ("srjf", 1), ("srjf", 0)):
        output = tmp_path / f"trace-{policy}-{capacity}.json"
        subprocess.run([
            sys.executable, "examples/jobpacer/run_replay.py", "--mode", "scheduler",
            "--policy", policy, "--workload", str(manifest), "--backend", "gloo",
            "--world-size", "2", "--max-outstanding", str(capacity), "--timeout", "20",
            "--comm-profile", str(profile_path), "--output", str(output),
        ], cwd=ROOT, check=True, timeout=30)
        replay = json.loads(output.read_text())
        assert replay["validation"]["status"] == "ok"
        assert len({rank["plan"]["digest"] for rank in replay["ranks"]}) == 1
        observations = replay["performance"]["capacity_observations"]
        if capacity:
            assert all(item["peak_admission_occupancy"] <= capacity for item in observations)
        if capacity > 1:
            observed_parallelism |= any(item["peak_admission_occupancy"] > 1 for item in observations)
    assert observed_parallelism, "the two-rank fixture must actually observe k>1 admission overlap"
