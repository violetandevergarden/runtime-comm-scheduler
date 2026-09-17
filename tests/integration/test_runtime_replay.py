"""Opt-in two-rank Gloo replay coverage for the independent runtime."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


CASES = (
    ("fifo", "balanced"),
    ("static_fifo", "delayed"),
    ("ltf", "tail"),
    ("lookahead", "delayed"),
)


@pytest.mark.parametrize(("policy", "workload"), CASES)
def test_two_rank_runtime_replay(policy, workload, tmp_path):
    if os.environ.get("RUN_JOBPACER_RUNTIME_REPLAY") != "1":
        pytest.skip("set RUN_JOBPACER_RUNTIME_REPLAY=1 to run the local TCP replay")
    pytest.importorskip("torch")

    root = Path(__file__).resolve().parents[2]
    output = tmp_path / f"{policy}-{workload}.json"
    env = dict(os.environ)
    src = str(root / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    command = [
        sys.executable,
        str(root / "examples/jobpacer/run_runtime_replay.py"),
        "--policy", policy,
        "--workload", workload,
        "--backend", "gloo",
        "--world-size", "2",
        "--timeout", "20",
        "--output", str(output),
    ]
    completed = subprocess.run(
        command,
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=35,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout[-4000:]
    payload = json.loads(output.read_text())
    assert payload["validation"]["status"] == "ok"
    assert payload["validation"]["rank_count"] == 2
    assert payload["validation"]["all_collectives_correct"] is True
