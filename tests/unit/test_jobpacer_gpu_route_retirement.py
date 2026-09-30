from __future__ import annotations

import sys
from pathlib import Path

import pytest

from examples.jobpacer.runtime import replay_worker
from examples.jobpacer.scripts import run_comm_profile
from examples.jobpacer.scripts import run_phase2


def test_phase2_cli_rejects_nccl_for_linear_workload():
    with pytest.raises(SystemExit) as error:
        run_phase2.main(["--backend", "nccl"])
    assert error.value.code == 2


def test_legacy_replay_worker_cli_rejects_nccl(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["replay_worker", "--mode", "bare", "--backend", "nccl"])
    with pytest.raises(SystemExit) as error:
        replay_worker.main()
    assert error.value.code == 2


def test_nccl_comm_profiler_rejects_schema1_dag_before_device_probe(monkeypatch, tmp_path):
    root = Path(__file__).parents[2]
    monkeypatch.setattr(sys, "argv", [
        "run_comm_profile", "--dag",
        str(root / "benchmark/phase3/experiments/dag-semantics/smoke/linear.json"),
        "--backend", "nccl", "--output", str(tmp_path / "profile.json"),
    ])
    with pytest.raises(SystemExit) as error:
        run_comm_profile.main()
    assert error.value.code == 2
