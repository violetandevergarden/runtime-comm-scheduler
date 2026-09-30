from __future__ import annotations

import sys

import pytest

from examples.jobpacer.runtime import replay_worker
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
