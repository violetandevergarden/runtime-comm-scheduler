from __future__ import annotations

import time
from types import SimpleNamespace

import torch

from examples.jobpacer.diagnostics import bare_nccl_mechanism as mechanism


def test_completion_serial_preserves_first_a_completion_observation(monkeypatch):
    class Receipt:
        def is_completed(self):
            return True

    class Executor:
        def launch(self, binding):
            binding.tensor.fill_(3.0)
            return Receipt()

    monkeypatch.setattr(
        mechanism, "_binding",
        lambda tensor, _group, _device: SimpleNamespace(tensor=tensor),
    )
    monkeypatch.setattr(
        mechanism, "_wait_receipt",
        lambda _receipt, *, deadline: time.perf_counter_ns(),
    )

    sample = mechanism._run_case(
        mode="completion-serial", repeat=0, rank=0, numel=1,
        groups={"A": object(), "B": object()}, executor=Executor(),
        device=torch.device("cpu"), timeout_s=1.0,
    )
    by_name = {row["collective"]: row for row in sample["api_calls"]}

    assert by_name["A"]["physical_complete_observed_ns"] <= by_name["B"]["api_start_ns"]
    assert by_name["B"]["physical_complete_observed_ns"] >= by_name["B"]["api_start_ns"]
    assert sample["numeric_checks"] == {"A": True, "B": True}


def test_late_observation_alone_does_not_qualify_multi_inflight():
    from examples.jobpacer.experiments.seven_arm.qualify import _mechanism_checks
    sample = {"mode": "multi-inflight", "api_calls": [
        {"collective": "A", "api_start_ns": 1, "physical_complete_observed_ns": 10},
        {"collective": "B", "api_start_ns": 5, "prior_pending_at_launch_probe": []},
    ]}
    raw = {"ranks": [{"rank": rank, "samples": [sample]} for rank in (0, 1)]}
    assert not _mechanism_checks(raw)["launch_before_completion_observation"]
    sample["api_calls"][1]["prior_pending_at_launch_probe"] = ["A"]
    assert _mechanism_checks(raw)["launch_before_completion_observation"]
