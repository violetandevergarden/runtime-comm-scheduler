"""Checks for the explicit Phase 1.2 experiment matrices."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "benchmark/phase1.2"))

from batch_runner import scenario_matrix


def test_capacity_scan_matrix_has_bare_and_eight_static_capacity_scenarios():
    scenes = scenario_matrix("capacity-scan")
    assert list(scenes) == [
        "phase1_bare",
        "fifo_k1", "fifo_k2", "fifo_k3", "fifo_unbounded",
        "ltf_k1", "ltf_k2", "ltf_k3", "ltf_unbounded",
    ]
    assert scenes["phase1_bare"]["mode"] == "bare"
    assert scenes["fifo_k2"]["max_outstanding"] == 2
    assert scenes["ltf_unbounded"]["max_outstanding"] == 0


def test_srjf_matrix_uses_same_frozen_capacities_for_all_policies():
    scenes = scenario_matrix("srjf", [1, 2, 0])
    assert list(scenes) == [
        "phase1_bare",
        "fifo_k1", "ltf_k1", "srjf_k1",
        "fifo_k2", "ltf_k2", "srjf_k2",
        "fifo_unbounded", "ltf_unbounded", "srjf_unbounded",
    ]
    assert {item["max_outstanding"] for name, item in scenes.items() if name.endswith("_k2")} == {2}


def test_srjf_requires_k1_and_unbounded_and_rejects_unknown_capacity():
    with pytest.raises(ValueError, match="include k1 and unbounded"):
        scenario_matrix("srjf", [2, 3])
    with pytest.raises(ValueError, match="unique values"):
        scenario_matrix("srjf", [1, 4, 0])
