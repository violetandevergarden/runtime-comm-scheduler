"""Phase 1 entry-point and performance-summary tests."""

from __future__ import annotations

import argparse

import pytest

from examples.jobpacer import run_phase1
from examples.jobpacer.run_replay import _summarize_performance
from examples.jobpacer.workloads import built_workload


def _args(**overrides):
    values = {"workload": "balanced", "world_size": 2}
    values.update(overrides)
    return argparse.Namespace(**values)


def _rank_result(rank: int, makespans: tuple[int, int]) -> dict:
    workload = built_workload("balanced")
    return {
        "rank": rank,
        "replay_makespan_us": max(makespans) + 10,
        "jobs": [
            {
                "job_id": job.job_id,
                "makespan_us": makespans[index],
                "tasks": [{"ordinal": item.id} for item in job.communications],
            }
            for index, job in enumerate(workload.jobs)
        ],
    }


def test_phase1_entry_forces_bare_mode(monkeypatch):
    observed = []
    monkeypatch.setattr(run_phase1, "replay_main", lambda argv: observed.extend(argv) or 7)
    assert run_phase1.main(["--workload", "tail"]) == 7
    assert observed == ["--mode", "bare", "--workload", "tail"]


def test_phase1_entry_rejects_mode_override():
    with pytest.raises(ValueError, match="fixes --mode=bare"):
        run_phase1.main(["--mode", "scheduler"])


def test_performance_summary_uses_slowest_member_rank():
    summary = _summarize_performance(
        [_rank_result(0, (100, 300)), _rank_result(1, (120, 250))],
        _args(),
    )
    assert summary == {
        "job_makespans": [
            {
                "job_id": "job-0",
                "participating_ranks": [0, 1],
                "makespan_us": 120,
                "rank_makespans_us": [100, 120],
                "tasks_per_rank": 3,
            },
            {
                "job_id": "job-1",
                "participating_ranks": [0, 1],
                "makespan_us": 300,
                "rank_makespans_us": [300, 250],
                "tasks_per_rank": 3,
            },
        ],
        "workload_makespan_us": 310,
        "rank_replay_makespans_us": [310, 260],
    }


def test_performance_summary_rejects_missing_job_rank_record():
    result = _rank_result(0, (100, 200))
    with pytest.raises(ValueError, match="expected rank records"):
        _summarize_performance([result], _args())


def test_performance_summary_rejects_duplicate_job_rank_record():
    result = _rank_result(0, (100, 200))
    duplicate = _rank_result(0, (110, 210))
    with pytest.raises(ValueError, match="duplicate rank record"):
        _summarize_performance(
            [result, duplicate, _rank_result(1, (120, 220))], _args()
        )
