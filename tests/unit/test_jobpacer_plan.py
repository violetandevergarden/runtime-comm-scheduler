"""Static JobPacer Plan order, dependency, and determinism checks."""

import sys

import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from examples.jobpacer.plan_builder import (
    build_plan,
    key_labels,
    policy_diagnostics,
    policy_names,
)
from examples.jobpacer.workloads import (
    CollectiveComm,
    Job,
    Workload,
    ranks_for_job,
)


def _workload() -> Workload:
    return Workload(
        name="unit",
        jobs=(
            Job(
                "job-a",
                (
                    CollectiveComm(0, consumer_compute_s=0.01),
                    CollectiveComm(1, producer_compute_s=0.01),
                    CollectiveComm(2),
                ),
            ),
            Job(
                "job-b",
                (
                    CollectiveComm(0),
                    CollectiveComm(1),
                ),
            ),
        ),
    )


def test_fifo_is_fixed_round_robin_and_preserves_job_order():
    plan = build_plan(_workload(), "fifo")
    assert key_labels(plan) == ["job-a:0", "job-b:0", "job-a:1", "job-b:1", "job-a:2"]
    assert plan.group_sequence("job-a") == tuple(key for key in plan.keys if key.process_group_id == "job-a")


def test_ltf_uses_tail_and_has_a_different_legal_interleaving():
    plan = build_plan(_workload(), "ltf")
    assert key_labels(plan) == ["job-a:0", "job-a:1", "job-b:0", "job-a:2", "job-b:1"]
    for job_id in ("job-a", "job-b"):
        assert [key.ordinal for key in plan.group_sequence(job_id)] == sorted(
            key.ordinal for key in plan.group_sequence(job_id)
        )


def test_plan_and_digest_are_deterministic():
    first = build_plan(_workload(), "ltf", version=4, window_id=9)
    second = build_plan(_workload(), "ltf", version=4, window_id=9)
    assert first == second
    assert first.digest() == second.digest()
    assert [key.as_list() for key in first.keys] == [key.as_list() for key in second.keys]


def test_srjf_picks_shortest_remaining_job_while_ltf_picks_longest():
    workload = Workload(
        "long-short",
        (
            Job("long", (CollectiveComm(0, estimated_comm_s=0.01),
                          CollectiveComm(1, producer_compute_s=0.01, estimated_comm_s=0.02))),
            Job("short", (CollectiveComm(0, estimated_comm_s=0.005),)),
        ),
    )
    assert key_labels(build_plan(workload, "ltf"))[0] == "long:0"
    assert key_labels(build_plan(workload, "srjf"))[0] == "short:0"


def test_srjf_ties_and_diagnostics_are_stable_across_job_input_order():
    jobs = (
        Job("job-b", (CollectiveComm(0, estimated_comm_s=0.01),)),
        Job("job-a", (CollectiveComm(0, estimated_comm_s=0.01),)),
    )
    first = Workload("ties", jobs)
    second = Workload("ties", tuple(reversed(jobs)))
    plan = build_plan(first, "srjf")
    assert key_labels(plan) == ["job-a:0", "job-b:0"]
    assert plan.digest() == build_plan(second, "srjf").digest()
    diagnostics = policy_diagnostics(first, "srjf")
    assert diagnostics["score_definition"] == "zero-admission-delay estimated remaining critical path"
    assert diagnostics["steps"][0]["sort_direction"] == "min"
    assert diagnostics["steps"][0]["selected_key"] == plan.keys[0].as_list()
    assert diagnostics["steps"][0]["selected_score"] == diagnostics["steps"][0]["candidates"][1]["score"]
    for job_id in ("job-a", "job-b"):
        assert [key.ordinal for key in plan.group_sequence(job_id)] == [0]


def test_policy_registry_exposes_static_algorithms():
    assert policy_names() == ("fifo", "ltf", "srjf")


def test_job_membership_accepts_arbitrary_global_ranks():
    workload = Workload(
        "members",
        (Job("job-a", (CollectiveComm(0),), ranks=(3, 0, 2)),),
    )
    assert ranks_for_job(workload.jobs[0], 4) == (3, 0, 2)
    restored = Workload.from_dict(workload.to_dict())
    assert restored.jobs[0].ranks == (3, 0, 2)


def test_workload_manifest_round_trip_preserves_phase1_compute_windows():
    workload = _workload()
    restored = Workload.from_dict(workload.to_dict())
    assert restored == workload
    communication = restored.jobs[0].communications[0]
    assert communication.producer_compute_s == 0.0
    assert communication.consumer_compute_s == 0.01


def test_job_rejects_nonconsecutive_communication_ids():
    with pytest.raises(ValueError, match="ordinals must be consecutive"):
        Job("bad", (CollectiveComm(1),))
