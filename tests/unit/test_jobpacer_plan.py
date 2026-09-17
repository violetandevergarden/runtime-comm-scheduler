"""Static JobPacer Plan order, dependency, and determinism checks."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from examples.jobpacer.plan_builder import (
    PlannedTask,
    _tail_after,
    build_plan,
    key_labels,
    policy_names,
    task_key,
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


def test_tail_uses_compute_communication_overlap_critical_path():
    communications = (
        CollectiveComm(0, estimated_comm_s=0.004, consumer_compute_s=0.010),
        CollectiveComm(
            1,
            producer_compute_s=0.002,
            estimated_comm_s=0.005,
            consumer_compute_s=0.003,
        ),
        CollectiveComm(2, estimated_comm_s=0.002, consumer_compute_s=0.008),
    )
    tasks = tuple(
        PlannedTask("job-a", communication, task_key("job-a", communication.id))
        for communication in communications
    )

    assert abs(_tail_after(tasks, 0) - 0.021) < 1e-12


def test_plan_and_digest_are_deterministic():
    first = build_plan(_workload(), "ltf", version=4, window_id=9)
    second = build_plan(_workload(), "ltf", version=4, window_id=9)
    assert first == second
    assert first.digest() == second.digest()
    assert [key.as_list() for key in first.keys] == [key.as_list() for key in second.keys]


def test_policy_registry_exposes_static_algorithms():
    assert policy_names() == ("fifo", "ltf")


def test_job_membership_accepts_arbitrary_global_ranks():
    workload = Workload(
        "members",
        (Job("job-a", (CollectiveComm(0),), ranks=(3, 0, 2)),),
    )
    assert ranks_for_job(workload.jobs[0], 4) == (3, 0, 2)
    restored = Workload.from_dict(workload.to_dict())
    assert restored.jobs[0].ranks == (3, 0, 2)
