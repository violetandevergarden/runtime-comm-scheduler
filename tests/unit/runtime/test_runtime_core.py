from __future__ import annotations

import pytest

from runtime_comm_scheduler.runtime.coordinator import CoordinatorState
from runtime_comm_scheduler.runtime.model import CollectiveSpec, GroupSpec, TaskHint, TaskSpec
from runtime_comm_scheduler.runtime.policy import Candidate, PolicySnapshot, make_policy


def _task(job: str, ordinal: int) -> TaskSpec:
    return TaskSpec(0, job, f"{job}/comm-{ordinal}", job, ordinal, CollectiveSpec("all_reduce", 1, 4, "float32", (1,)))


def _payload(task: TaskSpec, tail: float = 0.0) -> dict:
    return {"task": task.to_dict(), "hint": TaskHint(0.0, 0.001, tail).to_dict()}


def test_model_json_round_trip_and_group_normalization():
    group = GroupSpec(0, "g", (1, 0))
    assert group.ranks == (0, 1)
    task = _task("g", 0)
    assert TaskSpec.from_dict(task.to_dict()) == task
    assert CollectiveSpec.from_dict(task.collective.to_dict()) == task.collective


def test_policy_fifo_and_ltf_choose_different_candidates():
    candidates = (
        Candidate("short", 0.001, 0.1, 1),
        Candidate("long", 0.001, 1.0, 2),
    )
    assert make_policy("fifo").decide(PolicySnapshot(0.0, candidates, ())).task_id == "short"
    assert make_policy("ltf").decide(PolicySnapshot(0.0, candidates, ())).task_id == "long"


def test_coordinator_requires_both_members_and_releases_one_inflight_task():
    coordinator = CoordinatorState((0, 1), policy="fifo")
    group = GroupSpec(0, "g", (0, 1))
    for endpoint in (0, 1):
        coordinator.apply(endpoint, "REGISTER_GROUP", 1, {"group": group.to_dict()}, 0.0)
    task = _task("g", 0)
    assert coordinator.apply(0, "OFFER", 2, _payload(task), 0.1) == []
    out = coordinator.apply(1, "OFFER", 2, _payload(task), 0.2)
    assert [item.kind for item in out] == ["GRANT", "GRANT"]
    assert coordinator.inflight == task.task_id
    with pytest.raises(Exception, match="event sequence"):
        coordinator.apply(0, "SUBMITTED", 2, {"task_id": task.task_id, "decision_seq": 1}, 0.3)
    coordinator.apply(0, "SUBMITTED", 3, {"task_id": task.task_id, "decision_seq": 1}, 0.3)
    coordinator.apply(1, "SUBMITTED", 3, {"task_id": task.task_id, "decision_seq": 1}, 0.3)
    coordinator.apply(0, "COMPLETED", 4, {"task_id": task.task_id, "decision_seq": 1}, 0.4)
    coordinator.apply(1, "COMPLETED", 4, {"task_id": task.task_id, "decision_seq": 1}, 0.4)
    assert coordinator.inflight is None


def test_static_policy_waits_at_head():
    policy = make_policy("static", static_order=("head", "tail"))
    decision = policy.decide(PolicySnapshot(0.0, (Candidate("tail", 0.1, 0.0, 1),), ()))
    assert decision.reason == "STATIC_HEAD_BLOCKED"


def test_coordinator_rejects_metadata_conflict_and_missing_static_task():
    coordinator = CoordinatorState((0, 1), policy="fifo")
    group = GroupSpec(0, "g", (0, 1))
    for endpoint in (0, 1):
        coordinator.apply(endpoint, "REGISTER_GROUP", 1, {"group": group.to_dict()}, 0.0)
    first = _task("g", 0)
    coordinator.apply(0, "OFFER", 2, _payload(first), 0.1)
    conflict = TaskSpec(0, "g", "g/comm-0", "g", 0, CollectiveSpec("all_reduce", 2, 8, "float32", (2,)))
    with pytest.raises(Exception, match="metadata mismatch"):
        coordinator.apply(1, "OFFER", 2, _payload(conflict), 0.1)

    static = CoordinatorState((0, 1), policy="static", static_order=("g/comm-0", "g/comm-1"))
    for endpoint in (0, 1):
        static.apply(endpoint, "REGISTER_GROUP", 1, {"group": group.to_dict()}, 0.0)
    for endpoint in (0, 1):
        static.apply(endpoint, "OFFER", 2, _payload(first), 0.1)
    static.apply(0, "INPUT_CLOSED", 3, {}, 0.2)
    failed = static.apply(1, "INPUT_CLOSED", 3, {}, 0.2)
    assert failed[0].kind == "FAILED"
    assert "g/comm-1" in failed[0].payload["missing_tasks"]
