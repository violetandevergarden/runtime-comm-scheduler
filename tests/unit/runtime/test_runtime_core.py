from __future__ import annotations

import pytest

from runtime_comm_scheduler.runtime.coordinator import CoordinatorState
from runtime_comm_scheduler.runtime.model import CollectiveSpec, GroupSpec, TaskHint, TaskSpec
from runtime_comm_scheduler.runtime.policy import (
    BoundedLookaheadPolicy,
    Candidate,
    FifoPolicy,
    LongestTailFirstPolicy,
    PolicySnapshot,
    StaticPolicy,
    make_policy,
)


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


def test_policy_factory_returns_separate_strategy_types():
    assert isinstance(make_policy("static", static_order=("task",)), StaticPolicy)
    assert isinstance(make_policy("fifo"), FifoPolicy)
    assert isinstance(make_policy("ltf"), LongestTailFirstPolicy)
    assert isinstance(make_policy("lookahead"), BoundedLookaheadPolicy)


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


def test_dynamic_fifo_uses_first_eligible_arrival_not_declare_order():
    coordinator = CoordinatorState((0, 1), policy="fifo")
    for group_seq, group_id in enumerate(("x", "a", "b"), 1):
        group = GroupSpec(0, group_id, (0, 1))
        for endpoint in (0, 1):
            coordinator.apply(
                endpoint,
                "REGISTER_GROUP",
                group_seq,
                {"group": group.to_dict()},
                0.0,
            )

    x, a, b = _task("x", 0), _task("a", 0), _task("b", 0)
    for endpoint in (0, 1):
        coordinator.apply(endpoint, "OFFER", 4, _payload(x), 0.1)
    coordinator.apply(0, "DECLARE", 5, _payload(a), 0.2)
    coordinator.apply(0, "DECLARE", 6, _payload(b), 0.21)
    coordinator.apply(0, "OFFER", 7, _payload(b), 0.22)
    coordinator.apply(0, "OFFER", 8, _payload(a), 0.23)
    coordinator.apply(1, "DECLARE", 5, _payload(b), 0.24)
    coordinator.apply(1, "DECLARE", 6, _payload(a), 0.25)
    coordinator.apply(1, "OFFER", 7, _payload(b), 0.26)
    coordinator.apply(1, "OFFER", 8, _payload(a), 0.27)

    for endpoint in (0, 1):
        coordinator.apply(endpoint, "SUBMITTED", 9, {"task_id": x.task_id, "decision_seq": 1}, 0.3)
    coordinator.apply(0, "COMPLETED", 10, {"task_id": x.task_id, "decision_seq": 1}, 0.31)
    out = coordinator.apply(1, "COMPLETED", 10, {"task_id": x.task_id, "decision_seq": 1}, 0.32)
    assert [item.payload["task"]["task_id"] for item in out] == [b.task_id, b.task_id]


def test_lookahead_deadline_falls_back_without_refreshing():
    coordinator = CoordinatorState((0, 1), policy="lookahead", wait_budget_s=0.005)
    for group_seq, group_id in enumerate(("a", "b"), 1):
        group = GroupSpec(0, group_id, (0, 1))
        for endpoint in (0, 1):
            coordinator.apply(endpoint, "REGISTER_GROUP", group_seq, {"group": group.to_dict()}, 0.0)
    a, b = _task("a", 0), _task("b", 0)
    a_payload = {"task": a.to_dict(), "hint": TaskHint(0.0, 0.01, 0.1).to_dict()}
    b_payload = {"task": b.to_dict(), "hint": TaskHint(0.004, 0.001, 10.0).to_dict()}
    for endpoint in (0, 1):
        coordinator.apply(endpoint, "DECLARE", 3, b_payload, 0.0)
    for endpoint in (0, 1):
        coordinator.apply(endpoint, "OFFER", 4, a_payload, 0.0)
    assert coordinator.active_wait is not None
    deadline = coordinator.active_wait.deadline
    out = coordinator.tick(deadline + 0.001)
    assert deadline == 0.004
    assert [item.kind for item in out] == ["GRANT", "GRANT"]
    assert coordinator.active_wait is None
