from __future__ import annotations

import pytest

from runtime_comm_scheduler.runtime.coordinator import CoordinatorState
from runtime_comm_scheduler.runtime.model import CollectiveSpec, GroupSpec, TaskHint, TaskSpec
from runtime_comm_scheduler.runtime.policy import (
    Anticipated,
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
    hint = TaskHint(0.0, 0.01, 0.02)
    payload = hint.to_dict()
    assert payload == {
        "ready_after_s": 0.0,
        "estimated_comm_s": 0.01,
        "remaining_tail_s": 0.02,
    }
    assert TaskHint.from_dict({**payload, "estimator_version": "tail-only"}) == hint


def test_policy_fifo_and_ltf_choose_different_candidates():
    candidates = (
        Candidate("short", 0.001, 0.1, 1),
        Candidate("long", 0.001, 1.0, 2),
    )
    assert make_policy("fifo").decide(PolicySnapshot(0.0, candidates, ())).task_id == "short"
    assert make_policy("ltf").decide(PolicySnapshot(0.0, candidates, ())).task_id == "long"


def test_ltf_always_ranks_current_comm_plus_postcompletion_tail():
    candidates = (
        Candidate("large-current-comm", 10.0, 0.0, 1),
        Candidate("larger-post-tail", 0.0, 8.0, 2),
    )
    assert make_policy("ltf").decide(PolicySnapshot(0.0, candidates, ())).task_id == "large-current-comm"


def test_coordinator_ignores_per_task_estimator_strings_when_ranking_ltf():
    coordinator = CoordinatorState((0, 1), policy="ltf")
    groups = tuple(GroupSpec(0, job, (0, 1)) for job in ("gate", "comm-heavy", "tail-heavy"))
    for endpoint in (0, 1):
        for sequence, group in enumerate(groups, 1):
            coordinator.apply(endpoint, "REGISTER_GROUP", sequence,
                              {"group": group.to_dict()}, 0.0)

    gate = _task("gate", 0)
    comm_heavy = _task("comm-heavy", 0)
    tail_heavy = _task("tail-heavy", 0)

    def offer(task, endpoint, sequence, comm, tail, old_version):
        hint = TaskHint(0.0, comm, tail).to_dict()
        hint["estimator_version"] = old_version
        return coordinator.apply(endpoint, "OFFER", sequence,
                                 {"task": task.to_dict(), "hint": hint}, float(sequence))

    for endpoint in (0, 1):
        grants = offer(gate, endpoint, 4, 0.001, 0.0, "tail-only")
    assert {grant.payload["task"]["task_id"] for grant in grants} == {gate.task_id}
    for endpoint in (0, 1):
        offer(comm_heavy, endpoint, 5, 10.0, 0.0, "tail-only")
        offer(tail_heavy, endpoint, 6, 0.0, 8.0, "linear-postcompletion-tail-v2")
        coordinator.apply(endpoint, "SUBMITTED", 7,
                          {"task_id": gate.task_id, "decision_seq": 1}, 7.0)
    coordinator.apply(0, "COMPLETED", 8,
                      {"task_id": gate.task_id, "decision_seq": 1}, 8.0)
    grants = coordinator.apply(1, "COMPLETED", 8,
                               {"task_id": gate.task_id, "decision_seq": 1}, 8.0)

    assert {grant.payload["task"]["task_id"] for grant in grants} == {comm_heavy.task_id}


def test_lookahead_ranks_anticipated_frontier_with_same_comm_plus_tail_score():
    current = Candidate("current", 10.0, 1.0, 1)
    anticipated = (
        Anticipated("high-comm", 0.05, 5.0, 2.0),
        Anticipated("high-tail", 0.05, 0.0, 5.0),
    )
    decision = BoundedLookaheadPolicy(0.1).decide(
        PolicySnapshot(0.0, (current,), anticipated)
    )
    assert decision.target_task_id == "high-comm"


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


def test_minimal_observation_preserves_protocol_counters_and_dispatch_order_only():
    coordinator = CoordinatorState((0, 1), policy="fifo", observation_mode="minimal")
    group = GroupSpec(0, "g", (0, 1))
    for endpoint in (0, 1):
        coordinator.apply(endpoint, "REGISTER_GROUP", 1, {"group": group.to_dict()}, 0.0)
    task = _task("g", 0)
    for endpoint in (0, 1):
        out = coordinator.apply(endpoint, "OFFER", 2, _payload(task), 0.1)
    assert [message.kind for message in out] == ["GRANT", "GRANT"]
    for endpoint in (0, 1):
        coordinator.apply(endpoint, "SUBMITTED", 3,
                           {"task_id": task.task_id, "decision_seq": 1}, 0.2)
    for endpoint in (0, 1):
        coordinator.apply(endpoint, "COMPLETED", 4,
                           {"task_id": task.task_id, "decision_seq": 1}, 0.3)
    assert coordinator.submitted_report_count == coordinator.completed_report_count == 2
    assert [record["task_id"] for record in coordinator.records
            if record.get("decision") == "dispatch"] == [task.task_id]
    assert all(record["kind"] not in {"submitted", "completed", "policy_snapshot", "eligible"}
               for record in coordinator.records)
    assert coordinator.instrumentation_records == []


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
    assert any(item.get("kind") == "lookahead_deadline" and item.get("target") == b.task_id
               for item in coordinator.records)
    assert any(item.get("kind") == "decision" and item.get("decision") == "wait"
               and item.get("target") == b.task_id for item in coordinator.records)
    assert any(item.get("kind") == "idle_interval" and item.get("reason") == "ACTIVE_LOOKAHEAD"
               and item.get("duration", 0) > 0 for item in coordinator.records)
    assert any(item.get("kind") == "decision" and item.get("decision") == "dispatch"
               and item.get("reason") == "LOOKAHEAD_DEADLINE_FALLBACK"
               for item in coordinator.records)


def test_one_members_completion_does_not_release_capacity_or_grant_next_task():
    coordinator = CoordinatorState((0, 1), policy="fifo")
    group = GroupSpec(0, "g", (0, 1))
    for endpoint in (0, 1):
        coordinator.apply(endpoint, "REGISTER_GROUP", 1, {"group": group.to_dict()}, 0.0)
    first, second = _task("g", 0), _task("g", 1)
    grants = []
    for endpoint in (0, 1):
        grants = coordinator.apply(endpoint, "OFFER", 2, _payload(first), 0.1)
    assert [item.payload["task"]["task_id"] for item in grants] == [first.task_id, first.task_id]
    for endpoint in (0, 1):
        assert coordinator.apply(endpoint, "OFFER", 3, _payload(second), 0.2) == []
        coordinator.apply(endpoint, "SUBMITTED", 4,
                          {"task_id": first.task_id, "decision_seq": 1}, 0.3)
    assert coordinator.apply(0, "COMPLETED", 5,
                             {"task_id": first.task_id, "decision_seq": 1}, 0.4) == []
    assert coordinator.inflight == first.task_id
    grants = coordinator.apply(1, "COMPLETED", 5,
                               {"task_id": first.task_id, "decision_seq": 1}, 0.5)
    assert [item.payload["task"]["task_id"] for item in grants] == [second.task_id, second.task_id]
    assert coordinator.inflight == second.task_id
