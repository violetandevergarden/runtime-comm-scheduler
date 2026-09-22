"""Pure policy decisions for online linear scheduling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Collection, Protocol


@dataclass(frozen=True)
class Candidate:
    """已经可以进入候选的任务"""
    task_id: str
    estimated_comm_s: float
    remaining_tail_s: float
    eligible_seq: int


@dataclass(frozen=True)
class Anticipated:
    """预期会到来的任务"""
    task_id: str
    predicted_ready_at: float
    estimated_comm_s: float
    remaining_tail_s: float


@dataclass(frozen=True)
class ActiveWait:
    """做出等待决策后的状态记忆"""
    target_task_id: str
    deadline: float
    round_id: int


@dataclass(frozen=True)
class PolicySnapshot:
    now: float
    eligible: tuple[Candidate, ...]
    anticipated: tuple[Anticipated, ...]
    active_wait: ActiveWait | None = None
    must_dispatch: bool = False


@dataclass(frozen=True)
class Dispatch:
    """立即调度任务"""
    task_id: str
    reason: str = "dispatch"


@dataclass(frozen=True)
class Wait:
    """等待某个任务到来"""
    target_task_id: str
    deadline: float
    reason: str = "active_lookahead"


@dataclass(frozen=True)
class Idle:
    """暂时无任务"""
    reason: str


@dataclass(frozen=True)
class Done:
    """全部任务已完成"""
    reason: str = "done"


Action = Dispatch | Wait | Idle | Done


class Policy(Protocol):
    def decide(self, snapshot: PolicySnapshot) -> Action:
        ...

    def missing_tasks(self, task_ids: Collection[str]) -> tuple[str, ...]:
        ...


def _no_work(snapshot: PolicySnapshot) -> Action | None:
    if snapshot.eligible:
        return None
    if snapshot.anticipated:
        return Idle("NO_ELIGIBLE")
    return Done()


def select_fifo(tasks: tuple[Candidate, ...]) -> Candidate:
    return min(tasks, key=lambda task: (task.eligible_seq, task.task_id))


def select_ltf(tasks: tuple[Candidate, ...]) -> Candidate:
    return min(
        tasks,
        key=lambda task: (-task.remaining_tail_s, task.eligible_seq, task.task_id),
    )


class StaticPolicy:
    def __init__(self, static_order: tuple[str, ...]) -> None:
        self._order = static_order
        self._cursor = 0

    def decide(self, snapshot: PolicySnapshot) -> Action:
        if self._cursor >= len(self._order):
            return Done()
        target = self._order[self._cursor]
        if not any(task.task_id == target for task in snapshot.eligible):
            return Idle("STATIC_HEAD_BLOCKED")
        self._cursor += 1
        return Dispatch(target, "static_order")

    def missing_tasks(self, task_ids: Collection[str]) -> tuple[str, ...]:
        known = set(task_ids)
        return tuple(task_id for task_id in self._order if task_id not in known)


class FifoPolicy:
    def decide(self, snapshot: PolicySnapshot) -> Action:
        no_work = _no_work(snapshot)
        if no_work is not None:
            return no_work
        selected = select_fifo(snapshot.eligible)
        return Dispatch(selected.task_id, "dynamic_fifo")

    def missing_tasks(self, task_ids: Collection[str]) -> tuple[str, ...]:
        return ()


class LongestTailFirstPolicy:
    def decide(self, snapshot: PolicySnapshot) -> Action:
        no_work = _no_work(snapshot)
        if no_work is not None:
            return no_work
        selected = select_ltf(snapshot.eligible)
        return Dispatch(selected.task_id, "dynamic_ltf")

    def missing_tasks(self, task_ids: Collection[str]) -> tuple[str, ...]:
        return ()


class BoundedLookaheadPolicy:
    def __init__(self, wait_budget_s: float) -> None:
        if wait_budget_s < 0:
            raise ValueError("wait_budget_s must be non-negative")
        self._wait_budget_s = wait_budget_s

    def decide(self, snapshot: PolicySnapshot) -> Action:
        no_work = _no_work(snapshot)
        if no_work is not None:
            return no_work
        selected = select_ltf(snapshot.eligible)
        if snapshot.must_dispatch:
            return Dispatch(selected.task_id, "LOOKAHEAD_DEADLINE_FALLBACK")
        anticipated = tuple(task for task in snapshot.anticipated if task.task_id != selected.task_id)
        if not anticipated:
            return Dispatch(selected.task_id, "dynamic_ltf")
        target = max(anticipated, key=lambda task: (task.remaining_tail_s, task.task_id))
        wait_s = max(0.0, target.predicted_ready_at - snapshot.now)
        dispatch_score = max(
            selected.estimated_comm_s + selected.remaining_tail_s,
            selected.estimated_comm_s + target.estimated_comm_s + target.remaining_tail_s,
        )
        wait_score = max(
            wait_s + target.estimated_comm_s + target.remaining_tail_s,
            wait_s + target.estimated_comm_s + selected.estimated_comm_s + selected.remaining_tail_s,
        )
        if 0.0 < wait_s <= self._wait_budget_s and wait_score < dispatch_score:
            return Wait(target.task_id, snapshot.now + wait_s)
        return Dispatch(selected.task_id, "dynamic_ltf")

    def missing_tasks(self, task_ids: Collection[str]) -> tuple[str, ...]:
        return ()


def make_policy(name: str, *, static_order: tuple[str, ...] = (), wait_budget_s: float = 0.02) -> Policy:
    aliases = {"static_order": "static", "dynamic_fifo": "fifo", "dynamic_ltf": "ltf", "bounded_lookahead": "lookahead"}
    name = aliases.get(name, name)
    if name == "static":
        return StaticPolicy(static_order)
    if name == "fifo":
        return FifoPolicy()
    if name == "ltf":
        return LongestTailFirstPolicy()
    if name == "lookahead":
        return BoundedLookaheadPolicy(wait_budget_s)
    raise ValueError(f"unknown runtime policy {name!r}")
