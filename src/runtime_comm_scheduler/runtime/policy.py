"""Pure policy decisions for online linear scheduling."""

from __future__ import annotations

from dataclasses import dataclass


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
    static_next: str | None = None
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


class Policy:
    def __init__(self, name: str, *, static_order: tuple[str, ...] = (), wait_budget_s: float = 0.02) -> None:
        if name not in {"static", "fifo", "ltf", "lookahead"}:
            raise ValueError(f"unknown runtime policy {name!r}")
        if wait_budget_s < 0:
            raise ValueError("wait_budget_s must be non-negative")
        self.name = name
        self.static_order = static_order
        self.wait_budget_s = wait_budget_s
        self._cursor = 0
        self._round = 0

    def decide(self, snapshot: PolicySnapshot) -> Action:
        if self.name == "static":
            if self._cursor >= len(self.static_order):
                return Done()
            target = self.static_order[self._cursor]
            if any(item.task_id == target for item in snapshot.eligible):
                self._cursor += 1
                return Dispatch(target, "static_order")
            return Idle("STATIC_HEAD_BLOCKED")
        if not snapshot.eligible:
            if snapshot.anticipated:
                return Idle("NO_ELIGIBLE")
            return Done()
        selected = min(
            snapshot.eligible,
            key=lambda item: (
                (-item.remaining_tail_s, item.eligible_seq, item.task_id)
                if self.name in {"ltf", "lookahead"}
                else (item.eligible_seq, item.task_id),
            ),
        )
        if self.name != "lookahead" or snapshot.must_dispatch:
            return Dispatch(selected.task_id, "dynamic_ltf" if self.name == "ltf" else "dynamic_fifo")
        anticipated = [item for item in snapshot.anticipated if item.task_id != selected.task_id]
        if not anticipated:
            return Dispatch(selected.task_id, "dynamic_ltf")
        target = max(anticipated, key=lambda item: (-item.remaining_tail_s, item.task_id))
        wait_s = max(0.0, target.predicted_ready_at - snapshot.now)
        dispatch_score = max(
            selected.estimated_comm_s + selected.remaining_tail_s,
            selected.estimated_comm_s + target.estimated_comm_s + target.remaining_tail_s,
        )
        wait_score = max(
            wait_s + target.estimated_comm_s + target.remaining_tail_s,
            wait_s + target.estimated_comm_s + selected.estimated_comm_s + selected.remaining_tail_s,
        )
        if wait_s <= self.wait_budget_s and wait_score < dispatch_score:
            self._round += 1
            return Wait(target.task_id, snapshot.now + min(wait_s, self.wait_budget_s))
        return Dispatch(selected.task_id, "LOOKAHEAD_DEADLINE_FALLBACK" if snapshot.must_dispatch else "dynamic_ltf")


def make_policy(name: str, *, static_order: tuple[str, ...] = (), wait_budget_s: float = 0.02) -> Policy:
    aliases = {"static_order": "static", "dynamic_fifo": "fifo", "dynamic_ltf": "ltf", "bounded_lookahead": "lookahead"}
    return Policy(aliases.get(name, name), static_order=static_order, wait_budget_s=wait_budget_s)
