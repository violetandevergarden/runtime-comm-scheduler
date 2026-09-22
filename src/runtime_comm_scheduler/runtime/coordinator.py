"""Single-owner coordinator state machine.

The class is transport agnostic.  A TCP event loop calls :meth:`apply` and
:meth:`tick`; tests can feed the same events directly without sockets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any

from .model import GroupSpec, TaskHint, TaskSpec
from .policy import (
    ActiveWait,
    Anticipated,
    Candidate,
    Dispatch,
    Done,
    Idle,
    Policy,
    PolicySnapshot,
    Wait,
    make_policy,
    select_ltf,
)


class CoordinatorError(RuntimeError):
    pass


@dataclass
class _EndpointState:
    next_event_seq: int = 1
    next_delivery_seq: int = 1
    input_closed: bool = False


@dataclass
class _Member:
    """记录一个 rank 对一个 collective 任务的进展"""

    declared: bool = False
    offered: bool = False
    submitted: bool = False
    completed: bool = False
    declared_at: float | None = None


@dataclass
class _Task:
    spec: TaskSpec
    hints: dict[int, TaskHint] = field(default_factory=dict)
    members: dict[int, _Member] = field(default_factory=dict)
    registered_seq: int = 0  # 第一次看到任务的顺序
    eligible_seq: int | None = None  # 任务满足调度的时间顺序
    grant_seq: int | None = None  # 调度顺序


@dataclass(frozen=True)
class Outbound:
    """准备发出的信息"""

    endpoint: int  # 现在就是目标rank
    kind: str
    payload: dict[str, Any]


class CoordinatorState:
    """Mutable state owned by exactly one coordinator event-loop thread."""

    def __init__(
        self,
        endpoints: tuple[int, ...],
        *,
        epoch: int = 0,
        policy: str = "fifo",
        static_order: tuple[str, ...] = (),
        max_inflight: int = 1,
        wait_budget_s: float = 0.02,
        epoch_timeout_s: float = 30.0,
    ) -> None:
        if not endpoints or len(set(endpoints)) != len(endpoints):
            raise ValueError("coordinator endpoints must be unique")
        if max_inflight != 1:
            raise ValueError("Stage 3.1 supports max_inflight=1")
        self.endpoints = tuple(sorted(endpoints))
        self.epoch = epoch
        self.max_inflight = max_inflight
        if epoch_timeout_s < 0:
            raise ValueError("epoch_timeout_s must be non-negative")
        self.epoch_timeout_s = epoch_timeout_s
        self.endpoint: dict[int, _EndpointState] = {
            rank: _EndpointState() for rank in self.endpoints
        }
        self.groups: dict[str, GroupSpec] = {}
        self._group_registered: dict[str, set[int]] = {}
        self.tasks: dict[str, _Task] = {}
        self._group_task_ids: dict[tuple[str, int], str] = {}
        self._next_group_seq: dict[str, int] = {}
        self._registered_seq = 0
        self._eligible_seq = 0
        self.decision_seq = 0
        self.inflight: str | None = None
        self.failed: dict[str, Any] | None = None
        self.finished = False
        self._finish_sent = False
        self.active_wait: ActiveWait | None = None
        self._wait_round = 0
        self._must_dispatch = False
        self._started_at: float | None = None
        self.records: list[dict[str, Any]] = []
        self._idle_reason: str | None = None
        self._idle_started_at: float | None = None
        self.policy: Policy = make_policy(
            policy, static_order=static_order, wait_budget_s=wait_budget_s
        )

    @property
    def done(self) -> bool:
        return self.finished or self.failed is not None

    @property
    def next_deadline(self) -> float | None:
        if self.done:
            return None
        deadlines = []
        if self._started_at is not None:
            deadlines.append(self._started_at + self.epoch_timeout_s)
        if self.active_wait is not None:
            deadlines.append(self.active_wait.deadline)
        return min(deadlines) if deadlines else None

    def apply(
        self,
        endpoint: int,
        kind: str,
        event_seq: int,
        payload: dict[str, Any],
        now: float,
    ) -> list[Outbound]:
        """处理rank上报事件"""

        if self.done:
            raise CoordinatorError("coordinator is in a terminal state")
        if endpoint not in self.endpoint:
            raise CoordinatorError(f"endpoint {endpoint} is not part of this epoch")
        if self._started_at is None:
            self._started_at = now
        self._check_deadlines(now)
        if self.failed is not None:
            return self._failure_messages()
        state = self.endpoint[endpoint]
        if state.input_closed and kind != "FAILED" and kind not in {"SUBMITTED", "COMPLETED"}:
            raise CoordinatorError(f"endpoint {endpoint} input is already closed")
        if kind == "INPUT_CLOSED" and state.input_closed:
            raise CoordinatorError(f"duplicate INPUT_CLOSED from endpoint {endpoint}")
        if event_seq != state.next_event_seq:
            raise CoordinatorError(
                f"event sequence error endpoint={endpoint}: expected {state.next_event_seq}, got {event_seq}"
            )

        state.next_event_seq += 1
        if kind == "REGISTER_GROUP":
            self._register_group(endpoint, GroupSpec.from_dict(payload["group"]), now)
        elif kind in {"DECLARE", "OFFER"}:
            task = self._task_from_payload(payload)
            self._accept_task(
                endpoint, task, TaskHint.from_dict(payload["hint"]), kind, now
            )
        elif kind in {"SUBMITTED", "COMPLETED"}:
            self._progress(endpoint, kind, payload, now)
        elif kind == "INPUT_CLOSED":
            state.input_closed = True
            self.records.append(
                {"kind": "input_closed", "endpoint": endpoint, "now": now}
            )
            missing_from_endpoint = [
                task.spec.task_id
                for task in self.tasks.values()
                if endpoint in task.members and not task.members[endpoint].offered
            ]
            if missing_from_endpoint:
                self._fail(
                    "missing_offer",
                    endpoint=endpoint,
                    missing_tasks=missing_from_endpoint,
                    now=now,
                )
        elif kind == "FAILED":
            failure = dict(payload)
            failure.setdefault("now", now)
            self._fail("remote_failed", endpoint=endpoint, **failure)
        else:
            raise CoordinatorError(f"unknown coordinator event {kind!r}")

        if self.failed is not None:
            return self._failure_messages()

        self._check_deadlines(now)
        if self.failed is not None:
            return self._failure_messages()
        self._refresh_eligibility(now)

        out = self._maybe_finish(now)
        if not out and not self.finished:
            out = self._decide(now)
        return out

    def tick(self, now: float) -> list[Outbound]:
        """定期检查状态"""

        if self.done:
            return []
        self._check_deadlines(now)
        if self.failed is not None:
            return self._failure_messages()
        self._refresh_eligibility(now)
        out = self._maybe_finish(now)
        return out or self._decide(now)

    def fail(self, reason: str, **details: Any) -> list[Outbound]:
        if self.finished:
            raise CoordinatorError("coordinator is in a terminal state")
        details.setdefault("now", time.monotonic())
        self._fail(reason, **details)
        return self._failure_messages()

    def _check_deadlines(self, now: float) -> None:
        if (
            self._started_at is not None
            and now - self._started_at >= self.epoch_timeout_s
        ):
            self._close_idle(now)
            self._fail("epoch_timeout", now=now, inflight=self.inflight)
            return
        if self.active_wait is not None and now >= self.active_wait.deadline:
            self.records.append(
                {
                    "kind": "lookahead_deadline",
                    "now": now,
                    "target": self.active_wait.target_task_id,
                    "deadline": self.active_wait.deadline,
                }
            )
            self.active_wait = None
            self._must_dispatch = True

    def _register_group(self, endpoint: int, group: GroupSpec, now: float) -> None:
        """注册通信组"""

        if group.epoch != self.epoch:
            raise CoordinatorError(f"old or future epoch {group.epoch}")
        if endpoint not in group.ranks:
            raise CoordinatorError(
                f"endpoint {endpoint} is not a member of group {group.group_id}"
            )
        if any(rank not in self.endpoint for rank in group.ranks):
            raise CoordinatorError(f"group {group.group_id} contains an unknown endpoint")

        previous = self.groups.get(group.group_id)
        if previous is not None and previous != group:
            raise CoordinatorError(f"group metadata mismatch for {group.group_id}")
        if endpoint in self._group_registered.get(group.group_id, set()):
            raise CoordinatorError(f"duplicate group registration for {group.group_id}")
        self.groups[group.group_id] = group
        self._group_registered.setdefault(group.group_id, set()).add(endpoint)
        self._next_group_seq.setdefault(group.group_id, 0)
        self.records.append(
            {"kind": "group_registered", "group_id": group.group_id, "endpoint": endpoint, "now": now}
        )

    def _task_from_payload(self, payload: dict[str, Any]) -> TaskSpec:
        task = TaskSpec.from_dict(payload["task"])
        if task.epoch != self.epoch:
            raise CoordinatorError(f"task {task.task_id} belongs to epoch {task.epoch}")
        group = self.groups.get(task.group_id)
        if group is not None and task.group_id != group.group_id:
            raise CoordinatorError("task group mismatch")
        return task

    def _accept_task(
        self, endpoint: int, spec: TaskSpec, hint: TaskHint, kind: str, now: float
    ) -> None:
        """把各个 rank 对同一个 collective 的声明汇总成一个 _Task"""

        group = self.groups.get(spec.group_id)
        if group is None:
            raise CoordinatorError(
                f"task {spec.task_id} arrived before group registration"
            )
        members = set(group.ranks)
        if endpoint not in members:
            raise CoordinatorError(
                f"endpoint {endpoint} is not a member of task {spec.task_id}"
            )
        existing = self.tasks.get(spec.task_id)

        if existing is None:
            # 检查组内序号冲突
            # 保证同一个 ProcessGroup 的 rank 必须以相同顺序发起 collective
            pair = (spec.group_id, spec.group_seq)
            if (
                pair in self._group_task_ids
                and self._group_task_ids[pair] != spec.task_id
            ):
                raise CoordinatorError(f"group sequence conflict for {pair}")
            self._registered_seq += 1
            existing = _Task(spec, registered_seq=self._registered_seq)
            existing.members = {rank: _Member() for rank in members}
            self.tasks[spec.task_id] = existing
            self._group_task_ids[pair] = spec.task_id
        elif existing.spec != spec:
            raise CoordinatorError(f"task metadata mismatch for {spec.task_id}")
        member = existing.members[endpoint]

        if kind == "DECLARE":
            if member.declared:
                if existing.hints[endpoint] != hint:
                    raise CoordinatorError(
                        f"declaration metadata mismatch for {spec.task_id}"
                    )
                raise CoordinatorError(f"duplicate declaration for {spec.task_id}")
            member.declared = True
            member.declared_at = (
                now if member.declared_at is None else member.declared_at
            )
        else:
            if member.offered:
                raise CoordinatorError(
                    f"duplicate offer for {spec.task_id} from endpoint {endpoint}"
                )
            member.offered = True
            member.declared = True
            member.declared_at = (
                now if member.declared_at is None else member.declared_at
            )

        existing.hints[endpoint] = hint
        self.records.append(
            {
                "kind": kind.lower(),
                "task_id": spec.task_id,
                "endpoint": endpoint,
                "now": now,
            }
        )

    def _progress(
        self, endpoint: int, kind: str, payload: dict[str, Any], now: float
    ) -> None:
        """处理 rank 对已获授权任务上报的执行进度"""

        task_id = payload.get("task_id")
        task = self.tasks.get(task_id)
        if task is None or task.grant_seq is None:
            raise CoordinatorError(f"{kind} for ungranted task {task_id}")
        decision_seq = payload.get("decision_seq")
        if (
            not isinstance(decision_seq, int)
            or isinstance(decision_seq, bool)
            or decision_seq <= 0
        ):
            raise CoordinatorError(f"{kind} decision sequence must be positive")
        if decision_seq != task.grant_seq:
            raise CoordinatorError(f"{kind} decision mismatch for {task_id}")

        member = task.members.get(endpoint)
        if member is None:
            raise CoordinatorError(f"endpoint {endpoint} is not a member of {task_id}")

        if kind == "SUBMITTED":
            if not member.offered or member.submitted:
                raise CoordinatorError(f"invalid submitted transition for {task_id}")
            member.submitted = True
        else:
            if not member.submitted or member.completed:
                raise CoordinatorError(f"invalid completed transition for {task_id}")
            member.completed = True
        self.records.append(
            {
                "kind": kind.lower(),
                "task_id": task_id,
                "endpoint": endpoint,
                "decision_seq": decision_seq,
                "now": now,
            }
        )

        if kind == "COMPLETED" and all(
            item.completed for item in task.members.values()
        ):
            self._next_group_seq[task.spec.group_id] = task.spec.group_seq + 1
            self.inflight = None
            self.active_wait = None
            self._must_dispatch = False

    def _refresh_eligibility(self, now: float) -> None:
        """Assign the first eligible sequence in event-processing order."""

        for task in self.tasks.values():
            if task.grant_seq is not None or task.eligible_seq is not None:
                continue
            if task.spec.group_seq != self._next_group_seq.get(task.spec.group_id, 0):
                continue
            if not all(member.offered for member in task.members.values()):
                continue
            self._eligible_seq += 1
            task.eligible_seq = self._eligible_seq
            self.records.append(
                {
                    "kind": "eligible",
                    "task_id": task.spec.task_id,
                    "eligible_seq": task.eligible_seq,
                    "now": now,
                }
            )

    def _eligible_candidates(self) -> list[Candidate]:
        """Build candidates without changing eligibility state."""

        result: list[Candidate] = []
        for task in self.tasks.values():
            if task.grant_seq is not None or task.eligible_seq is None:
                continue
            if task.spec.group_seq != self._next_group_seq.get(task.spec.group_id, 0):
                continue
            if not all(member.offered for member in task.members.values()):
                continue
            hints = tuple(task.hints.values())
            result.append(
                Candidate(
                    task.spec.task_id,
                    max(item.estimated_comm_s for item in hints),
                    max(item.remaining_tail_s for item in hints),
                    task.eligible_seq,
                )
            )
        return result

    def _eligible(self) -> list[Candidate]:
        """Compatibility helper for callers that inspect the coordinator."""
        self._refresh_eligibility(self._started_at or 0.0)
        return self._eligible_candidates()

    def _anticipated(self, now: float) -> list[Anticipated]:
        """已声明，但还没全部 ready，预计将来能执行"""

        result: list[Anticipated] = []
        for task in self.tasks.values():
            if (
                task.grant_seq is not None
                or task.spec.group_seq
                != self._next_group_seq.get(task.spec.group_id, 0)
            ):
                continue
            if all(member.offered for member in task.members.values()) or not any(
                member.declared for member in task.members.values()
            ):
                continue
            if not all(
                member.declared
                and task.hints.get(rank) is not None
                and task.hints[rank].ready_after_s is not None
                and member.declared_at is not None
                for rank, member in task.members.items()
            ):
                continue
            hints = tuple(task.hints[rank] for rank in task.members)
            predicted = max(
                member.declared_at + task.hints[rank].ready_after_s
                for rank, member in task.members.items()
            )
            result.append(
                Anticipated(
                    task.spec.task_id,
                    predicted,
                    max(item.estimated_comm_s for item in hints),
                    max(item.remaining_tail_s for item in hints),
                )
            )
        return result

    def _decide(self, now: float) -> list[Outbound]:
        """根据策略决定下一个动作"""

        if self.done:
            return []
        if self.inflight is not None:
            self._set_idle("CAPACITY_FULL", now)
            return []
        self._refresh_eligibility(now)
        eligible = tuple(self._eligible_candidates())
        anticipated = tuple(self._anticipated(now))
        previous_wait = self.active_wait
        if previous_wait is not None and now >= previous_wait.deadline:
            self.active_wait = None
            self._must_dispatch = True
        self.records.append(
            {
                "kind": "policy_snapshot",
                "now": now,
                "eligible": [
                    {
                        "task_id": item.task_id,
                        "estimated_comm_s": item.estimated_comm_s,
                        "remaining_tail_s": item.remaining_tail_s,
                        "eligible_seq": item.eligible_seq,
                    }
                    for item in eligible
                ],
                "anticipated": [
                    {
                        "task_id": item.task_id,
                        "predicted_ready_at": item.predicted_ready_at,
                        "estimated_comm_s": item.estimated_comm_s,
                        "remaining_tail_s": item.remaining_tail_s,
                    }
                    for item in anticipated
                ],
                "active_wait": (
                    None
                    if self.active_wait is None
                    else {
                        "target_task_id": self.active_wait.target_task_id,
                        "deadline": self.active_wait.deadline,
                        "round_id": self.active_wait.round_id,
                    }
                ),
                "must_dispatch": self._must_dispatch,
            }
        )
        if self._must_dispatch and not eligible:
            self._set_idle("NO_ELIGIBLE", now)
            return []
        decision = self.policy.decide(
            PolicySnapshot(
                now, eligible, anticipated, self.active_wait, self._must_dispatch
            )
        )
        if isinstance(decision, Wait):
            if self._must_dispatch:
                decision = Dispatch(select_ltf(eligible).task_id, "LOOKAHEAD_DEADLINE_FALLBACK")
            else:
                deadline = decision.deadline
                if previous_wait is not None:
                    deadline = min(deadline, previous_wait.deadline)
                if deadline <= now:
                    self._must_dispatch = True
                    self.active_wait = None
                    return self._decide(now)
                self._wait_round += 1
                self.active_wait = ActiveWait(
                    decision.target_task_id, deadline, self._wait_round
                )
                self._must_dispatch = False
                self._set_idle("ACTIVE_LOOKAHEAD", now)
                self.records.append(
                    {
                        "kind": "decision",
                        "decision": "wait",
                        "target": decision.target_task_id,
                        "deadline": deadline,
                        "now": now,
                        "eligible": [item.task_id for item in eligible],
                        "anticipated": [item.task_id for item in anticipated],
                        "reason": decision.reason,
                    }
                )
                return []
        if isinstance(decision, Dispatch):
            selected = next(
                (item for item in eligible if item.task_id == decision.task_id), None
            )
            if selected is None:
                raise CoordinatorError(
                    f"policy selected ineligible task {decision.task_id}"
                )
            self.decision_seq += 1
            task = self.tasks[selected.task_id]
            task.grant_seq = self.decision_seq
            self.inflight = task.spec.task_id
            self.active_wait = None
            self._must_dispatch = False
            self._close_idle(now)
            out: list[Outbound] = []
            for rank in sorted(task.members):
                delivery = self.endpoint[rank].next_delivery_seq
                self.endpoint[rank].next_delivery_seq += 1
                out.append(
                    Outbound(
                        rank,
                        "GRANT",
                        {
                            "task": task.spec.to_dict(),
                            "decision_seq": self.decision_seq,
                            "delivery_seq": delivery,
                        },
                    )
                )
            self.records.append(
                {
                    "kind": "decision",
                    "decision": "dispatch",
                    "task_id": task.spec.task_id,
                    "decision_seq": self.decision_seq,
                    "reason": decision.reason,
                    "now": now,
                    "eligible": [item.task_id for item in eligible],
                    "anticipated": [item.task_id for item in anticipated],
                }
            )
            return out
        if isinstance(decision, Idle):
            self._set_idle(decision.reason, now)
            self.records.append(
                {
                    "kind": "idle",
                    "reason": decision.reason,
                    "now": now,
                    "eligible": [item.task_id for item in eligible],
                    "anticipated": [item.task_id for item in anticipated],
                }
            )
        return []

    def _maybe_finish(self, now: float) -> list[Outbound]:
        """判断当前 epoch 是否已经可以正常结束"""

        if self._finish_sent or not all(
            state.input_closed for state in self.endpoint.values()
        ):
            return []
        missing = [
            task.spec.task_id
            for task in self.tasks.values()
            if not all(member.offered for member in task.members.values())
        ]
        if missing:
            self._fail("missing_offer", missing_tasks=missing, now=now)
            return self._failure_messages()
        if self.inflight is None:
            missing_sequences = [
                task.spec.task_id
                for task in self.tasks.values()
                if task.grant_seq is None
                and task.spec.group_seq != self._next_group_seq.get(task.spec.group_id, 0)
            ]
            if missing_sequences:
                self._fail(
                    "missing_group_sequence",
                    missing_tasks=missing_sequences,
                    now=now,
                )
                return self._failure_messages()
        missing_policy_tasks = self.policy.missing_tasks(self.tasks)
        if missing_policy_tasks:
            self._fail(
                "missing_static_task", missing_tasks=list(missing_policy_tasks), now=now
            )
            return self._failure_messages()
        if self.inflight is not None or any(
            not all(member.completed for member in task.members.values())
            for task in self.tasks.values()
        ):
            return []
        self.finished = True
        self._finish_sent = True
        self._close_idle(now)
        payload = {
            "epoch": self.epoch,
            "decision_seq": self.decision_seq,
            "task_count": len(self.tasks),
        }
        self.records.append({"kind": "finished", "now": now, **payload})
        return [Outbound(rank, "FINISHED", payload) for rank in self.endpoints]

    def _set_idle(self, reason: str, now: float) -> None:
        if self._idle_reason == reason:
            return
        self._close_idle(now)
        self._idle_reason = reason
        self._idle_started_at = now

    def _close_idle(self, now: float) -> None:
        if self._idle_reason is None or self._idle_started_at is None:
            return
        self.records.append(
            {
                "kind": "idle_interval",
                "reason": self._idle_reason,
                "start": self._idle_started_at,
                "end": now,
                "duration": max(0.0, now - self._idle_started_at),
            }
        )
        self._idle_reason = None
        self._idle_started_at = None

    def _fail(self, reason: str, **details: Any) -> None:
        if self.failed is None:
            self._close_idle(details.get("now", self._started_at or 0.0))
            self.failed = {"epoch": self.epoch, "reason": reason, **details}
            self.records.append({"kind": "failed", **self.failed})

    def _failure_messages(self) -> list[Outbound]:
        if self.failed is None:
            return []
        return [Outbound(rank, "FAILED", self.failed) for rank in self.endpoints]
