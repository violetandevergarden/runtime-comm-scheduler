"""Single-owner coordinator state machine.

The class is transport agnostic.  A TCP event loop calls :meth:`apply` and
:meth:`tick`; tests can feed the same events directly without sockets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .model import GroupSpec, TaskHint, TaskSpec
from .policy import ActiveWait, Anticipated, Candidate, Dispatch, Done, Idle, Policy, PolicySnapshot, Wait, make_policy


class CoordinatorError(RuntimeError):
    pass


@dataclass
class _EndpointState:
    next_event_seq: int = 1
    next_delivery_seq: int = 1
    input_closed: bool = False


@dataclass
class _Member:
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
    registered_seq: int = 0
    eligible_seq: int | None = None
    grant_seq: int | None = None


@dataclass(frozen=True)
class Outbound:
    endpoint: int
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
        self.epoch_timeout_s = epoch_timeout_s
        self.endpoint: dict[int, _EndpointState] = {rank: _EndpointState() for rank in self.endpoints}
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
        self._must_dispatch = False
        self._started_at: float | None = None
        self.records: list[dict[str, Any]] = []
        self.policy: Policy = make_policy(policy, static_order=static_order, wait_budget_s=wait_budget_s)

    @property
    def done(self) -> bool:
        return self.finished or self.failed is not None

    def apply(self, endpoint: int, kind: str, event_seq: int, payload: dict[str, Any], now: float) -> list[Outbound]:
        if self.failed is not None:
            return []
        if endpoint not in self.endpoint:
            raise CoordinatorError(f"endpoint {endpoint} is not part of this epoch")
        if self._started_at is None:
            self._started_at = now
        state = self.endpoint[endpoint]
        if event_seq != state.next_event_seq:
            raise CoordinatorError(
                f"event sequence error endpoint={endpoint}: expected {state.next_event_seq}, got {event_seq}"
            )
        state.next_event_seq += 1
        if kind == "REGISTER_GROUP":
            self._register_group(endpoint, GroupSpec.from_dict(payload["group"]))
        elif kind in {"DECLARE", "OFFER"}:
            task = self._task_from_payload(payload)
            self._accept_task(endpoint, task, TaskHint.from_dict(payload["hint"]), kind, now)
        elif kind in {"SUBMITTED", "COMPLETED"}:
            self._progress(endpoint, kind, payload)
        elif kind == "INPUT_CLOSED":
            state.input_closed = True
            self.records.append({"kind": "input_closed", "endpoint": endpoint, "now": now})
            missing_from_endpoint = [
                task.spec.task_id
                for task in self.tasks.values()
                if endpoint in task.members
                and not task.members[endpoint].offered
            ]
            if missing_from_endpoint:
                self._fail("missing_offer", endpoint=endpoint, missing_tasks=missing_from_endpoint, now=now)
        elif kind == "FAILED":
            self._fail("remote_failed", endpoint=endpoint, **payload)
        else:
            raise CoordinatorError(f"unknown coordinator event {kind!r}")
        if self.failed is not None:
            return self._failure_messages()
        out = self._maybe_finish(now)
        if not out and not self.finished:
            out = self._decide(now)
        return out

    def tick(self, now: float) -> list[Outbound]:
        if self.done:
            return []
        if self._started_at is not None and now - self._started_at > self.epoch_timeout_s:
            self._fail("epoch_timeout", now=now, inflight=self.inflight)
            return self._failure_messages()
        if self.active_wait is not None and now >= self.active_wait.deadline:
            self._must_dispatch = True
        return self._decide(now)

    def fail(self, reason: str, **details: Any) -> list[Outbound]:
        self._fail(reason, **details)
        return self._failure_messages()

    def _register_group(self, endpoint: int, group: GroupSpec) -> None:
        if group.epoch != self.epoch:
            raise CoordinatorError(f"old or future epoch {group.epoch}")
        if endpoint not in group.ranks:
            raise CoordinatorError(f"endpoint {endpoint} is not a member of group {group.group_id}")
        previous = self.groups.get(group.group_id)
        if previous is not None and previous != group:
            raise CoordinatorError(f"group metadata mismatch for {group.group_id}")
        self.groups[group.group_id] = group
        self._group_registered.setdefault(group.group_id, set()).add(endpoint)
        self._next_group_seq.setdefault(group.group_id, 0)

    def _task_from_payload(self, payload: dict[str, Any]) -> TaskSpec:
        task = TaskSpec.from_dict(payload["task"])
        if task.epoch != self.epoch:
            raise CoordinatorError(f"task {task.task_id} belongs to epoch {task.epoch}")
        group = self.groups.get(task.group_id)
        if group is not None and task.group_id != group.group_id:
            raise CoordinatorError("task group mismatch")
        return task

    def _accept_task(self, endpoint: int, spec: TaskSpec, hint: TaskHint, kind: str, now: float) -> None:
        group = self.groups.get(spec.group_id)
        if group is None:
            raise CoordinatorError(f"task {spec.task_id} arrived before group registration")
        members = set(group.ranks)
        if endpoint not in members:
            raise CoordinatorError(f"endpoint {endpoint} is not a member of task {spec.task_id}")
        existing = self.tasks.get(spec.task_id)
        if existing is None:
            pair = (spec.group_id, spec.group_seq)
            if pair in self._group_task_ids and self._group_task_ids[pair] != spec.task_id:
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
            if member.declared and existing.hints[endpoint] != hint:
                raise CoordinatorError(f"declaration metadata mismatch for {spec.task_id}")
            member.declared = True
            member.declared_at = now if member.declared_at is None else member.declared_at
        else:
            if member.offered:
                raise CoordinatorError(f"duplicate offer for {spec.task_id} from endpoint {endpoint}")
            member.offered = True
            member.declared = True
            member.declared_at = now if member.declared_at is None else member.declared_at
        existing.hints[endpoint] = hint
        self.records.append({"kind": kind.lower(), "task_id": spec.task_id, "endpoint": endpoint, "now": now})

    def _progress(self, endpoint: int, kind: str, payload: dict[str, Any]) -> None:
        task_id = payload.get("task_id")
        task = self.tasks.get(task_id)
        if task is None or task.grant_seq is None:
            raise CoordinatorError(f"{kind} for ungranted task {task_id}")
        decision_seq = int(payload.get("decision_seq", -1))
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
        self.records.append({"kind": kind.lower(), "task_id": task_id, "endpoint": endpoint})
        if kind == "COMPLETED" and all(item.completed for item in task.members.values()):
            self._next_group_seq[task.spec.group_id] = task.spec.group_seq + 1
            self.inflight = None
            self.active_wait = None
            self._must_dispatch = False

    def _eligible(self) -> list[Candidate]:
        result: list[Candidate] = []
        for task in self.tasks.values():
            if task.grant_seq is not None or task.eligible_seq is not None and task.members[next(iter(task.members))].completed:
                continue
            if task.spec.group_seq != self._next_group_seq.get(task.spec.group_id, 0):
                continue
            if not all(member.offered for member in task.members.values()):
                continue
            if task.eligible_seq is None:
                self._eligible_seq += 1
                task.eligible_seq = self._eligible_seq
                self.records.append({"kind": "eligible", "task_id": task.spec.task_id, "eligible_seq": task.eligible_seq})
            hint = next(iter(task.hints.values()))
            result.append(Candidate(task.spec.task_id, hint.estimated_comm_s, hint.remaining_tail_s, task.eligible_seq))
        return result

    def _anticipated(self, now: float) -> list[Anticipated]:
        result: list[Anticipated] = []
        for task in self.tasks.values():
            if task.grant_seq is not None or task.spec.group_seq != self._next_group_seq.get(task.spec.group_id, 0):
                continue
            if all(member.offered for member in task.members.values()) or not any(member.declared for member in task.members.values()):
                continue
            hints = [task.hints[rank] for rank in task.hints]
            hint = max(hints, key=lambda item: item.ready_after_s or 0)
            declared_at = max(member.declared_at or now for member in task.members.values() if member.declared)
            predicted = declared_at + (hint.ready_after_s or 0.0)
            result.append(Anticipated(task.spec.task_id, predicted, hint.estimated_comm_s, hint.remaining_tail_s))
        return result

    def _decide(self, now: float) -> list[Outbound]:
        if self.done or self.inflight is not None:
            return []
        eligible = tuple(self._eligible())
        anticipated = tuple(self._anticipated(now))
        if self.active_wait is not None and now < self.active_wait.deadline:
            if any(item.task_id == self.active_wait.target_task_id for item in eligible):
                self.active_wait = None
            elif not self._must_dispatch:
                self.records.append({"kind": "active_wait", "target": self.active_wait.target_task_id, "now": now})
                return []
        static_next = None
        if self.policy.name == "static" and self.policy._cursor < len(self.policy.static_order):
            static_next = self.policy.static_order[self.policy._cursor]
        decision = self.policy.decide(PolicySnapshot(now, eligible, anticipated, static_next, self.active_wait, self._must_dispatch))
        if isinstance(decision, Wait):
            self.active_wait = ActiveWait(decision.target_task_id, decision.deadline, self.policy._round)
            self._must_dispatch = False
            self.records.append({"kind": "decision", "decision": "wait", "target": decision.target_task_id, "deadline": decision.deadline, "eligible": [item.task_id for item in eligible]})
            return []
        if isinstance(decision, Dispatch):
            selected = next((item for item in eligible if item.task_id == decision.task_id), None)
            if selected is None:
                raise CoordinatorError(f"policy selected ineligible task {decision.task_id}")
            self.decision_seq += 1
            task = self.tasks[selected.task_id]
            task.grant_seq = self.decision_seq
            self.inflight = task.spec.task_id
            self.active_wait = None
            self._must_dispatch = False
            out: list[Outbound] = []
            for rank in sorted(task.members):
                delivery = self.endpoint[rank].next_delivery_seq
                self.endpoint[rank].next_delivery_seq += 1
                out.append(Outbound(rank, "GRANT", {"task": task.spec.to_dict(), "decision_seq": self.decision_seq, "delivery_seq": delivery}))
            self.records.append({"kind": "decision", "decision": "dispatch", "task_id": task.spec.task_id, "decision_seq": self.decision_seq, "reason": decision.reason, "eligible": [item.task_id for item in eligible], "anticipated": [item.task_id for item in anticipated]})
            return out
        if isinstance(decision, Idle):
            self.records.append({"kind": "idle", "reason": decision.reason, "eligible": [item.task_id for item in eligible]})
        return []

    def _maybe_finish(self, now: float) -> list[Outbound]:
        if self._finish_sent or not all(state.input_closed for state in self.endpoint.values()):
            return []
        missing = [task.spec.task_id for task in self.tasks.values() if not all(member.offered for member in task.members.values())]
        if missing:
            self._fail("missing_offer", missing_tasks=missing, now=now)
            return self._failure_messages()
        if self.policy.name == "static":
            missing_static = [task_id for task_id in self.policy.static_order if task_id not in self.tasks]
            if missing_static:
                self._fail("missing_static_task", missing_tasks=missing_static, now=now)
                return self._failure_messages()
        if self.inflight is not None or any(not all(member.completed for member in task.members.values()) for task in self.tasks.values()):
            return []
        self.finished = True
        self._finish_sent = True
        payload = {"epoch": self.epoch, "decision_seq": self.decision_seq, "task_count": len(self.tasks)}
        self.records.append({"kind": "finished", **payload})
        return [Outbound(rank, "FINISHED", payload) for rank in self.endpoints]

    def _fail(self, reason: str, **details: Any) -> None:
        if self.failed is None:
            self.failed = {"epoch": self.epoch, "reason": reason, **details}
            self.records.append({"kind": "failed", **self.failed})

    def _failure_messages(self) -> list[Outbound]:
        if self.failed is None:
            return []
        return [Outbound(rank, "FAILED", self.failed) for rank in self.endpoints]
