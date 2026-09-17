"""Static FIFO and longest-tail-first Plan construction for JobPacer."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from runtime_comm_scheduler import Plan, TaskKey

try:  # Support both package imports and direct worker script execution.
    from .workloads import CollectiveComm, Job, Workload
except ImportError:  # pragma: no cover - direct script execution
    from workloads import CollectiveComm, Job, Workload


@dataclass(frozen=True)
class PlannedTask:
    """A plan entry plus the source job metadata used by the replay."""

    job_id: str
    communication: CollectiveComm
    key: TaskKey


PolicyBuilder = Callable[
    [Workload, dict[str, tuple[PlannedTask, ...]]], list[PlannedTask]
]
_POLICY_BUILDERS: dict[str, PolicyBuilder] = {}


def register_policy(name: str, builder: PolicyBuilder) -> None:
    """Register a deterministic static policy builder."""
    if not name or not name.isidentifier():
        raise ValueError(f"policy name must be a non-empty identifier: {name!r}")
    if name in _POLICY_BUILDERS:
        raise ValueError(f"policy {name!r} is already registered")
    _POLICY_BUILDERS[name] = builder


def policy_names() -> tuple[str, ...]:
    """Return registered policy names in deterministic registration order."""
    return tuple(_POLICY_BUILDERS)


def _ordered_tasks(
    workload: Workload,
    tasks: dict[str, tuple[PlannedTask, ...]],
    policy: str,
) -> list[PlannedTask]:
    try:
        builder = _POLICY_BUILDERS[policy]
    except KeyError as exc:
        choices = ", ".join(policy_names())
        raise ValueError(f"unknown policy {policy!r}; choices={choices}") from exc
    return builder(workload, tasks)


def task_key(job_id: str, ordinal: int, *, window_id: int = 0) -> TaskKey:
    """Create the seven-field deterministic key required by the core API."""
    return TaskKey(
        iteration=window_id,
        microbatch=0,
        parallelism="jobpacer",
        process_group_id=job_id,
        layer_id=ordinal,
        bucket_id=0,
        ordinal=ordinal,
    )


def _planned_tasks(
    workload: Workload, *, window_id: int
) -> dict[str, tuple[PlannedTask, ...]]:
    return {
        job.job_id: tuple(
            PlannedTask(
                job_id=job.job_id,
                communication=communication,
                key=task_key(job.job_id, communication.id, window_id=window_id),
            )
            for communication in job.communications
        )
        for job in workload.jobs
    }


def _tail_after(tasks: tuple[PlannedTask, ...], index: int) -> float:
    """Estimate the critical path remaining after the candidate completes."""

    current = tasks[index].communication
    return max(
        current.consumer_compute_s - current.estimated_comm_s, 0.0
    ) + sum(
        item.communication.producer_compute_s
        + max(
            item.communication.estimated_comm_s,
            item.communication.consumer_compute_s,
        )
        for item in tasks[index + 1 :]
    )


def _fifo_order(
    workload: Workload, tasks: dict[str, tuple[PlannedTask, ...]]
) -> list[PlannedTask]:
    ordered: list[PlannedTask] = []
    for ordinal in range(max(len(job.communications) for job in workload.jobs)):
        for job in workload.jobs:
            if ordinal < len(tasks[job.job_id]):
                ordered.append(tasks[job.job_id][ordinal])
    return ordered


def _ltf_order(
    workload: Workload, tasks: dict[str, tuple[PlannedTask, ...]]
) -> list[PlannedTask]:
    positions = {job.job_id: 0 for job in workload.jobs}
    ordered: list[PlannedTask] = []

    while len(ordered) < sum(len(job.communications) for job in workload.jobs):
        candidates = []
        for job_index, job in enumerate(workload.jobs):
            position = positions[job.job_id]
            if position < len(tasks[job.job_id]):
                candidate = tasks[job.job_id][position]
                candidates.append(
                    (
                        _tail_after(tasks[job.job_id], position),
                        job.job_id,
                        candidate.communication.id,
                        job_index,
                        candidate,
                    )
                )

        # Largest tail first.  The ascending job_id/ordinal tie break is
        # independent of thread start order and of object identity.
        _tail, _job_id, _ordinal, _job_index, selected = sorted(
            candidates, key=lambda item: (-item[0], item[1], item[2], item[3])
        )[0]
        ordered.append(selected)
        positions[selected.job_id] += 1
    return ordered


def _validate_job_sequences(ordered: list[PlannedTask], workload: Workload) -> None:
    positions = {job.job_id: -1 for job in workload.jobs}
    for item in ordered:
        expected = positions[item.job_id] + 1
        if item.communication.id != expected:
            raise ValueError(
                f"plan violates job {item.job_id!r} dependency: "
                f"got ordinal {item.communication.id}, expected {expected}"
            )
        positions[item.job_id] = item.communication.id
    for job in workload.jobs:
        if positions[job.job_id] != len(job.communications) - 1:
            raise ValueError(
                f"plan does not contain every communication for job {job.job_id!r}"
            )


def build_plan(
    workload: Workload,
    policy: str = "fifo",
    *,
    version: int = 0,
    window_id: int = 0,
) -> Plan:
    """Build and validate a deterministic static plan."""

    tasks = _planned_tasks(workload, window_id=window_id)
    ordered = _ordered_tasks(workload, tasks, policy)
    _validate_job_sequences(ordered, workload)
    return Plan(
        version=version,
        window_id=window_id,
        entries=tuple(
            (item.key, item.communication.op, item.communication.num_bytes)
            for item in ordered
        ),
    )


def planned_tasks(
    workload: Workload, policy: str = "fifo", *, window_id: int = 0
) -> dict[TaskKey, PlannedTask]:
    """Return source metadata indexed by key in the chosen plan."""

    tasks = _planned_tasks(workload, window_id=window_id)
    ordered = _ordered_tasks(workload, tasks, policy)
    _validate_job_sequences(ordered, workload)
    return {item.key: item for item in ordered}


def key_labels(plan: Plan) -> list[str]:
    """Human-readable stable key sequence for result documents."""
    return [f"{key.process_group_id}:{key.ordinal}" for key in plan.keys]


register_policy("fifo", _fifo_order)
register_policy("ltf", _ltf_order)
