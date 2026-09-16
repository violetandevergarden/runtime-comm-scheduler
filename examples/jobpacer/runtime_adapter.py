"""Map the hand-written JobPacer workload to the Stage 3.1 runtime model."""

from __future__ import annotations

from runtime_comm_scheduler.runtime import CollectiveSpec, GroupSpec, TaskHint, TaskSpec

try:
    from .workloads import CollectiveComm, Job, Workload
except ImportError:  # pragma: no cover - direct worker execution
    from workloads import CollectiveComm, Job, Workload


def group_spec(job: Job, world_size: int, *, epoch: int = 0) -> GroupSpec:
    ranks = tuple(range(world_size)) if job.ranks is None else tuple(job.ranks)
    return GroupSpec(epoch, job.job_id, ranks)


def task_spec(job: Job, comm: CollectiveComm, *, epoch: int = 0) -> TaskSpec:
    numel = comm.num_bytes // 4
    return TaskSpec(
        epoch=epoch,
        job_id=job.job_id,
        task_id=f"{job.job_id}/comm-{comm.id}",
        group_id=job.job_id,
        group_seq=comm.id,
        collective=CollectiveSpec("all_reduce", numel, comm.num_bytes, "float32", (numel,)),
    )


def remaining_tail(job: Job, index: int) -> float:
    current = job.communications[index]
    return current.consumer_compute_s + sum(
        item.producer_compute_s + item.estimated_comm_s + item.consumer_compute_s
        for item in job.communications[index + 1 :]
    )


def task_hint(job: Job, index: int) -> TaskHint:
    comm = job.communications[index]
    return TaskHint(comm.producer_compute_s, comm.estimated_comm_s, remaining_tail(job, index))


def all_specs(workload: Workload, *, epoch: int = 0) -> tuple[tuple[Job, int, TaskSpec, TaskHint], ...]:
    return tuple(
        (job, index, task_spec(job, comm, epoch=epoch), task_hint(job, index))
        for job in workload.jobs
        for index, comm in enumerate(job.communications)
    )
