"""Legacy Workload to runtime bindings kept for CPU/Gloo replay."""

from __future__ import annotations

from typing import Any

from runtime_comm_scheduler.dag import DagJob
from runtime_comm_scheduler.runtime import CollectiveSpec, GroupSpec, TaskHint, TaskSpec
from examples.jobpacer.gloo.workloads import (
    CollectiveComm, Job, Workload, sample_linear_duration,
)
from examples.jobpacer.runtime.runtime_adapter import (
    ReplayExecutionConfig, make_replay_compute, sample_compute_duration,
)


def make_gloo_dag_compute(job: DagJob, execution: ReplayExecutionConfig, *, seed: int,
                          epoch: int, rank: int, jitter: float, samples, event_log):
    """Bind schema-v1 Gloo DAG compute with its optional linear sampling keys."""
    def duration(seed_value: int, epoch_value: int, job_id: str, node_id: str,
                 rank_value: int, base_s: float, jitter_value: float) -> float:
        sample_key = (execution.linear_sample_keys or {}).get(f"{job_id}/{node_id}")
        if sample_key is None:
            return sample_compute_duration(seed_value, epoch_value, job_id, node_id,
                                           rank_value, base_s, jitter_value)
        ordinal, segment = sample_key
        return sample_linear_duration(seed_value, epoch_value, job_id, ordinal,
                                      rank_value, segment, base_s, jitter_value)

    return make_replay_compute(job, execution, seed=seed, epoch=epoch, rank=rank,
                               jitter=jitter, samples=samples, event_log=event_log,
                               duration_sampler=duration)


def linear_static_order(workload: Workload, policy: str) -> tuple[str, ...]:
    """Construct the runtime's static order for a Gloo linear workload."""
    if policy == "static_fifo":
        return tuple(
            f"{job.job_id}/comm-{index}"
            for index in range(max(len(job.communications) for job in workload.jobs))
            for job in workload.jobs if index < len(job.communications)
        )
    if policy != "static_ltf":
        raise ValueError("linear static order requires static_fifo or static_ltf")
    positions = {job.job_id: 0 for job in workload.jobs}
    order: list[str] = []
    total = sum(len(job.communications) for job in workload.jobs)
    while len(order) < total:
        candidates = []
        for job in workload.jobs:
            index = positions[job.job_id]
            if index >= len(job.communications):
                continue
            comm = job.communications[index]
            tail = remaining_tail(job, index)
            candidates.append((-(comm.estimated_comm_s + tail), job.job_id, index))
        _negative_score, job_id, index = min(candidates)
        order.append(f"{job_id}/comm-{index}")
        positions[job_id] = index + 1
    return tuple(order)


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
    return max(current.consumer_compute_s - current.estimated_comm_s, 0.0) + sum(
        item.producer_compute_s + max(item.estimated_comm_s, item.consumer_compute_s)
        for item in job.communications[index + 1 :]
    )


def task_hint(job: Job, index: int) -> TaskHint:
    comm = job.communications[index]
    return TaskHint(comm.producer_compute_s, comm.estimated_comm_s,
                    remaining_tail(job, index))


def linear_ltf_estimates(workload: Workload) -> list[dict[str, Any]]:
    """Expose the c/u/tail inputs and score used by the Gloo linear LTF path."""
    rows = []
    for job in workload.jobs:
        for index, comm in enumerate(job.communications):
            tail = remaining_tail(job, index)
            rows.append({
                "task_id": f"{job.job_id}/comm-{comm.id}",
                "estimated_comm_s": comm.estimated_comm_s,
                "independent_consumer_s": comm.consumer_compute_s,
                "remaining_tail_s": tail,
                "ltf_score_s": comm.estimated_comm_s + tail,
            })
    return rows


def all_specs(workload: Workload, *, epoch: int = 0) -> tuple[tuple[Job, int, TaskSpec, TaskHint], ...]:
    return tuple(
        (job, index, task_spec(job, comm, epoch=epoch), task_hint(job, index))
        for job in workload.jobs
        for index, comm in enumerate(job.communications)
    )
