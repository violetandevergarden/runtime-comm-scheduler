"""Deterministic hand-written workloads used by the JobPacer replay.

The manifest is deliberately small.  A job is a linear sequence of
``CommunicationSpec`` objects.  ``producer_compute_s`` runs before the
collective is submitted and ``consumer_compute_s`` runs between submission and
the consumer's ``wait``.  This preserves a useful compute/communication
overlap window while keeping the dependency graph explicit.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class CollectiveComm:
    """
    用于描述一个job的一个集合通信
    id：此通信在本job中按次序的标号
    num_bytes：通信字节数
    op：集合通信操作名称
    producer_compute_s：上游产生数据需要的计算时间
    consumer_compute_s：下游暂时不依赖此次通信的计算时间，模拟计算和通信重叠
    estimated_comm_s：预估通信时间（秒）
    """

    id: int
    num_bytes: int = 4096
    op: str = "all_reduce"
    producer_compute_s: float = 0.0
    consumer_compute_s: float = 0.0
    estimated_comm_s: float = 0.001

    def __post_init__(self) -> None:
        if self.id < 0:
            raise ValueError("id must be non-negative")
        if self.op != "all_reduce":
            raise ValueError("phase 2 currently supports only all_reduce")
        if self.num_bytes <= 0 or self.num_bytes % 4:
            raise ValueError("num_bytes must be a positive multiple of 4")
        for name in (
            "producer_compute_s",
            "consumer_compute_s",
            "estimated_comm_s",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class Job:
    """One linear replay job with its own logical ProcessGroup."""

    job_id: str
    communications: tuple[CollectiveComm, ...]
    ranks: tuple[int, ...] | None = None     # 此job使用的进程组

    def __post_init__(self) -> None:
        if not self.job_id:
            raise ValueError("job_id must not be empty")
        ids = tuple(item.id for item in self.communications)
        if ids != tuple(range(len(ids))):
            raise ValueError(
                f"job {self.job_id!r} ordinals must be consecutive from zero"
            )
        if not self.communications:
            raise ValueError(f"job {self.job_id!r} must contain a communication")
        if self.ranks is not None and (
            not self.ranks or len(set(self.ranks)) != len(self.ranks)
        ):
            raise ValueError(f"job {self.job_id!r} ranks must be non-empty and unique")


@dataclass(frozen=True)
class Workload:
    """A replay manifest shared by all ranks."""

    name: str
    jobs: tuple[Job, ...]
    seed: int = 0  # 预留可能的随机数

    def __post_init__(self) -> None:
        if not self.jobs:
            raise ValueError("workload must contain at least one job")
        ids = tuple(job.job_id for job in self.jobs)
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate job_id in workload: {ids}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "seed": self.seed,
            "jobs": [
                {
                    "job_id": job.job_id,
                    "ranks": list(job.ranks) if job.ranks is not None else None,
                    "communications": [
                        asdict(communication) for communication in job.communications
                    ],
                }
                for job in self.jobs
            ],
        }

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> "Workload":
        """Build a validated workload from a JSON-compatible manifest."""
        jobs = []
        for job_document in document["jobs"]:
            communications = tuple(
                CollectiveComm(**communication)
                for communication in job_document["communications"]
            )
            configured_ranks = job_document.get("ranks")
            jobs.append(
                Job(
                    job_id=job_document["job_id"],
                    communications=communications,
                    ranks=None if configured_ranks is None else tuple(configured_ranks),
                )
            )
        return cls(
            name=str(document.get("name", "manifest")),
            seed=int(document.get("seed", 0)),
            jobs=tuple(jobs),
        )


def _job(
    job_id: str, timings: list[tuple[float, float, float]], *, num_bytes: int = 4096
) -> Job:
    return Job(
        job_id=job_id,
        communications=tuple(
            CollectiveComm(
                id=ordinal,
                num_bytes=num_bytes,
                producer_compute_s=producer,
                consumer_compute_s=consumer,
                estimated_comm_s=estimated_comm,
            )
            for ordinal, (producer, consumer, estimated_comm) in enumerate(timings)
        ),
    )


def built_workload(name: str) -> Workload:
    """Return one of the small deterministic phase 2 workloads.

    ``balanced`` is the default replay.  ``tail`` makes FIFO and LTF choose
    different first tasks.  ``delayed`` keeps the FIFO head unready briefly,
    exercising the scheduler's head-of-line wait and bounded finish path.
    """

    if name in ("default", "balanced"):
        return Workload(
            name="balanced",
            jobs=(
                _job(
                    "job-0",
                    [
                        (0.005, 0.002, 0.001),
                        (0.002, 0.001, 0.001),
                        (0.001, 0.001, 0.001),
                    ],
                ),
                _job(
                    "job-1",
                    [
                        (0.002, 0.001, 0.001),
                        (0.004, 0.001, 0.001),
                        (0.001, 0.002, 0.001),
                    ],
                ),
            ),
        )
    if name == "tail":
        return Workload(
            name="tail",
            jobs=(
                _job(
                    "job-0",
                    [
                        (0.001, 0.001, 0.001),
                        (0.001, 0.001, 0.001),
                        (0.001, 0.012, 0.001),
                    ],
                ),
                _job(
                    "job-1",
                    [
                        (0.001, 0.001, 0.001),
                        (0.001, 0.001, 0.001),
                        (0.001, 0.001, 0.001),
                    ],
                ),
            ),
        )
    if name == "delayed":
        return Workload(
            name="delayed",
            jobs=(
                _job("job-0", [(0.080, 0.002, 0.001), (0.001, 0.001, 0.001)]),
                _job("job-1", [(0.001, 0.001, 0.001), (0.001, 0.001, 0.001)]),
            ),
        )
    raise ValueError(
        f"unknown built-in workload {name!r}; choices=balanced, tail, delayed"
    )


def load_workload(value: str | Path) -> Workload:
    """Load a built-in workload or a JSON manifest path."""

    path = Path(value)
    if path.exists():
        return Workload.from_dict(json.loads(path.read_text()))
    return built_workload(str(value))


def ranks_for_job(job: Job, world_size: int) -> tuple[int, ...]:
    """Return and validate a job's global ProcessGroup rank list."""
    
    ranks = tuple(range(world_size)) if job.ranks is None else job.ranks
    if not ranks or any(rank < 0 or rank >= world_size for rank in ranks):
        raise ValueError(f"job {job.job_id!r} has invalid ranks {ranks}")
    return ranks
