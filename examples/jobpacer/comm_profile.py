"""Validated communication profiles for JobPacer replay."""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping

try:
    from .workloads import Job, Workload, ranks_for_job
except ImportError:  # pragma: no cover - direct script execution
    from workloads import Job, Workload, ranks_for_job


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True, order=True)
class CommSignature:
    op: str
    num_bytes: int
    dtype: str
    group_size: int
    backend: str
    device_type: str
    reduction: str

    def __post_init__(self) -> None:
        if self.op != "all_reduce" or self.reduction != "sum":
            raise ValueError("only all_reduce/sum profiles are supported")
        if self.num_bytes <= 0 or self.group_size <= 0:
            raise ValueError("num_bytes and group_size must be positive")
        if self.dtype != "float32":
            raise ValueError("phase 2 profiles require dtype=float32")
        if (self.backend, self.device_type) not in (("gloo", "cpu"), ("nccl", "cuda")):
            raise ValueError("backend/device_type must be gloo/cpu or nccl/cuda")

    @classmethod
    def for_communication(
        cls, communication, *, group_size: int, backend: str, device_type: str
    ) -> "CommSignature":
        return cls(
            op=communication.op,
            num_bytes=communication.num_bytes,
            dtype="float32",
            group_size=group_size,
            backend=backend,
            device_type=device_type,
            reduction="sum",
        )


@dataclass(frozen=True)
class ProfileRecord:
    op: str
    num_bytes: int
    dtype: str
    group_size: int
    backend: str
    device_type: str
    reduction: str
    p50_s: float
    p10_s: float
    p90_s: float
    mean_s: float
    stdev_s: float
    samples: int

    def __post_init__(self) -> None:
        CommSignature(**{name: getattr(self, name) for name in CommSignature.__dataclass_fields__})
        values = (self.p10_s, self.p50_s, self.p90_s, self.mean_s, self.stdev_s)
        if not all(math.isfinite(value) and value >= 0 for value in values):
            raise ValueError("profile timings must be finite and non-negative seconds")
        if self.p10_s > self.p50_s or self.p50_s > self.p90_s:
            raise ValueError("profile percentiles must satisfy p10 <= p50 <= p90")
        if self.p50_s <= 0 or self.samples <= 0:
            raise ValueError("p50_s and samples must be positive")

    @property
    def signature(self) -> CommSignature:
        return CommSignature(**{name: getattr(self, name) for name in CommSignature.__dataclass_fields__})


@dataclass(frozen=True)
class CommunicationProfile:
    schema_version: int
    environment: Mapping[str, Any]
    settings: Mapping[str, Any]
    records: tuple[ProfileRecord, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"unsupported communication profile schema {self.schema_version}")
        signatures = [record.signature for record in self.records]
        if len(signatures) != len(set(signatures)):
            raise ValueError("communication profile contains duplicate signatures")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "environment": dict(self.environment),
            "settings": dict(self.settings),
            "records": [asdict(record) for record in self.records],
        }

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> "CommunicationProfile":
        return cls(
            schema_version=int(document["schema_version"]),
            environment=dict(document["environment"]),
            settings=dict(document.get("settings", {})),
            records=tuple(ProfileRecord(**record) for record in document["records"]),
        )

    def digest(self) -> str:
        return hashlib.sha256(_stable_json(self.to_dict()).encode()).hexdigest()


def load_profile(path: str | Path) -> CommunicationProfile:
    return CommunicationProfile.from_dict(json.loads(Path(path).read_text()))


def workload_digest(workload: Workload) -> str:
    return hashlib.sha256(_stable_json(workload.to_dict()).encode()).hexdigest()


def apply_profile(
    workload: Workload,
    profile: CommunicationProfile,
    environment: Mapping[str, Any],
    *,
    strict: bool = True,
) -> Workload:
    """Return a copy of ``workload`` with p50 communication estimates applied."""

    required_environment = ("backend", "world_size", "device_type")
    mismatches = [
        f"{key}: profile={profile.environment.get(key)!r}, replay={environment.get(key)!r}"
        for key in required_environment
        if profile.environment.get(key) != environment.get(key)
    ]
    if mismatches:
        raise ValueError("profile environment mismatch: " + "; ".join(mismatches))

    world_size = int(environment["world_size"])
    expected_groups: list[list[int]] = []
    for job in workload.jobs:
        ranks = list(ranks_for_job(job, world_size))
        if ranks not in expected_groups:
            expected_groups.append(ranks)
    if profile.environment.get("group_ranks") != expected_groups:
        raise ValueError(
            "profile environment mismatch: group_ranks: "
            f"profile={profile.environment.get('group_ranks')!r}, replay={expected_groups!r}"
        )

    records = {record.signature: record for record in profile.records}
    missing: list[str] = []
    jobs = []
    backend = str(environment["backend"])
    device_type = str(environment["device_type"])
    for job in workload.jobs:
        communications = []
        group_size = len(ranks_for_job(job, world_size))
        for communication in job.communications:
            signature = CommSignature.for_communication(
                communication,
                group_size=group_size,
                backend=backend,
                device_type=device_type,
            )
            record = records.get(signature)
            if record is None:
                missing.append(f"{job.job_id}:{communication.id} {signature}")
                communications.append(communication)
            else:
                communications.append(replace(communication, estimated_comm_s=record.p50_s))
        jobs.append(replace(job, communications=tuple(communications)))
    if missing and strict:
        raise ValueError("profile is missing communication signatures: " + ", ".join(missing))
    if missing:
        warnings.warn("profile fallback to manifest for: " + ", ".join(missing), stacklevel=2)
    return replace(workload, jobs=tuple(jobs))
