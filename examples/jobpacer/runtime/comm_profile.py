"""Validated communication profiles for JobPacer replay."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping



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
    profile_service_time_s: float | None = None
    api_call_duration_p50_s: float | None = None
    return_to_sync_p50_s: float | None = None

    def __post_init__(self) -> None:
        CommSignature(**{name: getattr(self, name) for name in CommSignature.__dataclass_fields__})
        values = (self.p10_s, self.p50_s, self.p90_s, self.mean_s, self.stdev_s)
        if not all(math.isfinite(value) and value >= 0 for value in values):
            raise ValueError("profile timings must be finite and non-negative seconds")
        if self.p10_s > self.p50_s or self.p50_s > self.p90_s:
            raise ValueError("profile percentiles must satisfy p10 <= p50 <= p90")
        if self.p50_s <= 0 or self.samples <= 0:
            raise ValueError("p50_s and samples must be positive")
        if self.profile_service_time_s is None:
            object.__setattr__(self, "profile_service_time_s", self.p50_s)

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

