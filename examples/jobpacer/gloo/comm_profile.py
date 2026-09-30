"""Apply shared communication profiles to the legacy Gloo Workload model."""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import replace
from typing import Any, Mapping

from examples.jobpacer.gloo.workloads import Workload, ranks_for_job
from examples.jobpacer.runtime.comm_profile import (
    CommSignature, CommunicationProfile, load_profile,
)


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def workload_digest(workload: Workload) -> str:
    return hashlib.sha256(_stable_json(workload.to_dict()).encode()).hexdigest()


def apply_profile(
    workload: Workload,
    profile: CommunicationProfile,
    environment: Mapping[str, Any],
    *,
    strict: bool = True,
) -> Workload:
    """Apply matching profile estimates to a Gloo linear workload."""
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
                communication, group_size=group_size, backend=backend,
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


__all__ = ["apply_profile", "load_profile", "workload_digest"]
