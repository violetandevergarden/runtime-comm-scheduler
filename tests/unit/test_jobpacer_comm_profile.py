"""Communication profile validation and application checks."""

import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from examples.jobpacer.comm_profile import (
    CommSignature,
    CommunicationProfile,
    ProfileRecord,
    apply_profile,
    load_profile,
)
from examples.jobpacer.plan_builder import build_plan, key_labels
from examples.jobpacer.workloads import CollectiveComm, Job, Workload


ENVIRONMENT = {
    "backend": "gloo",
    "device_type": "cpu",
    "world_size": 2,
    "group_ranks": [[0, 1]],
}


def _record(num_bytes: int, p50_s: float) -> ProfileRecord:
    signature = CommSignature("all_reduce", num_bytes, "float32", 2, "gloo", "cpu", "sum")
    return ProfileRecord(
        **asdict(signature), p50_s=p50_s, p10_s=p50_s * 0.9,
        p90_s=p50_s * 1.1, mean_s=p50_s, stdev_s=0.0, samples=5,
    )


def _profile(*records: ProfileRecord) -> CommunicationProfile:
    return CommunicationProfile(1, ENVIRONMENT, {"warmup": 1, "iterations": 5}, records)


def test_signature_deduplicates_across_jobs():
    workload = Workload("same", (
        Job("a", (CollectiveComm(0, num_bytes=4096),)),
        Job("b", (CollectiveComm(0, num_bytes=4096),)),
    ))
    signatures = {
        CommSignature.for_communication(item, group_size=2, backend="gloo", device_type="cpu")
        for job in workload.jobs for item in job.communications
    }
    assert len(signatures) == 1


def test_profile_round_trip_and_digest_are_deterministic(tmp_path):
    profile = _profile(_record(4096, 0.003))
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile.to_dict(), sort_keys=True))
    restored = load_profile(path)
    assert restored == profile
    assert restored.digest() == profile.digest()


def test_apply_profile_returns_copy_and_only_replaces_estimate():
    original = Workload("apply", (Job("a", (CollectiveComm(0, consumer_compute_s=0.4),)),))
    applied = apply_profile(original, _profile(_record(4096, 0.007)), ENVIRONMENT)
    assert original.jobs[0].communications[0].estimated_comm_s == 0.001
    assert applied.jobs[0].communications[0].estimated_comm_s == 0.007
    assert applied.jobs[0].communications[0].consumer_compute_s == 0.4


def test_strict_application_rejects_missing_duplicate_and_environment_mismatch():
    workload = Workload("strict", (Job("a", (CollectiveComm(0),)),))
    with pytest.raises(ValueError, match="missing"):
        apply_profile(workload, _profile(), ENVIRONMENT)
    with pytest.raises(ValueError, match="duplicate"):
        _profile(_record(4096, 0.1), _record(4096, 0.2))
    with pytest.raises(ValueError, match="environment mismatch"):
        apply_profile(workload, _profile(_record(4096, 0.1)), ENVIRONMENT | {"backend": "nccl"})


def test_profile_estimate_changes_ltf_order_without_reordering_a_job():
    workload = Workload("ltf", (
        Job("a", (CollectiveComm(0), CollectiveComm(1, num_bytes=8192))),
        Job("b", (CollectiveComm(0), CollectiveComm(1, num_bytes=4096, consumer_compute_s=0.01))),
    ))
    before = key_labels(build_plan(workload, "ltf"))
    applied = apply_profile(workload, _profile(_record(4096, 0.001), _record(8192, 0.1)), ENVIRONMENT)
    after = key_labels(build_plan(applied, "ltf"))
    assert before != after
    assert [key.ordinal for key in build_plan(applied, "ltf").group_sequence("a")] == [0, 1]
