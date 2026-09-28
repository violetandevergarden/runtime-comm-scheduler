from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from examples.jobpacer.runtime.gpu_workload import (
    FIFO_ESTIMATOR_VERSION,
    GPU_TAIL_ESTIMATOR_VERSION,
    parse_gpu_linear,
)


def _segment(segment_id="s0", group_seq=0):
    return {
        "segment_id": segment_id, "group_id": "g", "group_seq": group_seq,
        "producer_compute": {"op": "matmul", "m": 2, "n": 2, "k": 3,
                              "dtype": "float32", "layout": "contiguous", "repeats": 2,
                              "output_role": "collective_input"},
        "collective": {"op": "all_reduce", "reduction": "sum", "shape": [2, 2],
                       "dtype": "float32", "numel": 4, "num_bytes": 16},
        "independent_compute": {"op": "matmul", "m": 2, "n": 2, "k": 2,
                                "dtype": "float32", "layout": "contiguous", "repeats": 1,
                                "input_role": "private_inputs", "output_role": "independent_output"},
        "dependent_compute": {"op": "sum_join", "inputs": ["collective_output", "independent_output"]},
        "estimates": {"compute_profile_ref": "profiles/compute.json",
                      "comm_profile_ref": "profiles/comm.json",
                      "estimator_version": FIFO_ESTIMATOR_VERSION},
    }


def _payload():
    return {"schema": "jobpacer-gpu-linear", "schema_version": 1, "name": "unit",
            "execution_seed": 44, "groups": [{"group_id": "g", "ranks": [0, 1]}],
            "execution_contract": {"producer_readiness": "physical-ready",
                                   "submit_order": "comm-request-before-independent",
                                   "segment_advance": "terminal-event-complete", "max_inflight": 1},
            "jobs": [{"job_id": "job", "segments": [_segment()]}]}


def test_gpu_linear_parser_normalizes_digest_and_assigns_stable_task_identity():
    raw = _payload()
    parsed = parse_gpu_linear(raw, world_size=2)
    reordered = json.loads(json.dumps(raw, sort_keys=True))
    assert parse_gpu_linear(reordered, world_size=2).manifest_digest == parsed.manifest_digest
    segment = parsed.jobs[0].segments[0]
    assert segment.task_id == "job/s0"
    assert parsed.expected_task_ids == ("job/s0",)


@pytest.mark.parametrize(("mutate", "message"), [
    (lambda x: x.update(schema_version=True), "schema"),
    (lambda x: x["jobs"][0]["segments"][0]["collective"].update(num_bytes=15), "num_bytes"),
    (lambda x: x["jobs"][0]["segments"][0]["producer_compute"].update(repeats=True), "positive integers"),
    (lambda x: x["jobs"][0]["segments"][0]["producer_compute"].update(_compute_s=0.01), "unknown"),
    (lambda x: x["jobs"][0]["segments"][0]["dependent_compute"].update(
        inputs=["collective_output", "unavailable_output"]), "available"),
    (lambda x: x["jobs"][0]["segments"][0]["estimates"].update(
        compute_profile_ref=None, estimator_version=GPU_TAIL_ESTIMATOR_VERSION),
     "require compute and comm"),
    (lambda x: x["execution_contract"].update(max_inflight=True),
     "max_inflight must be a non-boolean integer"),
])
def test_gpu_linear_parser_rejects_invalid_contracts(mutate, message):
    raw = copy.deepcopy(_payload())
    mutate(raw)
    with pytest.raises(ValueError, match=message):
        parse_gpu_linear(raw, world_size=2)


def test_gpu_linear_parser_rejects_duplicate_or_noncontiguous_group_sequence():
    raw = _payload()
    raw["jobs"][0]["segments"].append(_segment("s1", 0))
    with pytest.raises(ValueError, match="group_seq"):
        parse_gpu_linear(raw, world_size=2)
    raw["jobs"][0]["segments"][1]["group_seq"] = 2
    with pytest.raises(ValueError, match="group_seq"):
        parse_gpu_linear(raw, world_size=2)


def test_gpu_linear_canonical_digest_preserves_job_order():
    raw = _payload()
    raw["jobs"].append({"job_id": "job-2", "segments": [_segment("s1", 1)]})
    reordered = copy.deepcopy(raw)
    reordered["jobs"].reverse()
    assert parse_gpu_linear(raw).manifest_digest != parse_gpu_linear(reordered).manifest_digest
