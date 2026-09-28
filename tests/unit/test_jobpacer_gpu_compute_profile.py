from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from examples.jobpacer.runtime.gpu_compute_profile import (
    dag_compute_profile_signature,
    load_gpu_compute_profile,
)
from examples.jobpacer.runtime.gpu_workload import GpuComputeSpec


def _profile():
    return {
        "schema": "jobpacer-gpu-compute-profile", "schema_version": 2,
        "software": {"pytorch_version": "2.x", "cuda_version": "12.x",
                     "matmul_precision": "highest", "allow_tf32": False},
        "devices": [{"device_uuid": "GPU-0", "records": [
            {"stage": "producer", "signature": {"op": "matmul", "m": 2, "n": 2, "k": 3,
             "dtype": "float32", "layout": "contiguous", "repeats": 2,
             "output_role": "collective_input"},
             "device_event_ms_samples": [1.0, 3.0, 2.0],
             "host_enqueue_us_samples": [4.0, 5.0, 6.0], "preparation_us": 10.0},
        ]}],
    }


def test_compute_profile_matches_device_software_and_full_program_signature(tmp_path):
    path = tmp_path / "compute.json"
    path.write_text(json.dumps(_profile()))
    profile = load_gpu_compute_profile(path)
    spec = GpuComputeSpec("matmul", m=2, n=2, k=3, repeats=2,
                          output_role="collective_input")
    record = profile.record(device_uuid="GPU-0", stage="producer", spec=spec,
                            software={"pytorch_version": "2.x", "cuda_version": "12.x",
                                      "matmul_precision": "highest", "allow_tf32": False})
    assert record.device_event_p50_s == pytest.approx(0.002)
    assert len(profile.digest) == 64
    with pytest.raises(ValueError, match="UUID"):
        profile.record(device_uuid="GPU-1", stage="producer", spec=spec)
    with pytest.raises(ValueError, match="software mismatch"):
        profile.record(device_uuid="GPU-0", stage="producer", spec=spec,
                       software={"pytorch_version": "wrong", "cuda_version": "12.x",
                                 "matmul_precision": "highest", "allow_tf32": False})


def test_compute_profile_rejects_duplicate_signatures_and_invalid_samples(tmp_path):
    raw = _profile()
    raw["devices"][0]["records"].append(raw["devices"][0]["records"][0])
    path = tmp_path / "duplicate.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="duplicate"):
        load_gpu_compute_profile(path)
    raw = _profile()
    raw["devices"][0]["records"][0]["device_event_ms_samples"] = [float("nan")]
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="finite"):
        load_gpu_compute_profile(path)


def test_compute_profile_rejects_legacy_v1_signatures(tmp_path):
    raw = _profile()
    raw["schema_version"] = 1
    path = tmp_path / "legacy-v1.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="unsupported GPU compute profile schema/version"):
        load_gpu_compute_profile(path)


def test_fill_and_sum_join_profile_signatures_include_tensor_shape_dtype_and_layout(tmp_path):
    raw = _profile()
    raw["devices"][0]["records"] = [
        {"stage": "producer", "signature": {
            "op": "fill", "output_role": "collective_input", "shape": [2, 3],
            "dtype": "float32", "layout": "contiguous"},
         "device_event_ms_samples": [0.2], "host_enqueue_us_samples": [3.0],
         "preparation_us": 4.0},
        {"stage": "dependent", "signature": {
            "op": "sum_join", "inputs": ["collective_output", "independent_output"],
            "shape": [2, 3], "dtype": "float32", "layout": "contiguous"},
         "device_event_ms_samples": [0.4], "host_enqueue_us_samples": [5.0],
         "preparation_us": 6.0},
    ]
    path = tmp_path / "shape-specific.json"
    path.write_text(json.dumps(raw))
    profile = load_gpu_compute_profile(path)

    fill = GpuComputeSpec("fill", output_role="collective_input")
    assert profile.record(device_uuid="GPU-0", stage="producer", spec=fill,
                          tensor_shape=(2, 3)).signature["shape"] == [2, 3]
    join = {"op": "sum_join", "inputs": ["collective_output", "independent_output"]}
    assert profile.record(device_uuid="GPU-0", stage="dependent", spec=join,
                          tensor_shape=(2, 3)).signature["dtype"] == "float32"
    with pytest.raises(ValueError, match="signature missing"):
        profile.record(device_uuid="GPU-0", stage="producer", spec=fill,
                       tensor_shape=(3, 2))
    with pytest.raises(ValueError, match="signature missing"):
        profile.record(device_uuid="GPU-0", stage="dependent", spec=join,
                       tensor_shape=(2, 4))


@pytest.mark.parametrize(("field", "value", "message"), [
    ("shape", [2, True], "shape must contain positive integers"),
    ("dtype", "float64", "contiguous float32 tensors"),
    ("layout", "strided", "contiguous float32 tensors"),
])
def test_compute_profile_rejects_invalid_shape_specific_signature(tmp_path, field, value, message):
    raw = _profile()
    raw["devices"][0]["records"] = [{
        "stage": "producer", "signature": {
            "op": "fill", "output_role": "collective_input", "shape": [2, 3],
            "dtype": "float32", "layout": "contiguous"},
        "device_event_ms_samples": [0.2], "host_enqueue_us_samples": [3.0],
        "preparation_us": 4.0,
    }]
    raw["devices"][0]["records"][0]["signature"][field] = value
    path = tmp_path / "invalid-signature.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match=message):
        load_gpu_compute_profile(path)


def test_dag_profile_v3_signature_binds_all_input_output_shapes_and_dtype(tmp_path):
    buffers = {
        "a": {"shape": [2, 3], "dtype": "float32"},
        "b": {"shape": [3, 4], "dtype": "float32"},
        "out": {"shape": [2, 4], "dtype": "float32"},
    }
    program = {"op": "matmul", "inputs": ["a", "b"], "output": "out", "repeats": 3}
    signature = dag_compute_profile_signature(program, buffers)
    raw = {
        "schema": "jobpacer-gpu-compute-profile", "schema_version": 3,
        "software": {"pytorch_version": "2.x", "cuda_version": "12.x",
                     "matmul_precision": "highest", "allow_tf32": False},
        "devices": [{"device_uuid": "GPU-0", "records": [{
            "stage": "compute", "signature": signature,
            "device_event_ms_samples": [1.0, 2.0], "host_enqueue_us_samples": [3.0, 4.0],
            "preparation_us": 5.0,
        }]}],
    }
    path = tmp_path / "dag-compute-v3.json"
    path.write_text(json.dumps(raw))
    profile = load_gpu_compute_profile(path)
    record = profile.record(device_uuid="GPU-0", stage="compute", spec=signature,
                            software=raw["software"])
    assert record.device_event_p50_s == pytest.approx(0.0015)
    wrong = dict(signature, output_shape=[2, 5])
    with pytest.raises(ValueError, match="signature missing"):
        profile.record(device_uuid="GPU-0", stage="compute", spec=wrong,
                       software=raw["software"])
