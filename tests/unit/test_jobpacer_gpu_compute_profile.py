from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from examples.jobpacer.gpu.gpu_compute_profile import (
    dag_compute_profile_signature,
    load_gpu_compute_profile,
)
from examples.jobpacer.scripts import run_gpu_compute_profile


SOFTWARE = {"pytorch_version": "2.x", "cuda_version": "12.x",
            "matmul_precision": "highest", "allow_tf32": False}


def _profile():
    return {
        "schema": "jobpacer-gpu-compute-profile", "schema_version": 3,
        "software": dict(SOFTWARE),
        "devices": [{"device_uuid": "GPU-0", "records": [{
            "stage": "compute",
            "signature": {"op": "matmul", "input_shapes": [[2, 3], [3, 2]],
                          "output_shape": [2, 2], "dtype": "float32",
                          "layout": "contiguous", "repeats": 2},
            "device_event_ms_samples": [1.0, 3.0, 2.0],
            "host_enqueue_us_samples": [4.0, 5.0, 6.0], "preparation_us": 10.0,
        }]}],
    }


def test_profile_matches_device_software_and_full_schema_v2_program_signature(tmp_path):
    path = tmp_path / "compute.json"
    path.write_text(json.dumps(_profile()))
    profile = load_gpu_compute_profile(path)
    signature = _profile()["devices"][0]["records"][0]["signature"]
    record = profile.record(device_uuid="GPU-0", stage="compute", spec=signature,
                            software=SOFTWARE)
    assert record.device_event_p50_s == pytest.approx(0.002)
    assert len(profile.digest) == 64
    with pytest.raises(ValueError, match="UUID"):
        profile.record(device_uuid="GPU-1", stage="compute", spec=signature)
    with pytest.raises(ValueError, match="software mismatch"):
        profile.record(device_uuid="GPU-0", stage="compute", spec=signature,
                       software={**SOFTWARE, "pytorch_version": "wrong"})


def test_profile_accepts_only_schema_v3_dag_records(tmp_path):
    raw = _profile()
    raw["schema_version"] = 2
    path = tmp_path / "legacy-v2.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="unsupported GPU compute profile schema/version"):
        load_gpu_compute_profile(path)

    raw = _profile()
    raw["devices"][0]["records"][0]["stage"] = "producer"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="stage must be 'compute'"):
        load_gpu_compute_profile(path)


def test_profile_rejects_duplicate_signatures_and_invalid_samples(tmp_path):
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


@pytest.mark.parametrize(("signature", "message"), [
    ({"op": "fill", "input_shapes": [], "output_shape": [2, True],
      "dtype": "float32", "layout": "contiguous"}, "positive dimensions"),
    ({"op": "matmul", "input_shapes": [[2, 3], [4, 2]], "output_shape": [2, 2],
      "dtype": "float32", "layout": "contiguous", "repeats": 2}, "shapes are incompatible"),
    ({"op": "matmul", "m": 2, "n": 2, "k": 3, "dtype": "float32",
      "layout": "contiguous", "repeats": 2}, "fields are invalid"),
])
def test_profile_rejects_legacy_or_invalid_program_signatures(tmp_path, signature, message):
    raw = _profile()
    raw["devices"][0]["records"][0]["signature"] = signature
    path = tmp_path / "invalid-signature.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match=message):
        load_gpu_compute_profile(path)


def test_fill_and_join_signatures_bind_buffer_shapes(tmp_path):
    raw = _profile()
    raw["devices"][0]["records"] = [
        {"stage": "compute", "signature": {
            "op": "fill", "input_shapes": [], "output_shape": [2, 3],
            "dtype": "float32", "layout": "contiguous"},
         "device_event_ms_samples": [0.2], "host_enqueue_us_samples": [3.0],
         "preparation_us": 4.0},
        {"stage": "compute", "signature": {
            "op": "sum_join", "input_shapes": [[2, 3], [2, 3]], "output_shape": [],
            "dtype": "float32", "layout": "contiguous"},
         "device_event_ms_samples": [0.4], "host_enqueue_us_samples": [5.0],
         "preparation_us": 6.0},
    ]
    path = tmp_path / "shape-specific.json"
    path.write_text(json.dumps(raw))
    profile = load_gpu_compute_profile(path)
    fill = raw["devices"][0]["records"][0]["signature"]
    join = raw["devices"][0]["records"][1]["signature"]
    assert profile.record(device_uuid="GPU-0", stage="compute", spec=fill).signature == fill
    assert profile.record(device_uuid="GPU-0", stage="compute", spec=join).signature == join


def test_dag_profile_signature_uses_program_repeats_and_bound_buffers():
    buffers = {
        "a": {"shape": [2, 3], "dtype": "float32"},
        "b": {"shape": [3, 4], "dtype": "float32"},
        "out": {"shape": [2, 4], "dtype": "float32"},
    }
    program = {"op": "matmul", "inputs": ["a", "b"], "output": "out", "repeats": 3}
    assert dag_compute_profile_signature(program, buffers) == {
        "op": "matmul", "input_shapes": [[2, 3], [3, 4]], "output_shape": [2, 4],
        "dtype": "float32", "layout": "contiguous", "repeats": 3,
    }


def test_compute_profiler_rejects_schema1_before_cuda_probe(monkeypatch, tmp_path, capsys):
    root = Path(__file__).parents[2]
    monkeypatch.setattr(sys, "argv", [
        "run_gpu_compute_profile", "--dag",
        str(root / "benchmark/phase3/experiments/dag-semantics/smoke/linear.json"),
        "--output", str(tmp_path / "profile.json"),
    ])
    with pytest.raises(SystemExit) as error:
        run_gpu_compute_profile.main()
    assert error.value.code == 2
    assert "schema-v2 cuda-program" in capsys.readouterr().err
