"""Versioned, signature-checked CUDA compute calibration data."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

PROFILE_SCHEMA = "jobpacer-gpu-compute-profile"
DAG_PROFILE_VERSION = 3
PROFILE_VERSION = DAG_PROFILE_VERSION


@dataclass(frozen=True)
class ComputeProfileRecord:
    stage: str
    signature: Mapping[str, Any]
    device_event_ms_samples: tuple[float, ...]
    host_enqueue_us_samples: tuple[float, ...]
    preparation_us: float

    @property
    def device_event_p50_s(self) -> float:
        return statistics.median(self.device_event_ms_samples) / 1000.0

    def to_dict(self) -> dict[str, Any]:
        return {"stage": self.stage, "signature": dict(self.signature),
                "device_event_ms_samples": list(self.device_event_ms_samples),
                "host_enqueue_us_samples": list(self.host_enqueue_us_samples),
                "preparation_us": self.preparation_us}


@dataclass(frozen=True)
class GpuComputeProfile:
    software: Mapping[str, Any]
    devices: Mapping[str, tuple[ComputeProfileRecord, ...]]
    raw: Mapping[str, Any]

    @property
    def digest(self) -> str:
        payload = json.dumps(self.raw, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode()).hexdigest()

    def record(self, *, device_uuid: str, stage: str,
               spec: Mapping[str, Any],
               software: Mapping[str, Any] | None = None) -> ComputeProfileRecord:
        mismatches = []
        for key, expected in self.software.items():
            if software is not None and software.get(key) != expected:
                mismatches.append(f"{key}: profile={expected!r}, runtime={software.get(key)!r}")
        if mismatches:
            raise ValueError("GPU compute profile software mismatch: " + "; ".join(mismatches))
        records = self.devices.get(device_uuid)
        if records is None:
            raise ValueError(f"GPU compute profile has no records for device UUID {device_uuid!r}")
        target = compute_profile_signature(spec)
        found = [row for row in records if row.stage == stage and dict(row.signature) == target]
        if len(found) != 1:
            raise ValueError(f"GPU compute profile signature missing or duplicated: {stage} {target}")
        return found[0]


def load_gpu_compute_profile(path: str | Path) -> GpuComputeProfile:
    source = Path(path)
    try:
        raw = json.loads(source.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load GPU compute profile {source}: {exc}") from exc
    if not isinstance(raw, dict) or set(raw) != {"schema", "schema_version", "software", "devices"}:
        raise ValueError("GPU compute profile root fields must be schema, schema_version, software, devices")
    if (raw["schema"] != PROFILE_SCHEMA or isinstance(raw["schema_version"], bool)
            or not isinstance(raw["schema_version"], int)
            or raw["schema_version"] != PROFILE_VERSION):
        raise ValueError("unsupported GPU compute profile schema/version")
    software = _object(raw["software"], "software", {"pytorch_version", "cuda_version",
                                                        "matmul_precision", "allow_tf32"})
    if software["matmul_precision"] not in {"highest", "high", "medium"} or software["allow_tf32"] is not False:
        raise ValueError("profile must record a supported precision and allow_tf32=false")
    if any(not isinstance(software[key], str) or not software[key]
           for key in ("pytorch_version", "cuda_version")):
        raise ValueError("profile must record non-empty PyTorch and CUDA versions")
    devices_raw = raw["devices"]
    if not isinstance(devices_raw, list) or not devices_raw:
        raise ValueError("devices must be a non-empty array")
    devices: dict[str, tuple[ComputeProfileRecord, ...]] = {}
    for index, item in enumerate(devices_raw):
        row = _object(item, f"devices[{index}]", {"device_uuid", "records"})
        uuid = row["device_uuid"]
        if not isinstance(uuid, str) or not uuid.strip() or uuid in devices:
            raise ValueError(f"devices[{index}].device_uuid must be unique and non-empty")
        if not isinstance(row["records"], list) or not row["records"]:
            raise ValueError(f"devices[{index}].records must be non-empty")
        records = tuple(_record(record, f"devices[{index}].records[{record_index}]")
                        for record_index, record in enumerate(row["records"]))
        signatures = [(record.stage, json.dumps(record.signature, sort_keys=True)) for record in records]
        if len(signatures) != len(set(signatures)):
            raise ValueError(f"devices[{index}] contains duplicate compute signatures")
        devices[uuid] = records
    return GpuComputeProfile(dict(software), devices, raw)


def _record(raw: Any, where: str) -> ComputeProfileRecord:
    value = _object(raw, where, {"stage", "signature", "device_event_ms_samples",
                                 "host_enqueue_us_samples", "preparation_us"})
    if value["stage"] != "compute":
        raise ValueError(f"{where}.stage must be 'compute' for a schema-v2 DAG program")
    signature = value["signature"]
    if not isinstance(signature, dict):
        raise ValueError(f"{where}.signature is invalid")
    _validate_dag_signature(signature, where)
    device_samples = _samples(value["device_event_ms_samples"], f"{where}.device_event_ms_samples")
    host_samples = _samples(value["host_enqueue_us_samples"], f"{where}.host_enqueue_us_samples")
    if len(device_samples) != len(host_samples):
        raise ValueError(f"{where} device and host sample counts differ")
    prep = _finite_nonnegative(value["preparation_us"], f"{where}.preparation_us")
    return ComputeProfileRecord(value["stage"], dict(signature), device_samples, host_samples, prep)


def compute_profile_signature(spec: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(spec, Mapping):
        raise TypeError("compute profile signature must be a mapping")
    signature = dict(spec)
    _validate_dag_signature(signature, "compute profile signature")
    return signature


def dag_compute_profile_signature(program: Mapping[str, Any],
                                  buffers: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    op = program["op"]
    input_shapes = [list(buffers[key]["shape"]) for key in program["inputs"]]
    output_shape = list(buffers[program["output"]]["shape"])
    signature: dict[str, Any] = {
        "op": op, "input_shapes": input_shapes, "output_shape": output_shape,
        "dtype": buffers[program["output"]]["dtype"], "layout": "contiguous",
    }
    if op == "matmul":
        signature["repeats"] = program["repeats"]
    return signature


def _validate_dag_signature(signature: Mapping[str, Any], where: str) -> None:
    op = signature.get("op")
    expected = {"op", "input_shapes", "output_shape", "dtype", "layout"}
    if op == "matmul":
        expected.add("repeats")
    if set(signature) != expected or op not in {"fill", "matmul", "sum_join"}:
        raise ValueError(f"{where}.signature fields are invalid for a DAG node")
    input_shapes, output_shape = signature["input_shapes"], signature["output_shape"]
    if (not isinstance(input_shapes, list)
            or any(not isinstance(shape, list)
                   or any(not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0 for dim in shape)
                   for shape in input_shapes)
            or not isinstance(output_shape, list)
            or any(not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0 for dim in output_shape)):
        raise ValueError(f"{where}.signature shapes must contain positive dimensions (empty output means scalar)")
    if signature["dtype"] != "float32" or signature["layout"] != "contiguous":
        raise ValueError(f"{where}.signature supports only contiguous float32 buffers")
    if op == "matmul":
        if (len(input_shapes) != 2 or any(len(shape) != 2 for shape in input_shapes)
                or len(output_shape) != 2
                or input_shapes[0][1] != input_shapes[1][0]
                or output_shape != [input_shapes[0][0], input_shapes[1][1]]):
            raise ValueError(f"{where}.signature matmul shapes are incompatible")
        repeats = signature["repeats"]
        if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats <= 0:
            raise ValueError(f"{where}.signature repeats must be a positive integer")
    elif op == "fill" and input_shapes:
        raise ValueError(f"{where}.signature fill must not have inputs")
    elif op == "sum_join" and (not input_shapes or output_shape):
        raise ValueError(f"{where}.signature sum_join requires inputs and scalar output")


def _samples(raw: Any, where: str) -> tuple[float, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{where} must be a non-empty array")
    return tuple(_finite_nonnegative(item, where) for item in raw)


def _finite_nonnegative(raw: Any, where: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw) or raw < 0:
        raise ValueError(f"{where} values must be finite non-negative numbers")
    return float(raw)


def _object(raw: Any, where: str, allowed: set[str]) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != allowed:
        raise ValueError(f"{where} fields must be exactly {sorted(allowed)}")
    return raw
