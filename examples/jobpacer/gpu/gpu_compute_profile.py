"""Versioned, signature-checked CUDA compute calibration data."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from examples.jobpacer.gpu.gpu_workload import GpuComputeSpec


PROFILE_SCHEMA = "jobpacer-gpu-compute-profile"
PROFILE_VERSION = 2
DAG_PROFILE_VERSION = 3


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
               spec: GpuComputeSpec | Mapping[str, Any],
               tensor_shape: tuple[int, ...] | None = None,
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
        target = compute_profile_signature(spec, tensor_shape=tensor_shape)
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
            or raw["schema_version"] not in {PROFILE_VERSION, DAG_PROFILE_VERSION}):
        raise ValueError("unsupported GPU compute profile schema/version")
    version = raw["schema_version"]
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
        records = tuple(_record(record, f"devices[{index}].records[{record_index}]", version=version)
                        for record_index, record in enumerate(row["records"]))
        signatures = [(record.stage, json.dumps(record.signature, sort_keys=True)) for record in records]
        if len(signatures) != len(set(signatures)):
            raise ValueError(f"devices[{index}] contains duplicate compute signatures")
        devices[uuid] = records
    return GpuComputeProfile(dict(software), devices, raw)


def _record(raw: Any, where: str, *, version: int = PROFILE_VERSION) -> ComputeProfileRecord:
    value = _object(raw, where, {"stage", "signature", "device_event_ms_samples",
                                 "host_enqueue_us_samples", "preparation_us"})
    supported_stages = ({"compute"} if version == DAG_PROFILE_VERSION
                        else {"producer", "independent", "dependent"})
    if value["stage"] not in supported_stages:
        raise ValueError(f"{where}.stage is unsupported")
    signature = value["signature"]
    if not isinstance(signature, dict) or signature.get("op") not in {"fill", "matmul", "sum_join"}:
        raise ValueError(f"{where}.signature is invalid")
    op = signature.get("op")
    if version == DAG_PROFILE_VERSION:
        _validate_dag_signature(signature, where)
        device_samples = _samples(value["device_event_ms_samples"], f"{where}.device_event_ms_samples")
        host_samples = _samples(value["host_enqueue_us_samples"], f"{where}.host_enqueue_us_samples")
        if len(device_samples) != len(host_samples):
            raise ValueError(f"{where} device and host sample counts differ")
        prep = _finite_nonnegative(value["preparation_us"], f"{where}.preparation_us")
        return ComputeProfileRecord(value["stage"], dict(signature), device_samples, host_samples, prep)
    if ((value["stage"] == "dependent") != (op == "sum_join")
            or (value["stage"] == "independent" and op != "matmul")):
        raise ValueError(f"{where}.signature op does not match its stage")
    if op == "matmul":
        expected_fields = {"op", "m", "n", "k", "dtype", "layout", "repeats", "output_role"}
        if value["stage"] == "independent":
            expected_fields.add("input_role")
    elif op == "fill":
        expected_fields = {"op", "output_role", "shape", "dtype", "layout"}
    else:
        expected_fields = {"op", "inputs", "shape", "dtype", "layout"}
    if set(signature) != expected_fields:
        raise ValueError(f"{where}.signature fields are invalid for its stage")
    if op == "matmul":
        dims = (signature.get("m"), signature.get("n"), signature.get("k"), signature.get("repeats"))
        if any(not isinstance(item, int) or isinstance(item, bool) or item <= 0 for item in dims):
            raise ValueError(f"{where}.signature matmul dimensions/repeats must be positive integers")
        if signature.get("dtype") != "float32" or signature.get("layout") != "contiguous":
            raise ValueError(f"{where}.signature supports only contiguous float32 matmul")
        expected_roles = ((None, "independent_output") if value["stage"] == "independent"
                          else (None, "collective_input"))
        if signature.get("output_role") != expected_roles[1]:
            raise ValueError(f"{where}.signature output_role is invalid")
        if value["stage"] == "independent" and signature.get("input_role") != "private_inputs":
            raise ValueError(f"{where}.signature input_role is invalid")
    elif op == "fill" and signature.get("output_role") != "collective_input":
        raise ValueError(f"{where}.signature output_role is invalid")
    elif op == "sum_join":
        inputs = signature.get("inputs")
        if (not isinstance(inputs, list) or "collective_output" not in inputs
                or len(inputs) != len(set(inputs))
                or any(item not in {"collective_output", "independent_output"} for item in inputs)):
            raise ValueError(f"{where}.signature inputs are invalid")
    if op in {"fill", "sum_join"}:
        shape = signature.get("shape")
        if (not isinstance(shape, list) or not shape
                or any(not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0 for dim in shape)):
            raise ValueError(f"{where}.signature shape must contain positive integers")
        if signature.get("dtype") != "float32" or signature.get("layout") != "contiguous":
            raise ValueError(f"{where}.signature supports only contiguous float32 tensors")
    device_samples = _samples(value["device_event_ms_samples"], f"{where}.device_event_ms_samples")
    host_samples = _samples(value["host_enqueue_us_samples"], f"{where}.host_enqueue_us_samples")
    if len(device_samples) != len(host_samples):
        raise ValueError(f"{where} device and host sample counts differ")
    prep = _finite_nonnegative(value["preparation_us"], f"{where}.preparation_us")
    return ComputeProfileRecord(value["stage"], dict(signature), device_samples, host_samples, prep)


def compute_profile_signature(spec: GpuComputeSpec | Mapping[str, Any], *,
                              tensor_shape: tuple[int, ...] | None = None) -> dict[str, Any]:
    signature = spec.to_dict() if isinstance(spec, GpuComputeSpec) else dict(spec)
    if "input_shapes" in signature:
        return signature
    if signature.get("op") in {"fill", "sum_join"}:
        if (tensor_shape is None or not tensor_shape
                or any(not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0
                       for dim in tensor_shape)):
            raise ValueError("fill and sum_join profile signatures require a positive tensor_shape")
        signature.update({"shape": list(tensor_shape), "dtype": "float32", "layout": "contiguous"})
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
