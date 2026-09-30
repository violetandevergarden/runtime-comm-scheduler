"""Calibrate fixed GPU workload stages and write a versioned compute profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from examples.jobpacer.gpu.gpu_compute_profile import (
    DAG_PROFILE_VERSION,
    PROFILE_SCHEMA,
    dag_compute_profile_signature,
)
from examples.jobpacer.runtime.runtime_adapter import load_dag, require_gpu_dag_contract


def collect_calibration_cases(dag_paths: list[Path], *, world_size: int):
    """Load suite inputs and deduplicate exact compute signatures before calibration."""
    workloads = []
    cases: dict[str, dict[str, object]] = {}
    for dag_path in dag_paths:
        workload = load_dag(dag_path, world_size=world_size)
        require_gpu_dag_contract(workload)
        workloads.append((dag_path.resolve(), workload))
        for job in workload.graph.jobs:
            for node in job.nodes:
                if node.__class__.__name__ != "ComputeNode":
                    continue
                key_id = f"{job.job_id}/{node.node_id}"
                program = workload.execution.compute_programs[key_id]
                buffers = workload.execution.buffers[job.job_id]
                signature = dag_compute_profile_signature(program, buffers)
                key = json.dumps(signature, sort_keys=True)
                cases.setdefault(key, {"signature": signature, "program": program,
                                       "buffers": buffers, "first_node": key_id})
    if not workloads or not cases:
        raise ValueError("compute profile suite must contain at least one GPU compute node")
    return workloads, cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dag", type=Path, action="append", default=[],
                        help="schema-v2 CUDA DAG calibration input; repeat for a frozen sample suite")
    parser.add_argument("--suite-input-dir", type=Path, action="append", default=[],
                        help="recursively include generated workload-*.json samples under this directory")
    parser.add_argument("--output", type=Path, required=True)
    devices = parser.add_mutually_exclusive_group()
    devices.add_argument("--device", help="calibrate one explicit cuda:N device")
    devices.add_argument("--devices", help="comma-separated logical CUDA ordinals; defaults to all visible GPUs")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--matmul-precision", choices=("highest", "high", "medium"), default="highest")
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("warmup must be non-negative and iterations positive")
    dag_paths = list(args.dag)
    for directory in args.suite_input_dir:
        if not directory.is_dir():
            parser.error(f"suite input directory does not exist: {directory}")
        dag_paths.extend(path for path in sorted(directory.rglob("workload-*.json"))
                         if ".fifo-order." not in path.name and ".ltf-order." not in path.name)
    dag_paths = list(dict.fromkeys(path.resolve() for path in dag_paths))
    if not dag_paths:
        parser.error("provide at least one --dag or --suite-input-dir")
    try:
        for dag_path in dag_paths:
            require_gpu_dag_contract(load_dag(dag_path))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    torch.set_float32_matmul_precision(args.matmul_precision)
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.device:
        device_list = [torch.device(args.device)]
    elif args.devices:
        try:
            device_list = [torch.device(f"cuda:{int(item)}") for item in args.devices.split(",")]
        except ValueError:
            parser.error("--devices must be a comma-separated list of logical GPU ordinals")
    else:
        device_list = [torch.device(f"cuda:{index}") for index in range(torch.cuda.device_count())]
    if (not device_list or any(device.type != "cuda" or device.index is None
                               or device.index >= torch.cuda.device_count() for device in device_list)
            or len({device.index for device in device_list}) != len(device_list)):
        parser.error("each requested device must be a unique, explicitly visible cuda:N device")
    try:
        workloads, calibration_cases = collect_calibration_cases(
            dag_paths, world_size=torch.cuda.device_count())
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    device_profiles = []
    for device in device_list:
        rank = device.index
        records: dict[tuple[str, str], dict[str, object]] = {}
        for key_json, case in calibration_cases.items():
            program = case["program"]
            buffers = case["buffers"]
            signature = case["signature"]
            key = ("compute", key_json)
            prepare_start = time.perf_counter_ns()
            inputs = [torch.zeros(tuple(buffers[name]["shape"]), dtype=torch.float32, device=device)
                      for name in program["inputs"]]
            output = torch.empty(tuple(buffers[program["output"]]["shape"]),
                                 dtype=torch.float32, device=device)
            torch.cuda.synchronize(device)
            preparation_us = (time.perf_counter_ns() - prepare_start) / 1000.0

            def launch() -> None:
                if program["op"] == "fill":
                    output.fill_(0.25)
                elif program["op"] == "matmul":
                    for _ in range(program["repeats"]):
                        torch.mm(inputs[0], inputs[1], out=output)
                else:
                    output.zero_()
                    for tensor in inputs:
                        output.add_(torch.sum(tensor))

            for _ in range(args.warmup):
                launch()
            torch.cuda.synchronize(device)
            row = {"stage": "compute", "signature": signature,
                   "device_event_ms_samples": [], "host_enqueue_us_samples": [],
                   "preparation_us": preparation_us}
            for _ in range(args.iterations):
                start, done = (torch.cuda.Event(enable_timing=True, blocking=False),
                               torch.cuda.Event(enable_timing=True, blocking=False))
                before = time.perf_counter_ns()
                start.record()
                launch()
                done.record()
                row["host_enqueue_us_samples"].append((time.perf_counter_ns() - before) / 1000.0)
                done.synchronize()
                row["device_event_ms_samples"].append(start.elapsed_time(done))
            records[key] = row
        torch.cuda.synchronize(device)
        properties = torch.cuda.get_device_properties(device)
        uuid = str(getattr(properties, "uuid", ""))
        if not uuid:
            parser.error(f"PyTorch did not report the UUID for {device}")
        device_profiles.append({"device_uuid": uuid, "records": list(records.values())})
    version = DAG_PROFILE_VERSION
    payload = {"schema": PROFILE_SCHEMA, "schema_version": version,
               "software": {"pytorch_version": torch.__version__, "cuda_version": torch.version.cuda,
                            "matmul_precision": args.matmul_precision, "allow_tf32": False},
               "devices": device_profiles}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    args.output.write_text(serialized)
    profile_digest = hashlib.sha256(serialized.encode()).hexdigest()
    provenance = {
        "schema": "jobpacer-gpu-compute-profile-calibration",
        "schema_version": 1,
        "profile_sha256": profile_digest,
        "inputs": [{"path": str(path),
                    "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "input_hash": workload.input_hash,
                    "estimate_view_hash": workload.estimate_view_hash}
                   for path, workload in workloads],
        "settings": {"warmup": args.warmup, "iterations": args.iterations,
                     "matmul_precision": args.matmul_precision,
                     "allow_tf32": False, "timing": "CUDA events plus host enqueue"},
        "device_uuids": [row["device_uuid"] for row in device_profiles],
        "record_count": sum(len(row["records"]) for row in device_profiles),
    }
    Path(str(args.output) + ".manifest.json").write_text(
        json.dumps(provenance, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "device_uuids": [row["device_uuid"] for row in device_profiles],
                      "record_count": sum(len(row["records"]) for row in device_profiles),
                      "input_count": len(workloads), "profile_sha256": profile_digest},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
