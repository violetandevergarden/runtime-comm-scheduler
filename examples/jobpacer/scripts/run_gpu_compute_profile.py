"""Calibrate fixed GPU workload stages and write a versioned compute profile."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from examples.jobpacer.runtime.gpu_compute_profile import (
    DAG_PROFILE_VERSION,
    PROFILE_SCHEMA,
    dag_compute_profile_signature,
)
from examples.jobpacer.runtime.runtime_adapter import load_dag


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dag", type=Path, required=True,
                        help="schema-v2 CUDA DAG calibration input")
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
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
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
    workload = load_dag(args.dag, world_size=torch.cuda.device_count())
    if workload.execution.schema_version != 2:
        parser.error("--dag calibration requires a schema-v2 GPU DAG")
    device_profiles = []
    for device in device_list:
        rank = device.index
        records: dict[tuple[str, str], dict[str, object]] = {}
        for job in workload.graph.jobs:
            for node in job.nodes:
                if node.__class__.__name__ != "ComputeNode":
                    continue
                key_id = f"{job.job_id}/{node.node_id}"
                program = workload.execution.compute_programs[key_id]
                buffers = workload.execution.buffers[job.job_id]
                signature = dag_compute_profile_signature(program, buffers)
                key = ("compute", json.dumps(signature, sort_keys=True))
                if key in records:
                    continue
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
    args.output.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "device_uuids": [row["device_uuid"] for row in device_profiles],
                      "record_count": sum(len(row["records"]) for row in device_profiles),
                      "manifest_digest": workload.manifest_digest},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
