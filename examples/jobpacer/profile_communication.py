"""Measure uncontended collective service times for a JobPacer workload."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import socket
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

try:
    from .comm_profile import CommSignature, CommunicationProfile, ProfileRecord
    from .workloads import load_workload, ranks_for_job
except ImportError:  # pragma: no cover - direct script execution
    from comm_profile import CommSignature, CommunicationProfile, ProfileRecord
    from workloads import load_workload, ranks_for_job


HERE = Path(__file__).resolve().parent


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _record(signature: CommSignature, samples: list[float]) -> ProfileRecord:
    return ProfileRecord(
        **asdict(signature),
        p50_s=_percentile(samples, 0.5),
        p10_s=_percentile(samples, 0.1),
        p90_s=_percentile(samples, 0.9),
        mean_s=statistics.fmean(samples),
        stdev_s=statistics.pstdev(samples),
        samples=len(samples),
    )


def _measure(signature: CommSignature, group, device: str, warmup: int, iterations: int) -> list[float]:
    samples: list[float] = []
    for index in range(warmup + iterations):
        tensor = torch.ones(signature.num_bytes // 4, dtype=torch.float32, device=device)
        dist.barrier(group=group)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter_ns()
        work = dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group, async_op=True)
        work.wait()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = (time.perf_counter_ns() - start) / 1e9
        maximum = torch.tensor(elapsed, dtype=torch.float64, device=device)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=group)
        if index >= warmup:
            samples.append(float(maximum.item()))
    return samples


def _profile_rank(args: argparse.Namespace) -> dict[str, Any]:
    rank = args.rank
    workload = load_workload(args.workload)
    if args.backend == "nccl":
        torch.cuda.set_device(0)
        device = "cuda:0"
        device_type = "cuda"
    else:
        device = "cpu"
        device_type = "cpu"
    dist.init_process_group(args.backend)
    rank_sets: list[tuple[int, ...]] = []
    for job in workload.jobs:
        ranks = ranks_for_job(job, args.world_size)
        if ranks not in rank_sets:
            rank_sets.append(ranks)
    groups = {ranks: dist.new_group(ranks=list(ranks)) for ranks in rank_sets}

    measurements: list[tuple[CommSignature, tuple[int, ...]]] = []
    signature_ranks: dict[CommSignature, tuple[int, ...]] = {}
    for job in workload.jobs:
        ranks = ranks_for_job(job, args.world_size)
        for communication in job.communications:
            signature = CommSignature.for_communication(
                communication,
                group_size=len(ranks),
                backend=args.backend,
                device_type=device_type,
            )
            previous = signature_ranks.setdefault(signature, ranks)
            if previous != ranks:
                raise ValueError(
                    f"signature {signature} occurs on different rank sets {previous} and {ranks}"
                )
            if (signature, ranks) not in measurements:
                measurements.append((signature, ranks))

    records: list[ProfileRecord] = []
    try:
        for signature, ranks in measurements:
            local_record = None
            if rank in ranks:
                samples = _measure(
                    signature, groups[ranks], device, args.warmup, args.iterations
                )
                if rank == ranks[0]:
                    local_record = asdict(_record(signature, samples))
            gathered: list[dict[str, Any] | None] = [None] * args.world_size
            dist.all_gather_object(gathered, local_record)
            document = next((item for item in gathered if item is not None), None)
            if document is None:
                raise RuntimeError(f"no profile record produced for {signature}")
            records.append(ProfileRecord(**document))

        local_device_name = torch.cuda.get_device_name() if device_type == "cuda" else None
        device_names: list[str | None] = [None] * args.world_size
        dist.all_gather_object(device_names, local_device_name)
        environment = {
            "backend": args.backend,
            "device_type": device_type,
            "world_size": args.world_size,
            "group_ranks": [list(ranks) for ranks in rank_sets],
            "hostname": socket.gethostname(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "nccl_version": torch.cuda.nccl.version() if device_type == "cuda" else None,
            "device_names": device_names,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        profile = CommunicationProfile(
            schema_version=1,
            environment=environment,
            settings={"warmup": args.warmup, "iterations": args.iterations},
            records=tuple(records),
        )
        if rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.output.with_suffix(args.output.suffix + ".tmp")
            temporary.write_text(json.dumps(profile.to_dict(), indent=2, sort_keys=True) + "\n")
            temporary.replace(args.output)
        return {"rank": rank, "status": "ok", "profile_digest": profile.digest()}
    finally:
        dist.destroy_process_group()


def _spawn(rank: int, args: argparse.Namespace, port: int) -> subprocess.Popen[str]:
    env = dict(os.environ)
    env.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(args.world_size))
    env["PYTHONPATH"] = str(HERE.parents[1] / "src") + os.pathsep + env.get("PYTHONPATH", "")
    if args.backend == "nccl":
        env["CUDA_VISIBLE_DEVICES"] = str(rank)
    command = [
        sys.executable, str(Path(__file__).resolve()), "--workload", args.workload,
        "--backend", args.backend, "--world-size", str(args.world_size),
        "--warmup", str(args.warmup), "--iterations", str(args.iterations),
        "--timeout", str(args.timeout), "--output", str(args.output), "--rank", str(rank),
    ]
    return subprocess.Popen(command, env=env, cwd=HERE.parents[1], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _collect(process: subprocess.Popen[str], timeout: float) -> tuple[bool, str]:
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        return False, f"timed out: {stderr[-2000:]}"
    return process.returncode == 0, stdout.strip().splitlines()[-1] if process.returncode == 0 else stderr[-3000:]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", default="balanced")
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if Path(args.workload).exists():
        args.workload = str(Path(args.workload).resolve())
    args.output = args.output.resolve()
    if args.world_size < 2 or args.warmup < 0 or args.iterations <= 0:
        parser.error("world-size >= 2, warmup >= 0, and iterations > 0 are required")
    if args.rank is not None:
        try:
            result = _profile_rank(args)
            status = 0
        except BaseException as exc:  # noqa: BLE001 - subprocess JSON contract
            result = {"rank": args.rank, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
            status = 1
        print(json.dumps(result, sort_keys=True), flush=True)
        return status

    port = _free_port()
    processes = [_spawn(rank, args, port) for rank in range(args.world_size)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.world_size) as pool:
        results = list(pool.map(lambda process: _collect(process, args.timeout), processes))
    failures = [message for ok, message in results if not ok]
    if failures:
        print(json.dumps({"status": "failed", "errors": failures}, indent=2))
        return 1
    print(json.dumps({"status": "ok", "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
