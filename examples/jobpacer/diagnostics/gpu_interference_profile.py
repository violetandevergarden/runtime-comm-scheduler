"""Small compute/communication interference diagnostic; never a performance arm."""
from __future__ import annotations

import argparse
import json
import socket
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank: int, port: int, output: str) -> None:
    torch.cuda.set_device(rank)
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank,
                            world_size=2, device_id=device, timeout=timedelta(seconds=30))
    stream = torch.cuda.Stream(device=device)
    left = torch.full((256, 256), 0.125, device=device)
    right = torch.full_like(left, 0.125)
    result = torch.empty_like(left)
    rows = []
    try:
        for num_bytes in (262144, 67108864):
            tensor = torch.empty(num_bytes // 4, device=device)
            for repeats in (8, 96):
                for repeat in range(7):
                    modes = ("compute", "communication", "serial", "concurrent")
                    shift = repeat % 4
                    for mode in modes[shift:] + modes[:shift]:
                        tensor.fill_(rank + 1)
                        torch.cuda.synchronize(device)
                        dist.barrier()
                        begin = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start = time.perf_counter_ns()
                        if mode != "communication":
                            with torch.cuda.stream(stream):
                                begin.record()
                                for _ in range(repeats):
                                    torch.mm(left, right, out=result)
                                end.record()
                            if mode == "serial":
                                stream.synchronize()
                        if mode != "compute":
                            work = dist.all_reduce(tensor, async_op=True)
                            work.wait()
                        torch.cuda.synchronize(device)
                        elapsed_us = (time.perf_counter_ns() - start) / 1000
                        numeric = (mode == "compute" or bool(torch.all(tensor == 3).item()))
                        numeric = numeric and (mode == "communication" or bool(torch.all(result == 4).item()))
                        if repeat >= 2:
                            rows.append({"mode": mode, "repeat": repeat - 2, "bytes": num_bytes,
                                         "compute_repeats": repeats, "wall_us": elapsed_us,
                                         "compute_device_us": begin.elapsed_time(end) * 1000 if mode != "communication" else None,
                                         "numeric": numeric})
        Path(output, f"rank-{rank}.json").write_text(json.dumps({"rank": rank,
            "device_uuid": str(torch.cuda.get_device_properties(device).uuid), "samples": rows}, indent=2) + "\n")
    finally:
        dist.destroy_process_group()


def run_diagnostic(output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("exactly two visible CUDA devices are required")
    output.mkdir(parents=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(port, str(output.resolve())), nprocs=2, join=True)
    ranks = [json.loads((output / f"rank-{rank}.json").read_text()) for rank in range(2)]
    report = {"passed": all(row["numeric"] for rank in ranks for row in rank["samples"]),
              "ranks": ranks, "warmup_per_configuration": 2, "repeats": 5,
              "interpretation": "device duration includes resource interference; concurrent submission does not prove kernel overlap"}
    (output / "interference.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = run_diagnostic(args.output)
    except (FileExistsError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps({"passed": report["passed"], "output": str(args.output)}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
