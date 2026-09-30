"""Run a small two-GPU NCCL mechanism check for the bare-ordered contract."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import statistics
import time
from datetime import timedelta
from pathlib import Path

# The bare contract relies on NCCL's documented host-order mechanism. Set it
# before any process group or communicator is initialized, regardless of the
# caller's default environment.
os.environ["NCCL_LAUNCH_ORDER_IMPLICIT"] = "1"

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from runtime_comm_scheduler.runtime import CudaCollectiveExecutor, LocalBinding


MODES = ("a-alone", "b-alone", "completion-serial", "multi-inflight")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _binding(tensor: torch.Tensor, group, device: torch.device) -> LocalBinding:
    producer_event = torch.cuda.Event(blocking=False, interprocess=False)
    producer_event.record(torch.cuda.current_stream(device))
    return LocalBinding(
        tensor=tensor,
        process_group=group,
        launch=lambda: dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group, async_op=True),
        producer_event=producer_event,
        device=device,
        keepalive=(tensor, producer_event),
    )


def _wait_receipt(receipt, *, deadline: float) -> int:
    while not receipt.is_completed():
        if time.monotonic() >= deadline:
            raise TimeoutError("timed out observing bare NCCL device completion")
        time.sleep(0.0002)
    return time.perf_counter_ns()


def _run_case(*, mode: str, repeat: int, rank: int, numel: int,
              groups: dict[str, object], executor: CudaCollectiveExecutor,
              device: torch.device, timeout_s: float) -> dict[str, object]:
    tensors = {
        "A": torch.full((numel,), float(rank + 1), dtype=torch.float32, device=device),
        "B": torch.full((numel,), float(rank + 1), dtype=torch.float32, device=device),
    }
    bindings = {name: _binding(tensor, groups[name], device) for name, tensor in tensors.items()}
    started_ns = time.perf_counter_ns()
    launches: list[dict[str, object]] = []
    receipts = []
    completion_already_observed: set[str] = set()
    peak_inflight = 0

    def issue(name: str) -> None:
        nonlocal peak_inflight
        prior_pending = [key for key, prior in receipts if not prior.is_completed()]
        prior_probe_ns = time.perf_counter_ns()
        api_start_ns = time.perf_counter_ns()
        receipt = executor.launch(bindings[name])
        api_return_ns = time.perf_counter_ns()
        receipts.append((name, receipt))
        launches.append({
            "collective": name,
            "prior_pending_at_launch_probe": prior_pending,
            "prior_probe_ns": prior_probe_ns,
            "api_start_ns": api_start_ns,
            "api_return_ns": api_return_ns,
            "api_duration_us": (api_return_ns - api_start_ns) / 1000.0,
            "receipt_published_ns": api_return_ns,
        })
        peak_inflight = max(peak_inflight, sum(not item.is_completed() for _key, item in receipts))

    completion_deadline = time.monotonic() + timeout_s
    if mode in {"a-alone", "completion-serial", "multi-inflight"}:
        issue("A")
        if mode == "completion-serial":
            completion_ns = _wait_receipt(receipts[-1][1], deadline=completion_deadline)
            launches[-1]["physical_complete_observed_ns"] = completion_ns
            completion_already_observed.add("A")
    if mode in {"b-alone", "completion-serial", "multi-inflight"}:
        issue("B")

    pending = {name: receipt for name, receipt in receipts
               if name not in completion_already_observed}
    while pending:
        for name, receipt in tuple(pending.items()):
            peak_inflight = max(peak_inflight, sum(
                not candidate.is_completed() for _key, candidate in receipts
            ))
            if receipt.is_completed():
                completed_ns = time.perf_counter_ns()
                next(row for row in launches if row["collective"] == name)[
                    "physical_complete_observed_ns"
                ] = completed_ns
                del pending[name]
        if pending:
            if time.monotonic() >= completion_deadline:
                raise TimeoutError(f"{mode} timed out with pending collectives {sorted(pending)}")
            time.sleep(0.0002)
    ended_ns = time.perf_counter_ns()

    expected = 3.0  # The check is fixed to the two-rank scope above.
    numeric_checks = {
        name: bool(torch.all(tensors[name] == expected).item())
        for name in ("A", "B") if any(item["collective"] == name for item in launches)
    }
    return {
        "mode": mode,
        "repeat": repeat,
        "rank": rank,
        "message_bytes_per_collective": numel * torch.tensor([], dtype=torch.float32).element_size(),
        "api_calls": launches,
        "observed_peak_inflight": peak_inflight,
        "application_total_us": (ended_ns - started_ns) / 1000.0,
        "numeric_checks": numeric_checks,
    }


def _worker(rank: int, world_size: int, port: int, numel: int,
            warmup: int, repeats: int, timeout_s: float, output_root: str) -> None:
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group(
        "nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank,
        world_size=world_size, timeout=timedelta(seconds=timeout_s), device_id=device,
    )
    groups = {
        "A": dist.new_group(ranks=list(range(world_size)), backend="nccl"),
        "B": dist.new_group(ranks=list(range(world_size)), backend="nccl"),
    }
    executor = CudaCollectiveExecutor(device)
    samples = []
    try:
        for _ in range(warmup):
            for name in ("A", "B"):
                tensor = torch.full((numel,), float(rank + 1), dtype=torch.float32, device=device)
                receipt = executor.launch(_binding(tensor, groups[name], device))
                _wait_receipt(receipt, deadline=time.monotonic() + timeout_s)
        for repeat in range(repeats):
            # Rotate mode order to avoid always giving the same mode the same
            # thermal/cache position. Every rank uses the identical sequence.
            shift = repeat % len(MODES)
            scheduled_modes = MODES[shift:] + MODES[:shift]
            for mode in scheduled_modes:
                dist.barrier()
                samples.append(_run_case(
                    mode=mode, repeat=repeat, rank=rank, numel=numel,
                    groups=groups, executor=executor, device=device, timeout_s=timeout_s,
                ))
                dist.barrier()
        result = {
            "rank": rank,
            "device": torch.cuda.get_device_name(device),
            "device_uuid": str(getattr(torch.cuda.get_device_properties(device), "uuid", "")) or None,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "nccl_version": list(torch.cuda.nccl.version()),
            "nccl_launch_order_implicit": os.environ["NCCL_LAUNCH_ORDER_IMPLICIT"],
            "samples": samples,
        }
        Path(output_root, f"rank-{rank}.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
    finally:
        for group in reversed(tuple(groups.values())):
            dist.destroy_process_group(group)
        dist.destroy_process_group()


def run_mechanism_check(output: Path, *, message_bytes: int = 1 << 20,
                        repeats: int = 5, warmup: int = 5,
                        timeout: float = 30.0) -> dict[str, object]:
    if message_bytes <= 0 or message_bytes % 4:
        raise ValueError("message_bytes must be a positive multiple of 4 for float32")
    if repeats < 5 or warmup < 0 or timeout <= 0:
        raise ValueError("mechanism check requires >=5 repeats, non-negative warmup, and a positive timeout")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("bare NCCL mechanism check requires two visible CUDA devices")
    from examples.jobpacer.runtime.dag_comm_adapters import validate_bare_backend
    validate_bare_backend(nccl_version=tuple(torch.cuda.nccl.version()), cuda_version=torch.version.cuda,
                          implicit=os.environ.get("NCCL_LAUNCH_ORDER_IMPLICIT"),
                          blocking_wait=os.environ.get("TORCH_NCCL_BLOCKING_WAIT"))
    if output.exists():
        raise FileExistsError("refusing to overwrite existing mechanism output directory")
    output.mkdir(parents=True)
    world_size = 2
    mp.spawn(
        _worker,
        args=(world_size, _free_port(), message_bytes // 4, warmup,
              repeats, timeout, str(output.resolve())),
        nprocs=world_size,
        join=True,
    )
    ranks = [json.loads((output / f"rank-{rank}.json").read_text())
             for rank in range(world_size)]
    by_mode: dict[str, list[float]] = {mode: [] for mode in MODES}
    for rank_result in ranks:
        for sample in rank_result["samples"]:
            by_mode[sample["mode"]].append(float(sample["application_total_us"]))
            if not all(sample["numeric_checks"].values()):
                raise RuntimeError(f"numeric check failed: rank={rank_result['rank']} sample={sample}")
    source_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    summary = {
        "contract": "bare-ordered-v2-layered-round-robin",
        "backend_mechanism": "NCCL_LAUNCH_ORDER_IMPLICIT=1",
        "message_bytes_per_collective": message_bytes,
        "repeats_per_mode_per_rank": repeats,
        "warmup_per_group_per_rank": warmup,
        "median_application_us_across_rank_samples": {
            mode: statistics.median(values) for mode, values in by_mode.items()
        },
        "source_sha256": source_digest,
        "ranks": ranks,
    }
    path = output / "mechanism-summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return {"status": "ok", "summary": str(path), "source_sha256": source_digest}
