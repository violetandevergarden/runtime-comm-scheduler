from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from datetime import timedelta

import torch
import torch.distributed as dist

from runtime_comm_scheduler.runtime import CudaCollectiveExecutor, LocalBinding


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2 or torch.cuda.device_count() < world_size:
        raise RuntimeError("this semantic worker requires two visible CUDA devices")
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", timeout=timedelta(seconds=30), device_id=device)
    try:
        tensor = torch.full((4096,), -11.0, device=device)
        producer = torch.cuda.Stream(device=device)
        producer_done = torch.cuda.Event()
        with torch.cuda.stream(producer):
            torch.cuda._sleep(20_000_000)
            tensor.fill_(float(rank + 1))
            producer_done.record(producer)

        executor = CudaCollectiveExecutor(device)
        binding = LocalBinding(
            tensor, dist.group.WORLD,
            lambda: dist.all_reduce(tensor, op=dist.ReduceOp.SUM,
                                    group=dist.group.WORLD, async_op=True),
            device=device, producer_event=producer_done, keepalive=(tensor, producer_done),
        )
        receipt = executor.launch(binding)
        unrelated = torch.cuda.Stream(device=device)
        unrelated_done = torch.cuda.Event()
        with torch.cuda.stream(unrelated):
            torch.cuda._sleep(500_000_000)
            unrelated_done.record(unrelated)
        consumer = torch.cuda.Stream(device=device)
        consumer_done = torch.cuda.Event()
        start = time.perf_counter()
        with torch.cuda.stream(consumer):
            dependency_returned = receipt.wait_on(consumer)
            observed = tensor.clone()
            consumer_done.record(consumer)
        dependency_return_s = time.perf_counter() - start
        unrelated_after_enqueue = unrelated_done.query()

        deadline = time.monotonic() + 20
        while not consumer_done.query() or not receipt.is_completed():
            if time.monotonic() >= deadline:
                raise TimeoutError("consumer or collective event did not complete")
            time.sleep(0.001)
        consumer.synchronize()
        correct = bool(torch.all(observed == 3.0).item())
        unrelated_after_consumer = bool(unrelated_done.query())
        item = {
            "rank": rank,
            "device_uuid": str(torch.cuda.get_device_properties(device).uuid),
            "allreduce_correct": correct,
            "receipt_completed": bool(receipt.is_completed()),
            "consumer_dependency_returned": bool(dependency_returned),
            "consumer_dependency_return_s": dependency_return_s,
            "unrelated_done_after_enqueue": bool(unrelated_after_enqueue),
            "unrelated_done_when_consumer_completed": unrelated_after_consumer,
            "completion_source": receipt.completion_source,
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / f"rank-{rank}.json").write_text(json.dumps(item, indent=2) + "\n")
        torch.cuda.synchronize(device)
        dist.barrier()
    finally:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
