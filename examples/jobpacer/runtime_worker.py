"""One-rank Stage 3.1 JobPacer replay."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from runtime_comm_scheduler.runtime import CollectiveSpec, DirectExecutor, EventLog, GroupSpec, LocalBinding, RankRuntime
from runtime_comm_scheduler.runtime.transport import ControlClient

try:
    from .runtime_adapter import group_spec, task_hint, task_spec
    from .workloads import Job, Workload, load_workload, ranks_for_job
except ImportError:  # pragma: no cover
    from runtime_adapter import group_spec, task_hint, task_spec
    from workloads import Job, Workload, load_workload, ranks_for_job


def _free_device(backend: str) -> str:
    if backend == "nccl":
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        return "cuda"
    return "cpu"


def _make_tensor(comm, rank: int, device: str) -> torch.Tensor:
    return torch.full((comm.num_bytes // 4,), float(rank + 1), dtype=torch.float32, device=device)


class _FailingProbe:
    supports_physical_completion = True

    def __init__(self):
        self.failed = False

    def is_completed(self, work):
        if not self.failed:
            self.failed = True
            raise RuntimeError("injected completion probe failure")
        return bool(work.is_completed())


def _groups(workload: Workload, world_size: int) -> dict[str, Any]:
    # All ranks create groups in the same manifest order.  A rank only uses
    # the group objects for jobs whose membership contains that rank.
    return {
        job.job_id: dist.new_group(list(ranks_for_job(job, world_size)))
        for job in workload.jobs
    }


def run_rank(args: argparse.Namespace) -> dict[str, Any]:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = _free_device(args.backend)
    dist.init_process_group(args.backend, init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}", rank=rank, world_size=world_size)
    workload = load_workload(args.workload)
    groups = _groups(workload, world_size)
    dist.barrier()

    server = None
    from runtime_comm_scheduler.runtime.coordinator import CoordinatorState
    from runtime_comm_scheduler.runtime.transport import CoordinatorServer
    static_order: tuple[str, ...] = ()
    runtime_policy = args.policy
    if args.policy in {"static_fifo", "static_ltf"}:
        try:
            from .plan_builder import build_plan
        except ImportError:  # pragma: no cover
            from plan_builder import build_plan
        source_policy = "fifo" if args.policy == "static_fifo" else "ltf"
        plan = build_plan(workload, source_policy)
        static_order = tuple(f"{key.process_group_id}/comm-{key.ordinal}" for key in plan.keys)
        runtime_policy = "static"
    if rank == 0:
        coordinator = CoordinatorState(tuple(range(world_size)), epoch=args.epoch, policy=runtime_policy, static_order=static_order, epoch_timeout_s=args.timeout)
        server = CoordinatorServer(coordinator, "127.0.0.1", args.control_port)
        server.start()

    client = ControlClient(rank, args.epoch, "127.0.0.1", args.control_port, args.timeout)
    completion_probe = _FailingProbe() if args.fault == "completion_probe_failure" and rank == 0 else None
    runtime = RankRuntime(
        rank,
        args.epoch,
        client,
        executor=DirectExecutor(),
        completion_poll_interval_s=args.poll_interval,
        event_log=EventLog("runtime", rank),
        completion_probe=completion_probe,
    )
    local_jobs = [job for job in workload.jobs if rank in ranks_for_job(job, world_size)]
    for job in local_jobs:
        runtime.register_group(group_spec(job, world_size, epoch=args.epoch), groups[job.job_id])
    runtime.start(args.timeout)
    if server is not None:
        server.wait_ready(args.timeout)

    result_by_job: dict[str, dict[str, Any]] = {}
    errors: list[BaseException] = []
    stop = threading.Event()

    def run_job(job: Job) -> None:
        job_start = time.perf_counter_ns() // 1000
        result = {"job_id": job.job_id, "status": "ok", "tasks": [], "job_start_ts": job_start}
        try:
            for index, comm in enumerate(job.communications):
                if args.fault == "missing_task" and rank == 1 and job.job_id == "job-1" and index == len(job.communications) - 1:
                    continue
                if stop.is_set():
                    raise RuntimeError("another job failed")
                spec = task_spec(job, comm, epoch=args.epoch)
                hint = task_hint(job, index)
                runtime.declare(spec, hint)
                producer_start = time.perf_counter_ns() // 1000
                time.sleep(comm.producer_compute_s)
                ready_ts = time.perf_counter_ns() // 1000
                tensor = _make_tensor(comm, rank, device)
                if args.fault == "metadata_mismatch" and rank == 1 and job.job_id == "job-1" and index == 0:
                    spec = replace(spec, collective=CollectiveSpec("all_reduce", spec.collective.numel + 1, spec.collective.num_bytes + 4, "float32", (spec.collective.numel + 1,)))
                group = groups[job.job_id]

                def launch(tensor=tensor, group=group):
                    if args.fault == "launch_failure" and rank == 0 and job.job_id == "job-0" and index == 0:
                        raise RuntimeError("injected launch failure")
                    return dist.all_reduce(tensor, group=group, async_op=True)

                submit_call_ts = time.perf_counter_ns() // 1000
                handle = runtime.submit(spec, LocalBinding(tensor, group, launch, device=device, keepalive=(tensor,)), hint)
                submit_return_ts = time.perf_counter_ns() // 1000
                consume_start = time.perf_counter_ns() // 1000
                time.sleep(comm.consumer_compute_s)
                first_wait_ts = time.perf_counter_ns() // 1000
                if not handle.wait_host(args.timeout):
                    raise TimeoutError(f"wait timed out for {spec.task_id}")
                consumer_end_ts = time.perf_counter_ns() // 1000
                expected = sum(item + 1 for item in ranks_for_job(job, world_size))
                correct = bool(torch.all(tensor == expected).item())
                if not correct:
                    raise AssertionError(f"incorrect result for {spec.task_id}")
                result["tasks"].append({
                    "task_id": spec.task_id,
                    "job_id": job.job_id,
                    "ordinal": comm.id,
                    "correct": correct,
                    "decision_seq": handle.decision_seq,
                    "group_id": spec.group_id,
                    "group_seq": spec.group_seq,
                    "producer_start_ts": producer_start,
                    "ready_ts": ready_ts,
                    "submit_call_ts": submit_call_ts,
                    "submit_return_ts": submit_return_ts,
                    "consumer_start_ts": consume_start,
                    "first_wait_ts": first_wait_ts,
                    "consumer_end_ts": consumer_end_ts,
                })
        except BaseException as exc:  # noqa: BLE001
            result["status"] = "error"
            result["error"] = f"{type(exc).__name__}: {exc}"
            errors.append(exc)
            stop.set()
            runtime._fail(exc, stage="job", job_id=job.job_id)
        result["job_end_ts"] = time.perf_counter_ns() // 1000
        result_by_job[job.job_id] = result

    threads = [threading.Thread(target=run_job, args=(job,), name=f"job-{job.job_id}") for job in local_jobs]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(args.timeout)
    try:
        if any(thread.is_alive() for thread in threads):
            raise TimeoutError("job thread did not finish")
        if errors:
            raise errors[0]
        runtime.finish_epoch(args.timeout)
        output = {
            "rank": rank,
            "status": "ok",
            "mode": "runtime",
            "policy": args.policy,
            "backend": args.backend,
            "jobs": [result_by_job[job.job_id] for job in local_jobs],
            "task_sequence": [task["task_id"] for job in result_by_job.values() for task in job["tasks"]],
            "grant_sequence": runtime.grant_order,
            "launch_sequence": runtime.launch_order,
            "runtime_events": runtime.event_log.as_dict()["events"],
        }
        if server is not None:
            output["decision_records"] = list(server.coordinator.records)
        return output
    finally:
        runtime.close()
        if server is not None:
            server.close()
        for group in reversed(list(groups.values())):
            try:
                dist.destroy_process_group(group)
            except Exception:
                pass
        dist.destroy_process_group()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=("static_fifo", "static_ltf", "fifo", "ltf", "lookahead"), default="fifo")
    parser.add_argument("--workload", default="balanced")
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--poll-interval", type=float, default=0.001)
    parser.add_argument("--control-port", type=int, required=True)
    parser.add_argument(
        "--fault",
        choices=("none", "missing_task", "metadata_mismatch", "launch_failure", "completion_probe_failure"),
        default="none",
    )
    args = parser.parse_args()
    try:
        output = run_rank(args)
    except BaseException as exc:  # noqa: BLE001
        output = {"rank": int(os.environ.get("RANK", -1)), "status": "error", "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(output, sort_keys=True), flush=True)
        return 1
    print(json.dumps(output, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
