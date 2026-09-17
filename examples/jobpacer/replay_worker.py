"""One rank of the JobPacer phase 2 replay."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from runtime_comm_scheduler import (
    AdmissionScheduler,
    CommIntent,
    DirectLaunchExecutor,
    TorchProcessGroupExecutor,
)

try:  # Support both package imports in tests and direct script execution.
    from .comm_profile import apply_profile, load_profile, workload_digest
    from .plan_builder import build_plan, planned_tasks, policy_names
    from .workloads import Workload, load_workload, ranks_for_job
except ImportError:  # pragma: no cover - exercised by the subprocess driver
    from comm_profile import apply_profile, load_profile, workload_digest
    from plan_builder import build_plan, planned_tasks, policy_names
    from workloads import Workload, load_workload, ranks_for_job


def _now_us() -> int:
    return time.perf_counter_ns() // 1000


def _sleep(seconds: float) -> None:
    if seconds:
        time.sleep(seconds)


def _serialize_key(key) -> list[Any]:
    return key.as_list()


def _check_plan_digest(plan) -> None:
    observed: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(observed, plan.digest())
    if any(digest != plan.digest() for digest in observed):
        raise RuntimeError(f"plan digest mismatch across ranks: {observed}")


def _check_workload_digest(workload: Workload) -> str:
    digest = workload_digest(workload)
    observed: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(observed, digest)
    if any(item != digest for item in observed):
        raise RuntimeError(f"workload digest mismatch across ranks: {observed}")
    return digest


def _new_groups(workload: Workload, world_size: int) -> dict[str, Any]:
    # Every rank creates every group in manifest order. Membership can be an
    # arbitrary, but identical, global-rank tuple on every rank.
    return {
        job.job_id: dist.new_group(ranks=list(ranks_for_job(job, world_size)))
        for job in workload.jobs
    }


def _make_tensor(spec, rank: int, device: str) -> torch.Tensor:
    return torch.full(
        (spec.num_bytes // 4,),
        float(rank + 1),
        dtype=torch.float32,
        device=device,
    )


def _expected_sum(ranks: tuple[int, ...]) -> float:
    return sum(rank + 1 for rank in ranks)


def _run_job(
    job,
    *,
    rank: int,
    device: str,
    groups: dict[str, Any],
    mode: str,
    scheduler: AdmissionScheduler | None,
    task_by_key: dict,
    errors: list[BaseException],
    stop_event: threading.Event,
    fault: str,
    estimate_source: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {"job_id": job.job_id, "status": "ok", "tasks": []}
    try:
        for spec in job.communications:
            if stop_event.is_set():
                raise RuntimeError("another job thread failed")
            if (
                fault == "missing_key"
                and rank == 1
                and job is not None
                and job.job_id == "job-1"
                and spec.id == job.communications[-1].id
            ):
                # Fault injection used by the bounded-exit acceptance check.
                continue
            task = next(
                item
                for item in task_by_key.values()
                if item.job_id == job.job_id
                and item.communication.id == spec.id
            )
            tensor = _make_tensor(spec, rank, device)
            compute_start = _now_us()
            _sleep(spec.producer_compute_s)
            ready_ts = _now_us()
            group = groups[job.job_id]

            def launch(tensor=tensor, group=group):
                return dist.all_reduce(tensor, group=group, async_op=True)

            if mode == "scheduler":
                op = (
                    "all_gather"
                    if fault == "metadata_mismatch" and rank == 1 and spec.id == 0
                    else spec.op
                )
                num_bytes = (
                    spec.num_bytes + 4
                    if fault == "metadata_mismatch" and rank == 1 and spec.id == 0
                    else spec.num_bytes
                )
                intent = CommIntent(
                    key=task.key,
                    op=op,
                    tensor=tensor,
                    process_group=group,
                    num_bytes=num_bytes,
                    launch_fn=launch,
                    producer=f"{job.job_id}:compute:{spec.id}",
                    consumer=f"{job.job_id}:compute:{spec.id}:consumer",
                    device=device,
                    keepalive=(tensor,),
                )
                work = scheduler.submit(intent)
                timing = work.timing
                submit_ts = None
            else:
                submit_ts = _now_us()
                underlying = launch()
                timing = None
                work = underlying
            compute_end = _now_us()
            consumer_start = _now_us()
            _sleep(spec.consumer_compute_s)
            first_wait_ts = _now_us()
            if not work.wait():
                raise RuntimeError(
                    f"collective wait returned false for {task.key.as_list()}"
                )
            complete_ts = _now_us()
            expected = _expected_sum(ranks_for_job(job, dist.get_world_size()))
            correct = bool(torch.all(tensor == expected).item())
            if not correct:
                raise AssertionError(
                    f"incorrect all_reduce result for {task.key.as_list()}"
                )
            task_result = {
                "key": _serialize_key(task.key),
                "job_id": job.job_id,
                "ordinal": spec.id,
                "num_bytes": spec.num_bytes,
                "estimated_comm_s": spec.estimated_comm_s,
                "estimate_source": estimate_source,
                "correct": correct,
                "producer_compute_start_ts": compute_start,
                "ready_record_ts": ready_ts,
                "producer_compute_end_ts": compute_end,
                "consumer_compute_start_ts": consumer_start,
                "first_wait_ts": timing.first_wait_ts if timing else first_wait_ts,
                "consumer_end_ts": complete_ts,
                "submit_ts": timing.submit_ts if timing else submit_ts,
                "admit_ts": timing.admit_ts if timing else submit_ts,
                "complete_ts": timing.complete_ts if timing else complete_ts,
                "mode": mode,
            }
            result["tasks"].append(task_result)
    except BaseException as exc:  # noqa: BLE001 - worker reports the first error
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        errors.append(exc)
        stop_event.set()
    return result


def run_rank(args: argparse.Namespace) -> dict[str, Any]:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size < 2:
        raise ValueError("JobPacer replay requires at least two ranks")
    workload = load_workload(args.workload)
    if args.backend == "nccl":
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        device = f"cuda:{torch.cuda.current_device()}"
    else:
        device = "cpu"

    profile = load_profile(args.comm_profile) if args.comm_profile else None
    if profile is not None:
        workload = apply_profile(
            workload,
            profile,
            {"backend": args.backend, "device_type": "cuda" if args.backend == "nccl" else "cpu", "world_size": world_size},
            strict=args.profile_strict,
        )

    dist.init_process_group(backend=args.backend)
    groups = _new_groups(workload, world_size)
    dist.barrier()
    applied_workload_digest = _check_workload_digest(workload)
    plan = build_plan(
        workload, args.policy, version=args.plan_version, window_id=args.window_id
    )
    _check_plan_digest(plan)
    task_by_key = planned_tasks(workload, args.policy, window_id=args.window_id)
    local_jobs = [
        job for job in workload.jobs if rank in ranks_for_job(job, world_size)
    ]
    scheduler = None
    errors: list[BaseException] = []
    stop_event = threading.Event()
    jobs_result: list[dict[str, Any]] = []
    threads: list[threading.Thread] = []
    result_by_job: dict[str, dict[str, Any]] = {}
    completed = False
    try:
        if args.mode == "scheduler":
            executor = (
                TorchProcessGroupExecutor(torch.cuda.current_device())
                if args.backend == "nccl"
                else DirectLaunchExecutor()
            )
            scheduler = AdmissionScheduler(
                plan,
                local_group_ids={job.job_id for job in local_jobs},
                executor=executor,
                max_outstanding=args.max_outstanding,
                completion_poll_interval_s=args.completion_poll_interval_s,
            )

        def run_one(job):
            result_by_job[job.job_id] = _run_job(
                job,
                rank=rank,
                device=device,
                groups=groups,
                mode=args.mode,
                scheduler=scheduler,
                task_by_key=task_by_key,
                errors=errors,
                stop_event=stop_event,
                fault=args.fault,
                estimate_source="offline_profile" if profile else "manifest",
            )

        for job in local_jobs:
            thread = threading.Thread(
                target=run_one, args=(job,), name=f"replay-{job.job_id}"
            )
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join(timeout=args.thread_timeout)
        if any(thread.is_alive() for thread in threads):
            raise TimeoutError(
                f"job thread did not finish within {args.thread_timeout}s"
            )
        jobs_result = [result_by_job[job.job_id] for job in local_jobs]
        if errors:
            raise errors[0]
        if scheduler is not None:
            scheduler.finish_window(timeout=args.finish_timeout)

        trace: dict[str, Any] = {
            "rank": rank,
            "mode": args.mode,
            "policy": args.policy,
            "backend": args.backend,
            "world_size": world_size,
            "workload": workload.to_dict(),
            "workload_digest": applied_workload_digest,
            "communication_profile": {
                "path": str(args.comm_profile) if args.comm_profile else None,
                "digest": profile.digest() if profile else None,
                "schema_version": profile.schema_version if profile else None,
                "strict": args.profile_strict if profile else None,
                "environment": dict(profile.environment) if profile else None,
                "estimate_source": "offline_profile" if profile else "manifest",
            },
            "plan": {
                "version": plan.version,
                "window_id": plan.window_id,
                "digest": plan.digest(),
                "keys": [_serialize_key(key) for key in plan.keys],
                "labels": [
                    f"{key.process_group_id}:{key.ordinal}" for key in plan.keys
                ],
            },
            "max_outstanding": (
                args.max_outstanding if args.mode == "scheduler" else None
            ),
            "jobs": jobs_result,
            "status": "ok",
        }
        if scheduler is not None:
            trace["launch_sequence"] = [
                _serialize_key(key) for key in scheduler.sequence_log()
            ]
            trace["group_sequence"] = {
                group_id: [_serialize_key(key) for key in sequence]
                for group_id, sequence in scheduler.group_sequence_log().items()
            }
            timing_records = scheduler.timings()
            trace["timings"] = [
                asdict(timing) | {"key": _serialize_key(timing.key)}
                for timing in timing_records
            ]
            by_key = {timing.key: timing for timing in timing_records}
            for job_result in jobs_result:
                for task_result in job_result["tasks"]:
                    key = tuple(task_result["key"])
                    timing = next(
                        timing
                        for task_key, timing in by_key.items()
                        if tuple(task_key.as_list()) == key
                    )
                    task_result.update(
                        {
                            "first_wait_ts": timing.first_wait_ts,
                            "admit_ts": timing.admit_ts,
                            "submit_ts": timing.submit_ts,
                            "complete_ts": timing.complete_ts,
                        }
                    )
        else:
            trace["launch_sequence"] = []
            trace["group_sequence"] = {}
            trace["timings"] = []
        completed = True
        return trace
    finally:
        if scheduler is not None:
            try:
                scheduler.close()
            except BaseException as exc:  # noqa: BLE001 - preserve result error
                errors.append(exc)
        if completed:
            for group in reversed(list(groups.values())):
                try:
                    dist.destroy_process_group(group)
                except Exception:
                    pass
            dist.destroy_process_group()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("bare", "scheduler"), required=True)
    parser.add_argument("--policy", choices=policy_names(), default="fifo")
    parser.add_argument("--workload", default="balanced")
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    parser.add_argument("--max-outstanding", type=int, default=1)
    parser.add_argument("--plan-version", type=int, default=0)
    parser.add_argument("--window-id", type=int, default=0)
    parser.add_argument("--finish-timeout", type=float, default=20.0)
    parser.add_argument("--thread-timeout", type=float, default=20.0)
    parser.add_argument("--completion-poll-interval-s", type=float, default=0.001)
    parser.add_argument("--comm-profile", type=Path)
    parser.add_argument("--profile-strict", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--fault", choices=("none", "missing_key", "metadata_mismatch"), default="none"
    )
    args = parser.parse_args()
    output: dict[str, Any]
    try:
        output = run_rank(args)
        status = 0
    except BaseException as exc:  # noqa: BLE001 - JSON is the driver contract
        output = {
            "rank": int(os.environ.get("RANK", -1)),
            "mode": args.mode,
            "policy": args.policy,
            "backend": args.backend,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }
        status = 1
    print(json.dumps(output, sort_keys=True), flush=True)
    return status


if __name__ == "__main__":
    sys.exit(main())
