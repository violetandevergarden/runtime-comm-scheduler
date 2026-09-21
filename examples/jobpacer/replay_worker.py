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
    WorkIsCompletedProbe,
)

try:  # Support both package imports in tests and direct script execution.
    from .comm_profile import apply_profile, load_profile, workload_digest
    from .plan_builder import build_plan, planned_tasks, policy_diagnostics, policy_names
    from .workloads import Workload, load_workload, ranks_for_job
except ImportError:  # pragma: no cover - exercised by the subprocess driver
    from comm_profile import apply_profile, load_profile, workload_digest
    from plan_builder import build_plan, planned_tasks, policy_diagnostics, policy_names
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


TRACE_SCHEMA_VERSION = 2


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


class _CompletionObserver:
    """Observe bare-mode Work completion with one rank-local poller."""

    def __init__(self, poll_interval_s: float, on_error) -> None:
        if poll_interval_s <= 0:
            raise ValueError("completion poll interval must be positive")
        self._probe = WorkIsCompletedProbe()
        self._poll_interval_s = poll_interval_s
        self._on_error = on_error
        self._condition = threading.Condition()
        self._pending: dict[object, dict[str, Any]] = {}
        self._error: BaseException | None = None
        self._stopping = False
        self._thread = threading.Thread(
            target=self._run, name="bare-completion-observer"
        )
        self._thread.start()

    def register(self, key: object, work: Any, submit_ts: int) -> dict[str, Any]:
        record = {
            "work": work,
            "submit_ts": submit_ts,
            "collective_call_start_ts": submit_ts,
            "complete_ts": None,
            "completion_observed_ts": None,
            "actual_duration_us": None,
        }
        with self._condition:
            self._raise_if_failed_locked()
            if self._stopping:
                raise RuntimeError("completion observer is stopping")
            if key in self._pending:
                raise ValueError(f"duplicate observed work {key!r}")
            self._pending[key] = record
            self._condition.notify_all()
        return record

    def finish(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._pending and self._error is None:
                remaining = max(0.0, deadline - time.monotonic())
                if remaining == 0:
                    self._stopping = True
                    self._condition.notify_all()
                    break
                self._condition.wait(remaining)
            error = self._error
            timed_out = bool(self._pending)
            self._stopping = True
            self._condition.notify_all()
        self._join(timeout)
        if timed_out:
            raise TimeoutError(
                f"timed out observing {len(self._pending)} bare collective(s)"
            )
        if error is not None:
            raise error

    def close(self, timeout: float) -> None:
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        self._join(timeout)

    @property
    def error(self) -> BaseException | None:
        with self._condition:
            return self._error

    def _run(self) -> None:
        while True:
            with self._condition:
                if self._stopping:
                    return
                pending = list(self._pending.items())
                if not pending:
                    self._condition.wait()
                    continue
            for key, record in pending:
                try:
                    completed = self._probe.is_completed(record["work"])
                except BaseException as exc:  # noqa: BLE001 - retain first probe error
                    with self._condition:
                        if self._error is None:
                            self._error = exc
                        self._pending.clear()
                        self._stopping = True
                        self._condition.notify_all()
                    self._on_error(exc)
                    return
                if completed:
                    complete_ts = _now_us()
                    record["completion_observed_ts"] = complete_ts
                    record["actual_duration_us"] = float(
                        complete_ts - record["collective_call_start_ts"]
                    )
                    with self._condition:
                        self._pending.pop(key, None)
                        self._condition.notify_all()
            with self._condition:
                if not self._stopping and self._pending:
                    self._condition.wait(self._poll_interval_s)

    def _join(self, timeout: float) -> None:
        self._thread.join(timeout=max(0.0, timeout))
        if self._thread.is_alive():
            raise TimeoutError("bare completion observer did not stop")

    def _raise_if_failed_locked(self) -> None:
        if self._error is not None:
            raise self._error


class _GlobalReadyController:
    """Small all-rank control plane for the ready-first comparison."""

    def __init__(self, group, plan_keys, group_ranks: dict[str, tuple[int, ...]], serial: bool):
        self._group = group
        self._plan_keys = tuple(plan_keys)
        self._group_ranks = group_ranks
        self._serial = serial
        self._selected: set[tuple[Any, ...]] = set()
        self._first_ready_round: dict[tuple[Any, ...], int] = {}
        self._round = 0
        self._coordination_wait_us = 0.0
        self._collective_count = 0
        self._completion_barrier_count = 0
        self._control_bytes = 0
        self._coordination_intervals: list[dict[str, Any]] = []

    def choose(self, local_ready) -> tuple[bool, Any]:
        started = _now_us()
        local = [key.as_list() for key in local_ready]
        gathered: list[list[list[Any]]] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, local, group=self._group)
        decision = [None]
        if dist.get_rank(group=self._group) == 0:
            ready_by_rank = [
                {tuple(item) for item in (items or [])} for items in gathered
            ]
            candidates = []
            for key in self._plan_keys:
                serialized = tuple(key.as_list())
                if serialized in self._selected:
                    continue
                if any(
                    earlier.process_group_id == key.process_group_id
                    and earlier.ordinal < key.ordinal
                    and tuple(earlier.as_list()) not in self._selected
                    for earlier in self._plan_keys
                ):
                    continue
                members = self._group_ranks[key.process_group_id]
                if all(serialized in ready_by_rank[rank] for rank in members):
                    self._first_ready_round.setdefault(serialized, self._round)
                    candidates.append(key)
            if len(self._selected) == len(self._plan_keys):
                decision[0] = {"done": True, "key": None}
            elif candidates:
                selected = min(
                    candidates,
                    key=lambda key: (
                        self._first_ready_round[tuple(key.as_list())],
                        tuple(str(item) for item in key.as_list()),
                    ),
                )
                serialized = tuple(selected.as_list())
                self._selected.add(serialized)
                self._round += 1
                decision[0] = {"done": False, "key": serialized}
            else:
                self._round += 1
                decision[0] = {"done": False, "key": None}
        dist.broadcast_object_list(decision, src=0, group=self._group)
        self._collective_count += 2
        self._control_bytes += len(json.dumps(local).encode()) + len(
            json.dumps(decision[0]).encode()
        )
        self._coordination_wait_us += _now_us() - started
        self._coordination_intervals.append(
            {"operation": "ready_round", "start_ts": started, "end_ts": _now_us()}
        )
        if decision[0]["done"]:
            return True, None
        if decision[0]["key"] is None:
            return False, None
        selected = tuple(decision[0]["key"])
        return False, next(key for key in self._plan_keys if tuple(key.as_list()) == selected)

    def wait_for_completion(self, _key) -> None:
        started = _now_us()
        dist.barrier(group=self._group)
        self._completion_barrier_count += 1
        self._control_bytes += 1
        self._coordination_wait_us += _now_us() - started
        self._coordination_intervals.append(
            {"operation": "completion_barrier", "start_ts": started, "end_ts": _now_us()}
        )

    def stats(self) -> dict[str, Any]:
        return {
            "control_collective_count": self._collective_count,
            "global_completion_barrier_count": self._completion_barrier_count,
            "coordination_wait_us": self._coordination_wait_us,
            "control_bytes_estimate": self._control_bytes,
            "coordination_intervals": self._coordination_intervals,
            "global_completion_barrier_before_next_selection": self._serial,
        }


def _run_job(
    job,
    *,
    rank: int,
    device: str,
    tensors: dict[tuple[str, int], torch.Tensor],
    groups: dict[str, Any],
    mode: str,
    scheduler: AdmissionScheduler | None,
    task_by_key: dict,
    errors: list[BaseException],
    stop_event: threading.Event,
    completion_observer: _CompletionObserver | None,
    fault: str,
    estimate_source: str,
    start_barrier: threading.Barrier,
    start_event: threading.Event,
    release_ts: list[int],
) -> dict[str, Any]:
    start_barrier.wait()
    start_event.wait()
    thread_first_run_ts = _now_us()
    job_start_ts = release_ts[0]
    result: dict[str, Any] = {
        "job_id": job.job_id,
        "status": "ok",
        "start_ts": thread_first_run_ts,
        "thread_first_run_ts": thread_first_run_ts,
        "tasks": [],
    }
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
            tensor = tensors[(job.job_id, spec.id)]
            compute_start = _now_us()
            _sleep(spec.producer_compute_s)
            ready_ts = _now_us()
            group = groups[job.job_id]

            def launch(tensor=tensor, group=group):
                return dist.all_reduce(tensor, group=group, async_op=True)

            submit_api_start_ts = _now_us()
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
                timing.producer_compute_start_ts = compute_start
                timing.ready_record_ts = ready_ts
                timing.submit_api_start_ts = submit_api_start_ts
                timing.submit_api_return_ts = timing.submit_api_return_ts or _now_us()
                timing.submit_api_duration_us = float(
                    timing.submit_api_return_ts - submit_api_start_ts
                )
                submit_return_ts = timing.submit_api_return_ts
                submit_ts = None
                observation = None
            else:
                collective_call_start_ts = _now_us()
                underlying = launch()
                collective_call_return_ts = _now_us()
                submit_ts = collective_call_return_ts
                submit_return_ts = collective_call_return_ts
                observation = completion_observer.register(
                    task.key, work=underlying, submit_ts=collective_call_start_ts
                )
                observation.update(
                    {
                        "submit_api_start_ts": submit_api_start_ts,
                        "submit_api_return_ts": submit_return_ts,
                        "collective_call_start_ts": collective_call_start_ts,
                        "collective_call_return_ts": collective_call_return_ts,
                        "admit_ts": collective_call_start_ts,
                    }
                )
                timing = None
                work = underlying
            consumer_start = _now_us()
            _sleep(spec.consumer_compute_s)
            consumer_compute_end = _now_us()
            application_wait_start_ts = _now_us()
            if not work.wait():
                raise RuntimeError(
                    f"collective wait returned false for {task.key.as_list()}"
                )
            wait_return_ts = _now_us()
            application_task_end_ts = wait_return_ts
            if timing is not None:
                timing.consumer_compute_start_ts = consumer_start
                timing.consumer_compute_end_ts = consumer_compute_end
                timing.set_wait_start(application_wait_start_ts)
                timing.set_underlying_wait_start(application_wait_start_ts)
                timing.set_wait_return(wait_return_ts)
                timing.application_task_end_ts = application_task_end_ts
            else:
                observation.update(
                    {
                        "producer_compute_start_ts": compute_start,
                        "ready_record_ts": ready_ts,
                        "consumer_compute_start_ts": consumer_start,
                        "consumer_compute_end_ts": consumer_compute_end,
                        "application_wait_start_ts": application_wait_start_ts,
                        "underlying_wait_start_ts": application_wait_start_ts,
                        "wait_return_ts": wait_return_ts,
                        "application_task_end_ts": application_task_end_ts,
                    }
                )
            task_result = {
                "key": _serialize_key(task.key),
                "job_id": job.job_id,
                "ordinal": spec.id,
                "num_bytes": spec.num_bytes,
                "estimated_comm_s": spec.estimated_comm_s,
                "estimate_source": estimate_source,
                # Filled by the deferred validation pass after communication
                # drain.  Keeping the field on each task preserves the trace
                # contract without putting a tensor scan on the application path.
                "correct": None,
                "producer_compute_start_ts": compute_start,
                "ready_record_ts": ready_ts,
                "producer_compute_end_ts": ready_ts,
                "submit_api_start_ts": (
                    timing.submit_api_start_ts
                    if timing
                    else observation["submit_api_start_ts"]
                ),
                "submit_api_return_ts": (
                    timing.submit_api_return_ts
                    if timing
                    else observation["submit_api_return_ts"]
                ),
                "collective_call_start_ts": (
                    timing.collective_call_start_ts
                    if timing
                    else observation["collective_call_start_ts"]
                ),
                "collective_call_return_ts": (
                    timing.collective_call_return_ts
                    if timing
                    else observation["collective_call_return_ts"]
                ),
                "submit_call_ts": (
                    timing.submit_api_start_ts
                    if timing
                    else observation["submit_api_start_ts"]
                ),
                "submit_return_ts": submit_return_ts,
                "consumer_compute_start_ts": consumer_start,
                "consumer_compute_end_ts": consumer_compute_end,
                "application_wait_start_ts": (
                    timing.application_wait_start_ts
                    if timing
                    else observation["application_wait_start_ts"]
                ),
                "underlying_wait_start_ts": (
                    timing.underlying_wait_start_ts
                    if timing
                    else observation["underlying_wait_start_ts"]
                ),
                "wait_return_ts": wait_return_ts,
                "application_task_end_ts": application_task_end_ts,
                "submit_ts": timing.submit_ts if timing else submit_ts,
                "admit_ts": timing.admit_ts if timing else observation["admit_ts"],
                "completion_observed_ts": (
                    timing.completion_observed_ts if timing else observation["completion_observed_ts"]
                ),
                "actual_duration_us": (
                    timing.actual_duration_us
                    if timing
                    else observation["actual_duration_us"]
                ),
                "submit_api_duration_us": (
                    timing.submit_api_duration_us
                    if timing
                    else observation["submit_api_return_ts"]
                    - observation["submit_api_start_ts"]
                ),
                "collective_call_duration_us": (
                    timing.collective_call_duration_us
                    if timing
                    else observation["collective_call_return_ts"]
                    - observation["collective_call_start_ts"]
                ),
                "deferred_binding_wait_us": (
                    timing.deferred_binding_wait_us
                    if timing
                    else observation["underlying_wait_start_ts"]
                    - observation["application_wait_start_ts"]
                ),
                "underlying_wait_duration_us": (
                    timing.underlying_wait_duration_us
                    if timing
                    else wait_return_ts - application_wait_start_ts
                ),
                "call_to_completion_observation_us": (
                    timing.call_to_completion_observation_us
                    if timing
                    else None
                ),
                "mode": mode,
            }
            task_result["first_wait_ts"] = task_result["application_wait_start_ts"]
            task_result["complete_ts"] = task_result["completion_observed_ts"]
            task_result["consumer_end_ts"] = application_task_end_ts
            if observation is not None:
                task_result["_completion_record"] = observation
            result["tasks"].append(task_result)
    except BaseException as exc:  # noqa: BLE001 - worker reports the first error
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        errors.append(exc)
        stop_event.set()
    finally:
        job_end_ts = _now_us()
        result["end_ts"] = job_end_ts
        result["application_end_ts"] = job_end_ts
        result["makespan_us"] = job_end_ts - thread_first_run_ts
        result["application_makespan_us"] = result["makespan_us"]
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
    control_group = dist.new_group(ranks=list(range(world_size)))
    dist.barrier()
    applied_workload_digest = _check_workload_digest(workload)
    plan = build_plan(
        workload, args.policy, version=args.plan_version, window_id=args.window_id
    )
    _check_plan_digest(plan)
    selection_diagnostics = policy_diagnostics(
        workload, args.policy, window_id=args.window_id
    )
    ltf_scores = {
        tuple(step["selected_key"]): step.get("selected_score")
        for step in selection_diagnostics["steps"]
    }
    task_by_key = planned_tasks(workload, args.policy, window_id=args.window_id)
    local_jobs = [
        job for job in workload.jobs if rank in ranks_for_job(job, world_size)
    ]
    # Keep one-time tensor allocation and ProcessGroup connection setup outside
    # the measured replay window. Every rank warms groups in manifest order.
    tensors = {
        (job.job_id, spec.id): _make_tensor(spec, rank, device)
        for job in local_jobs
        for spec in job.communications
    }
    for job in local_jobs:
        dist.barrier(group=groups[job.job_id])
    scheduler = None
    errors: list[BaseException] = []
    stop_event = threading.Event()
    jobs_result: list[dict[str, Any]] = []
    threads: list[threading.Thread] = []
    result_by_job: dict[str, dict[str, Any]] = {}
    completed = False
    completion_observer = None
    selection_controller = None
    start_barrier = threading.Barrier(len(local_jobs) + 1)
    start_event = threading.Event()
    try:
        if args.mode == "scheduler":
            executor = (
                TorchProcessGroupExecutor(torch.cuda.current_device())
                if args.backend == "nccl"
                else DirectLaunchExecutor()
            )
            if args.selection == "ready_first":
                selection_controller = _GlobalReadyController(
                    control_group,
                    plan.keys,
                    {
                        job.job_id: ranks_for_job(job, world_size)
                        for job in workload.jobs
                    },
                    serial=args.max_outstanding == 1,
                )
            scheduler = AdmissionScheduler(
                plan,
                local_group_ids={job.job_id for job in local_jobs},
                executor=executor,
                max_outstanding=args.max_outstanding,
                completion_poll_interval_s=args.completion_poll_interval_s,
                selection_controller=selection_controller,
                selection_serial=args.max_outstanding == 1 and selection_controller is not None,
            )
        else:
            completion_observer = _CompletionObserver(
                args.completion_poll_interval_s,
                lambda _error: stop_event.set(),
            )

        def run_one(job):
            result_by_job[job.job_id] = _run_job(
                job,
                rank=rank,
                device=device,
                tensors=tensors,
                groups=groups,
                mode=args.mode,
                scheduler=scheduler,
                task_by_key=task_by_key,
                errors=errors,
                stop_event=stop_event,
                completion_observer=completion_observer,
                fault=args.fault,
                estimate_source="offline_profile" if profile else "manifest",
                start_barrier=start_barrier,
                start_event=start_event,
                release_ts=release_timestamp,
            )

        release_timestamp: list[int] = []
        for job in local_jobs:
            thread = threading.Thread(
                target=run_one, args=(job,), name=f"replay-{job.job_id}"
            )
            threads.append(thread)
            thread.start()
        start_barrier.wait()
        dist.barrier()
        application_release_ts = _now_us()
        release_timestamp.append(application_release_ts)
        start_event.set()
        for thread in threads:
            thread.join(timeout=args.thread_timeout)
        application_end_ts = _now_us()
        if any(thread.is_alive() for thread in threads):
            raise TimeoutError(
                f"job thread did not finish within {args.thread_timeout}s"
            )
        jobs_result = [result_by_job[job.job_id] for job in local_jobs]
        if errors:
            raise errors[0]
        if args.mode == "bare":
            missing = [
                job.job_id
                for job in local_jobs
                if len(result_by_job[job.job_id]["tasks"]) != len(job.communications)
            ]
            if missing:
                raise RuntimeError(f"bare replay has incomplete jobs: {missing}")
        if scheduler is not None:
            scheduler.finish_window(timeout=args.finish_timeout)
            timing_by_key = {
                tuple(timing.key.as_list()): timing for timing in scheduler.timings()
            }
            for job_result in jobs_result:
                for task_result in job_result["tasks"]:
                    timing = timing_by_key[tuple(task_result["key"])]
                    task_result.update(
                        {
                            "admit_ts": timing.admit_ts,
                            "submit_ts": timing.submit_ts,
                            "collective_call_start_ts": timing.collective_call_start_ts,
                            "collective_call_return_ts": timing.collective_call_return_ts,
                            "completion_observed_ts": timing.completion_observed_ts,
                            "complete_ts": timing.completion_observed_ts,
                            "actual_duration_us": timing.actual_duration_us,
                            "call_to_completion_observation_us": timing.call_to_completion_observation_us,
                            "post_return_completion_observation_us": timing.post_return_completion_observation_us,
                        }
                    )
        if completion_observer is not None:
            completion_observer.finish(args.finish_timeout)
            for job_result in jobs_result:
                for task_result in job_result["tasks"]:
                    observation = task_result.pop("_completion_record")
                    task_result["completion_observed_ts"] = observation[
                        "completion_observed_ts"
                    ]
                    task_result["complete_ts"] = observation[
                        "completion_observed_ts"
                    ]
                    task_result["actual_duration_us"] = observation[
                        "actual_duration_us"
                    ]
                    task_result["submit_api_duration_us"] = float(
                        observation["submit_api_return_ts"]
                        - observation["submit_api_start_ts"]
                    )
                    task_result["collective_call_duration_us"] = float(
                        observation["collective_call_return_ts"]
                        - observation["collective_call_start_ts"]
                    )
                    task_result["post_return_completion_observation_us"] = float(
                        observation["completion_observed_ts"]
                        - observation["collective_call_return_ts"]
                    )
                    task_result["call_to_completion_observation_us"] = float(
                        observation["completion_observed_ts"]
                        - observation["collective_call_start_ts"]
                    )
        communication_drain_end_ts = _now_us()
        for job_result in jobs_result:
            job_end = job_result["application_end_ts"]
            for task_result in job_result["tasks"]:
                task_result["ready_to_job_end_observed_us"] = float(
                    job_end - task_result["ready_record_ts"]
                )
                task_result["completion_to_job_end_observed_us"] = float(
                    job_end - task_result["completion_observed_ts"]
                )
                task_result["estimated_remaining_critical_path_s"] = ltf_scores.get(
                    tuple(task_result["key"])
                )

        validation_start_ts = _now_us()
        validation_error: BaseException | None = None
        timing_by_key = {}
        if scheduler is not None:
            timing_by_key = {timing.key: timing for timing in scheduler.timings()}
        for job in local_jobs:
            job_result = result_by_job[job.job_id]
            for task_result in job_result["tasks"]:
                task_result["validation_start_ts"] = _now_us()
                key = task_result["key"]
                timing = next(
                    (
                        item
                        for item_key, item in timing_by_key.items()
                        if item_key.as_list() == key
                    ),
                    None,
                )
                if timing is not None:
                    timing.validation_start_ts = task_result["validation_start_ts"]
                spec = job.communications[int(task_result["ordinal"])]
                tensor = tensors[(job.job_id, spec.id)]
                expected = _expected_sum(ranks_for_job(job, world_size))
                correct = bool(torch.all(tensor == expected).item())
                task_result["correct"] = correct
                task_result["validation_end_ts"] = _now_us()
                task_result["validation_duration_us"] = float(
                    task_result["validation_end_ts"]
                    - task_result["validation_start_ts"]
                )
                if timing is not None:
                    timing.validation_end_ts = task_result["validation_end_ts"]
                    timing.validation_duration_us = task_result[
                        "validation_duration_us"
                    ]
                if not correct and validation_error is None:
                    validation_error = AssertionError(
                        f"incorrect all_reduce result for {task_result['key']}"
                    )
        validation_end_ts = _now_us()
        if validation_error is not None:
            raise validation_error

        harness_start_ts = _now_us()
        trace: dict[str, Any] = {
            "rank": rank,
            "mode": args.mode,
            "policy": args.policy,
            "selection": args.selection,
            "scenario": args.scenario,
            "backend": args.backend,
            "world_size": world_size,
            "completion_poll_interval_s": args.completion_poll_interval_s,
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
                "policy": args.policy,
                "score_definition": selection_diagnostics.get("score_definition"),
                "selection_diagnostics": selection_diagnostics,
            },
            "trace_schema_version": TRACE_SCHEMA_VERSION,
            "max_outstanding": (
                args.max_outstanding if args.mode == "scheduler" else None
            ),
            "application_release_ts": application_release_ts,
            "application_end_ts": application_end_ts,
            "communication_drain_end_ts": communication_drain_end_ts,
            "validation_start_ts": validation_start_ts,
            "validation_end_ts": validation_end_ts,
            "harness_start_ts": harness_start_ts,
            "harness_end_ts": None,
            "application_makespan_us": application_end_ts - application_release_ts,
            "communication_drain_makespan_us": (
                communication_drain_end_ts - application_release_ts
            ),
            "validation_total_us": validation_end_ts - validation_start_ts,
            # Compatibility names. New summaries use the explicit boundaries.
            "replay_start_ts": application_release_ts,
            "replay_end_ts": application_end_ts,
            "replay_makespan_us": application_end_ts - application_release_ts,
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
                            "application_wait_start_ts": timing.application_wait_start_ts,
                            "underlying_wait_start_ts": timing.underlying_wait_start_ts,
                            "first_wait_ts": timing.application_wait_start_ts,
                            "wait_return_ts": timing.wait_return_ts,
                            "admit_ts": timing.admit_ts,
                            "submit_ts": timing.submit_ts,
                            "collective_call_start_ts": timing.collective_call_start_ts,
                            "collective_call_return_ts": timing.collective_call_return_ts,
                            "completion_observed_ts": timing.completion_observed_ts,
                            "complete_ts": timing.completion_observed_ts,
                            "actual_duration_us": timing.actual_duration_us,
                            "submit_api_duration_us": timing.submit_api_duration_us,
                            "collective_call_duration_us": timing.collective_call_duration_us,
                            "post_return_completion_observation_us": timing.post_return_completion_observation_us,
                            "deferred_binding_wait_us": timing.deferred_binding_wait_us,
                            "underlying_wait_duration_us": timing.underlying_wait_duration_us,
                            "call_to_completion_observation_us": timing.call_to_completion_observation_us,
                            "validation_start_ts": timing.validation_start_ts,
                            "validation_end_ts": timing.validation_end_ts,
                            "validation_duration_us": timing.validation_duration_us,
                        }
                    )
        else:
            if completion_observer is not None and completion_observer.error is not None:
                raise completion_observer.error
            bare_tasks = [
                task
                for job_result in jobs_result
                for task in job_result["tasks"]
            ]
            ordered_bare_tasks = sorted(
                bare_tasks,
                key=lambda task: (task["submit_ts"], task["job_id"]),
            )
            trace["launch_sequence"] = [task["key"] for task in ordered_bare_tasks]
            trace["group_sequence"] = {
                job.job_id: [
                    task["key"]
                    for task in ordered_bare_tasks
                    if task["job_id"] == job.job_id
                ]
                for job in local_jobs
            }
            trace["timings"] = []
        if selection_controller is not None:
            trace["control"] = selection_controller.stats()
        else:
            trace["control"] = {
                "control_collective_count": 0,
                "global_completion_barrier_count": 0,
                "coordination_wait_us": 0.0,
                "global_completion_barrier_before_next_selection": False,
            }
        trace["harness_end_ts"] = _now_us()
        trace["harness_total_us"] = trace["harness_end_ts"] - harness_start_ts
        completed = True
        return trace
    finally:
        if completion_observer is not None:
            try:
                completion_observer.close(args.finish_timeout)
            except BaseException as exc:  # noqa: BLE001 - preserve result error
                errors.append(exc)
        if scheduler is not None:
            try:
                scheduler.close()
            except BaseException as exc:  # noqa: BLE001 - preserve result error
                errors.append(exc)
        if completed:
            try:
                dist.destroy_process_group(control_group)
            except Exception:
                pass
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
    parser.add_argument(
        "--selection",
        choices=("runtime_arrival", "ready_first"),
        default="runtime_arrival",
    )
    parser.add_argument("--scenario")
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
