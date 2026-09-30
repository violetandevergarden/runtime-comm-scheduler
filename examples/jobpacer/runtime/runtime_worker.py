"""One-rank JobPacer linear or validated-DAG replay worker."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import threading
import time
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from runtime_comm_scheduler.dag import CommNode, ComputeNode, DagJob, DagRunner, dag_task_spec, build_static_order
from runtime_comm_scheduler.dag.model import compute_tails
from runtime_comm_scheduler.runtime import (
    CollectiveSpec, CudaCollectiveExecutor, DirectExecutor, EventLog,
    LocalBinding, RankRuntime,
)
from runtime_comm_scheduler.runtime.transport import ControlClient
from examples.jobpacer.runtime.runtime_adapter import (
    apply_dag_compute_profile,
    DagInput,
    apply_dag_profile,
    group_spec,
    linear_static_order,
    load_dag,
    load_static_order,
    linear_ltf_estimates,
    make_collective_binding,
    make_replay_compute,
    task_hint,
    task_spec,
)
from examples.jobpacer.gpu.gpu_dag_resources import GpuDagResources
from examples.jobpacer.gpu.cuda_devices import compute_profile_software, visible_cuda_uuids
from examples.jobpacer.workloads import Job, Workload, linear_execution_duration, load_workload, ranks_for_job
from examples.jobpacer.comm_profile import apply_profile, load_profile
from examples.jobpacer.gpu.gpu_compute import CudaMatmulProgram


class _FailingProbe:
    supports_physical_completion = True

    def __init__(self):
        self.failed = False

    def is_completed(self, work):
        if not self.failed:
            self.failed = True
            raise RuntimeError("injected completion probe failure")
        return bool(work.is_completed())


class _DroppedHandle:
    state = type("PendingState", (), {"value": "pending"})()

    def wait_host(self, timeout):
        return False


class _DropOneSubmit:
    """Fault-only proxy: pretend one OFFER was dropped so the runner hits its deadline."""

    def __init__(self, runtime, missing_task_id):
        self._runtime = runtime
        self.missing_task_id = missing_task_id
        self.dropped = False

    @property
    def failure(self):
        return self._runtime.failure

    def declare(self, *args, **kwargs):
        return self._runtime.declare(*args, **kwargs)

    def submit(self, spec, binding, hint):
        if not self.dropped and spec.task_id == self.missing_task_id:
            self.dropped = True
            return _DroppedHandle()
        return self._runtime.submit(spec, binding, hint)

    def abort(self, *args, **kwargs):
        return self._runtime.abort(*args, **kwargs)


def _make_linear_binding(spec, group, *, rank: int, device: str, fault: str,
                         job_id: str, ordinal: int) -> tuple[LocalBinding, dict[str, Any]]:
    start_ts = time.perf_counter_ns() // 1000
    if fault == "binding_failure" and rank == 0 and job_id == "job-0" and ordinal == 0:
        raise RuntimeError("injected linear binding failure")
    binding = make_collective_binding(spec, group, rank=rank, device=device)
    end_ts = time.perf_counter_ns() // 1000
    return binding, {
        "task_id": spec.task_id, "job_id": job_id, "ordinal": ordinal,
        "binding_create_start_ts": start_ts, "binding_create_end_ts": end_ts,
        "binding_create_duration_us": end_ts - start_ts,
    }


def _prepare_linear_bindings(workload: Workload, local_jobs: list[Job], *, groups,
                             rank: int, world_size: int, device: str, epoch: int, fault: str
                             ) -> tuple[dict[str, LocalBinding], list[dict[str, Any]], int, int]:
    """Create every rank-local linear tensor/binding before application release."""
    start_ts = time.perf_counter_ns() // 1000
    bindings: dict[str, LocalBinding] = {}
    events: list[dict[str, Any]] = []
    local_error: BaseException | None = None
    try:
        for job in local_jobs:
            for comm in job.communications:
                spec = task_spec(job, comm, epoch=epoch)
                binding, event = _make_linear_binding(
                    spec, groups[job.job_id], rank=rank, device=device, fault=fault,
                    job_id=job.job_id, ordinal=comm.id,
                )
                if spec.task_id in bindings:
                    raise ValueError(f"duplicate precreated binding for {spec.task_id}")
                bindings[spec.task_id] = binding
                events.append(event)
    except BaseException as exc:  # synchronize a local allocation failure across ranks
        local_error = exc

    errors: list[str | None] = [None] * world_size
    dist.all_gather_object(errors, None if local_error is None else
                           f"{type(local_error).__name__}: {local_error}")
    end_ts = time.perf_counter_ns() // 1000
    failures = [(endpoint, error) for endpoint, error in enumerate(errors) if error]
    if failures:
        if local_error is not None:
            raise local_error
        raise RuntimeError(f"linear binding preparation failed on another rank: {failures}")
    return bindings, events, start_ts, end_ts


def _free_device(backend: str) -> str:
    if backend == "nccl":
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        return f"cuda:{local_rank}"
    return "cpu"


def _new_groups(workload: Workload | DagInput, world_size: int,
                setup_deadline: float | None = None,
                group_timeout_s: float | None = None) -> dict[str, Any]:
    if isinstance(workload, DagInput):
        group_defs = ((group.group_id, group.ranks) for group in workload.graph.groups)
    else:
        group_defs = ((job.job_id, ranks_for_job(job, world_size)) for job in workload.jobs)
    # Every process creates communicators in the same manifest order.
    groups: dict[str, Any] = {}
    try:
        for group_id, ranks in group_defs:
            remaining = None if setup_deadline is None else _remaining(setup_deadline)
            if remaining is not None and remaining <= 0:
                raise TimeoutError("setup deadline exceeded while creating process groups")
            timeout_s = remaining
            if timeout_s is not None and group_timeout_s is not None:
                timeout_s = min(timeout_s, group_timeout_s)
            timeout = {} if timeout_s is None else {"timeout": timedelta(seconds=timeout_s)}
            groups[group_id] = dist.new_group(list(ranks), **timeout)
    except BaseException:
        for group in reversed(list(groups.values())):
            try:
                dist.destroy_process_group(group)
            except Exception:
                pass
        raise
    return groups


def _remaining(deadline: float) -> float:
    return max(0.0, deadline - time.monotonic())


def _warmup_collectives(inputs: Workload | DagInput, groups: dict[str, Any], *, rank: int,
                        world_size: int, device: str, iterations: int) -> int:
    """Warm each measured group/signature before runtime registration and timing."""
    if iterations < 0:
        raise ValueError("warmup_iterations must be non-negative")
    if iterations == 0:
        return 0
    signatures: list[tuple[str, tuple[int, ...], int]] = []
    if isinstance(inputs, DagInput):
        group_ranks = inputs.group_ranks
        for group in inputs.graph.groups:
            sizes = sorted({node.collective.num_bytes for job in inputs.graph.jobs for node in job.nodes
                            if isinstance(node, CommNode) and node.group_id == group.group_id})
            signatures.extend((group.group_id, group_ranks[group.group_id], size) for size in sizes)
    else:
        for job in inputs.jobs:
            ranks = ranks_for_job(job, world_size)
            signatures.extend((job.job_id, ranks, size)
                              for size in sorted({comm.num_bytes for comm in job.communications}))
    count = 0
    for group_id, ranks, num_bytes in signatures:
        if rank not in ranks:
            continue
        tensor = torch.full((num_bytes // 4,), float(rank + 1), dtype=torch.float32, device=device)
        for _ in range(iterations):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=groups[group_id])
            tensor.fill_(float(rank + 1))
            count += 1
        dist.barrier(group=groups[group_id])
    dist.barrier()
    return count


def _local_dag_jobs(dag: DagInput, rank: int) -> list[DagJob]:
    group_ranks = dag.group_ranks
    return [job for job in dag.graph.jobs
            if rank in group_ranks[next(node.group_id for node in job.nodes if isinstance(node, CommNode))]]



def _prepare_dag_gpu_compute(dag: DagInput, local_jobs: list[DagJob], *, device: str,
                             rank: int, world_size: int, matrix_size: int,
                             repeats: int, epoch: int, warmup_iterations: int
                             ) -> tuple[dict[str, CudaMatmulProgram], int, int]:
    """Allocate and warm fixed CUDA compute programs before the release barrier."""
    started = time.perf_counter_ns() // 1000
    programs: dict[str, CudaMatmulProgram] = {}
    local_error: BaseException | None = None
    try:
        for job in local_jobs:
            for node in job.nodes:
                if not isinstance(node, ComputeNode):
                    continue
                key = f"{dag.seed}:{epoch}:{job.job_id}:{node.node_id}:{rank}:compute"
                seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
                program = CudaMatmulProgram(device, matrix_size=matrix_size,
                                            repeats=repeats, seed=seed)
                program.warmup(warmup_iterations)
                programs[f"{job.job_id}/{node.node_id}"] = program
    except BaseException as exc:
        local_error = exc
    errors: list[str | None] = [None] * world_size
    dist.all_gather_object(errors, None if local_error is None else
                           f"{type(local_error).__name__}: {local_error}")
    ended = time.perf_counter_ns() // 1000
    failures = [(endpoint, error) for endpoint, error in enumerate(errors) if error]
    if failures:
        if local_error is not None:
            raise local_error
        raise RuntimeError(f"GPU compute preparation failed on another rank: {failures}")
    return programs, started, ended


def _prepare_dag_gpu_resources(dag: DagInput, local_jobs: list[DagJob], *, device: str,
                               rank: int, world_size: int, warmup_iterations: int,
                               matmul_precision: str) -> tuple[dict[str, GpuDagResources], int, int]:
    started = time.perf_counter_ns() // 1000
    resources: dict[str, GpuDagResources] = {}
    local_error: BaseException | None = None
    try:
        for job in local_jobs:
            resources[job.job_id] = GpuDagResources(
                dag.graph, job, execution=dag.execution, seed=dag.seed, device=device, rank=rank,
                group_ranks=dag.group_ranks, warmup_iterations=warmup_iterations,
                matmul_precision=matmul_precision,
            )
    except BaseException as exc:
        local_error = exc
    errors: list[str | None] = [None] * world_size
    dist.all_gather_object(errors, None if local_error is None else
                           f"{type(local_error).__name__}: {local_error}")
    ended = time.perf_counter_ns() // 1000
    failures = [(endpoint, error) for endpoint, error in enumerate(errors) if error]
    if failures:
        if local_error is not None:
            raise local_error
        raise RuntimeError(f"GPU DAG resource preparation failed on another rank: {failures}")
    return resources, started, ended


def _validate_deferred(
    validation_records: list[tuple[dict[str, Any], torch.Tensor, Any]],
) -> tuple[int, int]:
    """Validate tensors after communication drain, outside the application path."""
    start_ts = time.perf_counter_ns() // 1000
    for task_result, tensor, expected in validation_records:
        if isinstance(expected, torch.Tensor):
            correct = bool(torch.allclose(tensor.detach().cpu(), expected, rtol=1e-4, atol=1e-4))
        else:
            correct = bool(torch.all(tensor == expected).item())
        task_result["correct"] = correct
        if not correct:
            raise AssertionError(f"incorrect all_reduce result for {task_result['task_id']}")
    return start_ts, time.perf_counter_ns() // 1000


def _finish_epoch_then_validate(runtime, validation_records, *, deadline: float):
    """Drain accepted protocol work before running any deferred tensor checks."""
    runtime.finish_epoch(_remaining(deadline))
    communication_drain_end_ts = time.perf_counter_ns() // 1000
    cpu_drain_end_s = time.process_time()
    usage_end = resource.getrusage(resource.RUSAGE_SELF)
    validation_start_ts, validation_end_ts = _validate_deferred(validation_records)
    return (communication_drain_end_ts, cpu_drain_end_s, usage_end,
            validation_start_ts, validation_end_ts)


def _run_jobs(jobs, run_one, *, runtime, deadline: float, stop_event: threading.Event,
              thread_name_prefix: str) -> list[dict[str, Any]]:
    if not jobs:
        return []
    barrier = threading.Barrier(len(jobs))
    results: list[dict[str, Any] | None] = [None] * len(jobs)
    first_error: BaseException | None = None
    error_lock = threading.Lock()

    def fail(exc: BaseException, job_id: str | None = None) -> None:
        nonlocal first_error
        with error_lock:
            is_first = first_error is None
            if is_first:
                first_error = exc
        if is_first:
            stop_event.set()
            barrier.abort()
            runtime.abort(exc, stage="job", job_id=job_id, deadline=deadline)

    def run(index: int, job) -> None:
        try:
            try:
                barrier.wait(max(0.0, deadline - time.monotonic()))
            except threading.BrokenBarrierError as exc:
                if time.monotonic() >= deadline:
                    raise TimeoutError("job barrier exceeded shared replay deadline") from exc
                raise
            if stop_event.is_set():
                raise RuntimeError("another job failed")
            results[index] = run_one(job)
        except BaseException as exc:  # noqa: BLE001
            fail(exc, getattr(job, "job_id", None))

    threads = [threading.Thread(target=run, args=(index, job),
                                name=f"{thread_name_prefix}-{job.job_id}")
               for index, job in enumerate(jobs)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(max(0.0, deadline - time.monotonic()))
    if any(thread.is_alive() for thread in threads):
        fail(TimeoutError("job runners exceeded shared replay deadline"))
    if first_error is not None:
        raise first_error
    if any(result is None for result in results):
        raise RuntimeError("job runner returned without a result")
    return [result for result in results if result is not None]


def _run_linear_job(job: Job, *, workload: Workload, args, runtime, groups, rank: int, world_size: int,
                    device: str, deadline: float, stop_event: threading.Event,
                    validation_records: list[tuple[dict[str, Any], torch.Tensor, int]],
                    precreated_bindings: dict[str, LocalBinding] | None,
                    binding_creation_events: list[dict[str, Any]], binding_events_lock: threading.Lock,
                    ) -> dict[str, Any]:
    result: dict[str, Any] = {"job_id": job.job_id, "status": "ok", "tasks": [],
                              "compute_samples_s": {},
                              "job_start_ts": time.perf_counter_ns() // 1000}
    diagnostic = getattr(args, "observation_mode", "full") != "minimal"
    comm_engine = getattr(args, "comm_engine", "new")
    if comm_engine not in {"new", "old", "bare"}:
        raise ValueError(f"unsupported communication engine {comm_engine!r}")
    for index, comm in enumerate(job.communications):
        if args.fault == "missing_task" and rank == 1 and job.job_id == "job-1" and index == len(job.communications) - 1:
            continue
        if stop_event.is_set():
            raise RuntimeError("another job failed")
        spec = task_spec(job, comm, epoch=args.epoch)
        hint = task_hint(job, index)
        declaration_mode = getattr(args, "declaration_mode", "before-producer")
        if declaration_mode == "before-producer":
            runtime.declare(spec, hint)
        elif declaration_mode != "on-submit":
            raise ValueError(f"unsupported declaration mode: {declaration_mode!r}")
        producer_start = time.perf_counter_ns() // 1000 if diagnostic else None
        if diagnostic:
            runtime.event_log.record_at("producer_start", producer_start,
                                        task_id=f"{job.job_id}/comm-{comm.id}")
        producer_s = linear_execution_duration(
            workload, args.epoch, job.job_id, comm.id, rank, "producer",
            comm.producer_compute_s, args.compute_jitter,
        )
        consumer_s = linear_execution_duration(
            workload, args.epoch, job.job_id, comm.id, rank, "consumer",
            comm.consumer_compute_s, args.compute_jitter,
        )
        result["compute_samples_s"][f"comm-{comm.id}/producer"] = producer_s
        result["compute_samples_s"][f"comm-{comm.id}/consumer"] = consumer_s
        stop_event.wait(producer_s)
        ready_ts = time.perf_counter_ns() // 1000 if diagnostic else None
        group = groups[job.job_id]
        if precreated_bindings is None:
            binding, binding_event = _make_linear_binding(
                spec, group, rank=rank, device=device, fault=args.fault,
                job_id=job.job_id, ordinal=comm.id,
            )
            with binding_events_lock:
                binding_creation_events.append(binding_event)
        else:
            try:
                binding = precreated_bindings.pop(spec.task_id)
            except KeyError as exc:
                raise RuntimeError(f"missing precreated binding for {spec.task_id}") from exc
            binding_event = (next(
                item for item in binding_creation_events if item["task_id"] == spec.task_id
            ) if diagnostic else {})
        if args.fault == "metadata_mismatch" and rank == 1 and job.job_id == "job-1" and index == 0:
            spec = replace(spec, collective=CollectiveSpec("all_reduce", spec.collective.numel + 1,
                                                           spec.collective.num_bytes + 4, "float32",
                                                           (spec.collective.numel + 1,)))
        if args.fault == "launch_failure" and rank == 0 and job.job_id == "job-0" and index == 0:
            def fail_launch():
                raise RuntimeError("injected launch failure")
            binding = replace(binding, launch=fail_launch)
        submit_call_ts = time.perf_counter_ns() // 1000 if diagnostic else None
        if diagnostic:
            runtime.event_log.record("submit_call", task_id=spec.task_id)
        handle = runtime.submit(spec, binding, hint)
        submit_return_ts = time.perf_counter_ns() // 1000 if diagnostic else None
        if diagnostic:
            runtime.event_log.record("submit_return", task_id=spec.task_id)
        consume_start = time.perf_counter_ns() // 1000 if diagnostic else None
        first_wait_ts = time.perf_counter_ns() // 1000 if diagnostic else None
        stop_event.wait(consumer_s)
        if diagnostic:
            runtime.event_log.record("application_wait_start", task_id=spec.task_id)
        if not handle.wait_host(max(0.0, deadline - time.monotonic())):
            raise TimeoutError(f"wait timed out for {spec.task_id}")
        consumer_end_ts = time.perf_counter_ns() // 1000 if diagnostic else None
        if diagnostic:
            runtime.event_log.record("application_wait_return", task_id=spec.task_id)
        expected = sum(item + 1 for item in ranks_for_job(job, world_size))
        task_result = {
            "task_id": spec.task_id, "job_id": job.job_id, "ordinal": comm.id,
            "correct": None, "decision_seq": handle.decision_seq,
            "group_id": spec.group_id, "group_seq": spec.group_seq,
        }
        if diagnostic:
            task_result.update({
                "producer_start_ts": producer_start, "ready_ts": ready_ts,
                "collective_spec": spec.collective.to_dict(), "task_hint": hint.to_dict(),
                **binding_event,
                "submit_call_ts": submit_call_ts, "submit_return_ts": submit_return_ts,
                "consumer_start_ts": consume_start, "first_wait_ts": first_wait_ts,
                "consumer_end_ts": consumer_end_ts,
            })
        result["tasks"].append(task_result)
        validation_records.append((task_result, binding.tensor, expected))
    result["job_end_ts"] = time.perf_counter_ns() // 1000
    return result


def _run_dag_job(job: DagJob, *, dag: DagInput, args, runtime, groups, rank: int,
                 device: str, deadline: float, stop_event: threading.Event,
                 dag_events: EventLog, group_ranks: dict[str, tuple[int, ...]],
                 missing_id: str, tails: dict[str, float],
                 validation_records: list[tuple[dict[str, Any], torch.Tensor, Any]],
                 gpu_compute_programs: dict[str, Any] | None = None,
                 gpu_dag_resources: Mapping[str, GpuDagResources] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"job_id": job.job_id, "status": "ok", "tasks": [],
                              "job_start_ts": time.perf_counter_ns() // 1000}
    comm_nodes = [node for node in job.nodes if isinstance(node, CommNode)]
    runner_runtime = runtime
    if args.fault == "missing_task" and rank == 1 and missing_id in {
        f"{job.job_id}/{node.node_id}" for node in comm_nodes
    }:
        runner_runtime = _DropOneSubmit(runtime, missing_id)

    node_positions = {node.node_id: index for index, node in enumerate(comm_nodes)}
    def make_binding(comm: CommNode) -> LocalBinding:
        comm_index = node_positions[comm.node_id]
        if args.fault == "binding_failure" and rank == 0 and comm_index == 0:
            raise RuntimeError("injected DAG binding failure")
        resources = (gpu_dag_resources or {}).get(job.job_id)
        if resources is not None:
            binding = resources.make_binding(comm, groups[comm.group_id])
        else:
            binding = make_collective_binding(dag_task_spec(job, comm, epoch=args.epoch),
                                              groups[comm.group_id], rank=rank, device=device)
        if args.fault == "launch_failure" and rank == 0 and job.job_id == "job-0" and comm_index == 0:
            def fail_launch():
                raise RuntimeError("injected launch failure")
            binding = replace(binding, launch=fail_launch)
        return binding

    samples: dict[str, float] = {}
    job_gpu_programs = {
        node.node_id: gpu_compute_programs[f"{job.job_id}/{node.node_id}"]
        for node in job.nodes
        if gpu_compute_programs is not None and f"{job.job_id}/{node.node_id}" in gpu_compute_programs
    }
    resources = (gpu_dag_resources or {}).get(job.job_id)
    if resources is not None:
        job_gpu_programs.update(resources.node_programs)
    compute = make_replay_compute(job, dag.execution, seed=dag.seed, epoch=args.epoch,
                                  rank=rank, jitter=args.compute_jitter, samples=samples,
                                  event_log=dag_events, gpu_programs=job_gpu_programs)
    first_compute = next((node.node_id for node in job.nodes if isinstance(node, ComputeNode)), None)
    def run_compute(node, cancellation):
        if args.fault == "compute_failure" and rank == 0 and node.node_id == first_compute:
            raise RuntimeError("injected DAG compute failure")
        return compute(node, cancellation)

    runner = DagRunner(dag.graph, job, runner_runtime, epoch=args.epoch, rank=rank,
                       compute_fn=run_compute, make_binding=make_binding,
                       deadline=deadline, poll_interval=args.dag_poll_interval, tails=tails,
                       submit_after={
                           compute_id.split("/", 1)[1]: tuple(comm_id.split("/", 1)[1]
                                                             for comm_id in comm_ids)
                           for compute_id, comm_ids in dag.execution.submit_after.items()
                           if compute_id.startswith(job.job_id + "/")
                       },
                       enable_lookahead=args.policy == "lookahead",
                       stop_event=stop_event, event_log=dag_events)
    result.update(runner.run())
    result["compute_samples_s"] = samples
    expected = sum(rank_id + 1 for rank_id in group_ranks[comm_nodes[0].group_id])
    for node in job.nodes:
        if not isinstance(node, CommNode):
            continue
        binding = runner.bindings[node.node_id]
        spec = dag_task_spec(job, node, epoch=args.epoch)
        task_result = {"task_id": spec.task_id, "node_id": node.node_id,
                       "job_id": job.job_id, "group_id": spec.group_id,
                       "group_seq": spec.group_seq, "decision_seq": runner.handles[node.node_id].decision_seq,
                       "correct": None}
        result["tasks"].append(task_result)
        expected_result = (resources.expected_comm_by_node[node.node_id]
                           if resources is not None else expected)
        validation_records.append((task_result, binding.tensor, expected_result))
    result["gpu_resource_job_id"] = job.job_id if resources is not None else None
    result["completed_node_ids"] = [f"{job.job_id}/{node_id}" for node_id in runner.completed_node_ids]
    result["job_end_ts"] = result["physical_completion_observed_ts"]
    return result


def run_rank(args: argparse.Namespace) -> dict[str, Any]:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    comm_engine = getattr(args, "comm_engine", "new")
    if comm_engine not in {"new", "old", "bare"}:
        raise ValueError(f"unsupported communication engine {comm_engine!r}")
    if args.timeout <= 0 or args.setup_timeout <= 0 or args.poll_interval <= 0 or args.dag_poll_interval <= 0:
        raise ValueError("setup/replay timeouts and poll intervals must be positive")
    compute_mode = getattr(args, "compute_mode", "host-sleep")
    compute_matrix_size = getattr(args, "compute_matrix_size", 256)
    compute_repeats = getattr(args, "compute_repeats", 1)
    binding_preparation = getattr(args, "binding_preparation", "on-ready")
    if compute_mode not in {"host-sleep", "cuda-matmul"}:
        raise ValueError(f"unsupported compute mode: {compute_mode!r}")
    if compute_matrix_size < 16 or compute_repeats <= 0:
        raise ValueError("compute matrix size must be >= 16 and repeats must be positive")
    if compute_mode == "cuda-matmul" and args.backend != "nccl":
        raise ValueError("cuda-matmul compute requires NCCL")
    if args.backend == "nccl" and args.dag is None:
        raise ValueError("GPU/NCCL Phase 3 replay accepts DAG inputs only; linear workloads require Gloo")
    if compute_mode == "cuda-matmul" and args.dag is None:
        raise ValueError("cuda-matmul is supported only for DAG inputs")
    if not 0 <= args.compute_jitter < 1:
        raise ValueError("compute_jitter must be in [0, 1)")
    comm_profile = getattr(args, "comm_profile", None)
    compute_profile_path = getattr(args, "compute_profile", None)
    profile_strict = getattr(args, "profile_strict", True)
    wait_budget_s = getattr(args, "wait_budget_s", 0.02)
    if wait_budget_s < 0:
        raise ValueError("wait_budget_s must be non-negative")
    setup_deadline = time.monotonic() + args.setup_timeout
    replay_deadline = setup_deadline
    dag = load_dag(args.dag, epoch=args.epoch, world_size=world_size) if args.dag else None
    gpu_program_dag = bool(dag is not None and dag.execution.schema_version == 2
                           and dag.execution.mode == "cuda-program")
    if gpu_program_dag and args.backend != "nccl":
        raise ValueError("DAG schema-v2 cuda-program execution requires --backend nccl")
    if comm_engine != "new" and (not gpu_program_dag or args.backend != "nccl"):
        raise ValueError("old and bare adapters require a schema-v2 CUDA DAG on NCCL")
    if comm_engine == "old" and args.policy not in {"static_fifo", "static_ltf"}:
        raise ValueError("old scheduler adapter requires static_fifo or static_ltf")
    if comm_engine == "bare" and args.policy != "bare":
        raise ValueError("bare adapter uses DAG readiness and requires the bare policy marker")
    if comm_engine == "bare":
        from examples.jobpacer.runtime.dag_comm_adapters import validate_bare_backend
        validate_bare_backend(nccl_version=tuple(torch.cuda.nccl.version()), cuda_version=torch.version.cuda,
                              implicit=os.environ.get("NCCL_LAUNCH_ORDER_IMPLICIT"),
                              blocking_wait=os.environ.get("TORCH_NCCL_BLOCKING_WAIT"))
    if comm_engine == "bare" and world_size != 2:
        raise ValueError("bare-ordered-v2-layered-round-robin is validated only for a two-rank NCCL world")
    if comm_engine == "bare" and os.environ.get("NCCL_LAUNCH_ORDER_IMPLICIT", "").strip() != "1":
        raise ValueError("bare requires NCCL_LAUNCH_ORDER_IMPLICIT=1 for ordered multi-communicator NCCL issue")
    workload = None if dag else load_workload(args.workload or "balanced")
    profile = None
    if comm_profile:
        profile = load_profile(comm_profile)
    if workload is not None and profile is not None:
        workload = apply_profile(
            workload,
            profile,
            {"backend": args.backend, "device_type": "cuda" if args.backend == "nccl" else "cpu",
             "world_size": world_size},
            strict=profile_strict,
        )
    if dag is not None and profile is not None:
        dag = apply_dag_profile(
            dag, profile,
            {"backend": args.backend, "device_type": "cuda" if args.backend == "nccl" else "cpu",
             "world_size": world_size},
            strict=profile_strict,
        )
    compute_profile = None
    if compute_profile_path:
        if dag is None or dag.execution.schema_version != 2:
            raise ValueError("--compute-profile requires a schema-v2 GPU DAG")
        from examples.jobpacer.gpu.gpu_compute_profile import load_gpu_compute_profile
        compute_profile = load_gpu_compute_profile(compute_profile_path)
        dag = apply_dag_compute_profile(
            dag, compute_profile, device_uuids=visible_cuda_uuids(world_size),
            software=compute_profile_software(getattr(args, "matmul_precision", "highest")),
            strict=profile_strict,
        )
    elif dag is not None and dag.execution.schema_version == 2 and args.policy in {"static_ltf", "ltf"}:
        raise ValueError("schema-v2 static_ltf/ltf requires --compute-profile")
    dag_tails = compute_tails(dag.graph) if dag is not None else None
    if args.static_order and (dag is None or args.policy not in {"static_fifo", "static_ltf", "bare"}):
        raise ValueError("--static-order is only valid with a DAG and static policy")
    if dag is not None:
        order = (load_static_order(args.static_order, dag.graph, extra_predecessors=dag.execution.submit_after) if args.static_order
                 else build_static_order(
                     dag.graph, args.policy, tails=dag_tails,
                     extra_predecessors=(dag.execution.submit_after
                                         if args.policy == "static_fifo" else None),
                 )
                 if args.policy in {"static_fifo", "static_ltf"} else ())
        bare_order = None
        if comm_engine == "bare":
            from examples.jobpacer.runtime.dag_comm_adapters import build_bare_order
            bare_order = build_bare_order(
                dag.graph, submit_after=dag.execution.submit_after, input_hash=dag.input_hash,
            )
            if args.static_order and tuple(order) != bare_order.sequence:
                raise ValueError("frozen bare order differs from the versioned default sequence")
    else:
        order = ()
        bare_order = None
    device = _free_device(args.backend)
    inputs = dag if dag is not None else workload
    from runtime_comm_scheduler.runtime.coordinator import CoordinatorState
    from runtime_comm_scheduler.runtime.transport import CoordinatorServer
    server = None
    runtime = None
    failure_signal = None
    groups: dict[str, Any] = {}
    gpu_compute_programs: dict[str, CudaMatmulProgram] = {}
    gpu_dag_resources: dict[str, GpuDagResources] = {}
    gpu_compute_preparation_start_ts = gpu_compute_preparation_end_ts = None
    try:
        process_group_timeout_s = (
            min(20.0, args.timeout) if args.backend == "nccl"
            else _remaining(setup_deadline)
        )
        init_kwargs = {
            "init_method": f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
            "rank": rank,
            "world_size": world_size,
            "timeout": timedelta(seconds=min(_remaining(setup_deadline), process_group_timeout_s)),
        }
        if args.backend == "nccl":
            init_kwargs["device_id"] = torch.device(device)
        dist.init_process_group(args.backend, **init_kwargs)
        if comm_engine != "new":
            from examples.jobpacer.runtime.dag_comm_adapters import EpochFailureSignal
            failure_signal = EpochFailureSignal(args.epoch, rank, world_size)
        if comm_engine == "bare":
            assert dag is not None and bare_order is not None and failure_signal is not None
            failure_signal.verify_common_contract(
                "bare-launch-order-v1",
                {
                    "input_hash": dag.input_hash,
                    "manifest_digest": dag.manifest_digest,
                    "estimate_view_hash": dag.estimate_view_hash,
                    "order_digest": bare_order.digest,
                    "order": list(bare_order.sequence),
                },
                setup_deadline,
            )
        groups = _new_groups(inputs, world_size, setup_deadline,
                             group_timeout_s=process_group_timeout_s)
        local_jobs = ([job for job in workload.jobs if rank in ranks_for_job(job, world_size)]
                      if dag is None else _local_dag_jobs(dag, rank))
        if gpu_program_dag:
            (gpu_dag_resources, gpu_compute_preparation_start_ts,
             gpu_compute_preparation_end_ts) = _prepare_dag_gpu_resources(
                dag, local_jobs, device=device, rank=rank, world_size=world_size,
                warmup_iterations=getattr(args, "warmup_iterations", 0),
                matmul_precision=getattr(args, "matmul_precision", "highest"),
            )
        elif dag is not None and compute_mode == "cuda-matmul":
            (gpu_compute_programs, gpu_compute_preparation_start_ts,
             gpu_compute_preparation_end_ts) = _prepare_dag_gpu_compute(
                dag, local_jobs, device=device, rank=rank, world_size=world_size,
                matrix_size=compute_matrix_size, repeats=compute_repeats, epoch=args.epoch,
                warmup_iterations=getattr(args, "warmup_iterations", 0))
        precreated_bindings: dict[str, LocalBinding] | None = None
        binding_creation_events: list[dict[str, Any]] = []
        preparation_start_ts = preparation_end_ts = None
        binding_preparation = getattr(args, "binding_preparation", "on-ready")
        if dag is None and binding_preparation == "precreate":
            (precreated_bindings, binding_creation_events,
             preparation_start_ts, preparation_end_ts) = _prepare_linear_bindings(
                workload, local_jobs, groups=groups, rank=rank, device=device,
                world_size=world_size, epoch=args.epoch, fault=args.fault,
            )
        dist.barrier()
        warmup_collective_count = _warmup_collectives(
            inputs, groups, rank=rank, world_size=world_size, device=device,
            iterations=getattr(args, "warmup_iterations", 0),
        )

        runtime_policy = "static" if args.policy in {"static_fifo", "static_ltf"} else args.policy
        if dag is None and runtime_policy == "static":
            order = linear_static_order(workload, args.policy)
        if comm_engine == "new" and rank == 0:
            # The coordinator starts its watchdog at the first group event, before local setup ends.
            coordinator = CoordinatorState(tuple(range(world_size)), epoch=args.epoch, policy=runtime_policy,
                                          static_order=order,
                                          wait_budget_s=wait_budget_s,
                                          epoch_timeout_s=args.setup_timeout + args.timeout,
                                          observation_mode=getattr(args, "observation_mode", "full"))
            server = CoordinatorServer(coordinator, "127.0.0.1", args.control_port)
            server.start()

        runtime_event_log = EventLog(
            "runtime", rank,
            enabled=getattr(args, "observation_mode", "full") != "minimal",
            thread_sharded=True,
        )
        if comm_engine == "new":
            client = ControlClient(rank, args.epoch, "127.0.0.1", args.control_port,
                                   _remaining(setup_deadline),
                                   getattr(args, "observation_mode", "full") != "minimal",
                                   runtime_event_log)
            probe = _FailingProbe() if args.fault == "completion_probe_failure" and rank == 0 else None
            executor = (CudaCollectiveExecutor(device) if args.backend == "nccl" else DirectExecutor())
            runtime = RankRuntime(rank, args.epoch, client, executor=executor,
                                  completion_poll_interval_s=args.poll_interval,
                                  wake_completion_on_submit=getattr(
                                      args, "wake_completion_on_submit", False),
                                  event_log=runtime_event_log,
                                  completion_probe=probe)
        elif comm_engine == "old":
            from examples.jobpacer.runtime.dag_comm_adapters import LegacySchedulerDagAdapter
            used_groups = {group_id for group_id, members in dag.group_ranks.items() if rank in members}
            runtime = LegacySchedulerDagAdapter(
                dag.graph, policy=args.policy, epoch=args.epoch, rank=rank, group_ids=used_groups,
                tails=dag_tails, device=device, poll_interval_s=args.poll_interval,
                event_log=runtime_event_log, order=order, failure_signal=failure_signal,
                failure_drain_timeout_s=args.timeout,
            )
            executor = runtime.executor
        else:
            from examples.jobpacer.runtime.dag_comm_adapters import BareDagAdapter
            bare_probe = (_FailingProbe() if args.fault == "completion_probe_failure" and rank == 0
                          else None)
            runtime = BareDagAdapter(
                dag.graph, rank=rank, group_ranks=dag.group_ranks, device=device,
                order=bare_order, submit_after=dag.execution.submit_after,
                deadline_s=args.timeout, poll_interval_s=args.poll_interval,
                event_log=runtime_event_log, failure_signal=failure_signal,
                completion_probe=bare_probe, failure_drain_timeout_s=args.timeout,
            )
            executor = runtime.executor
        if dag is not None:
            group_ranks = dag.group_ranks
            used_groups = {node.group_id for job in dag.graph.jobs for node in job.nodes
                           if isinstance(node, CommNode) and rank in group_ranks[node.group_id]}
            group_specs = {group.group_id: group for group in dag.graph.groups}
            for group_id in sorted(used_groups):
                runtime.register_group(group_specs[group_id], groups[group_id])
        else:
            for job in workload.jobs:
                if rank in ranks_for_job(job, world_size):
                    runtime.register_group(group_spec(job, world_size, epoch=args.epoch), groups[job.job_id])
        runtime.start(_remaining(setup_deadline))

        observation_mode = getattr(args, "observation_mode", "full")
        dag_events = EventLog("dag", rank, enabled=observation_mode != "minimal",
                              thread_sharded=True)
        stop_event = threading.Event()
        # This barrier is the common application release boundary.  It is the
        # default process group, never a job collective controlled by the
        # coordinator.
        if server is not None:
            server.wait_ready(_remaining(setup_deadline))
        dist.barrier()
        cpu_start_s = time.process_time()
        usage_start = resource.getrusage(resource.RUSAGE_SELF)
        application_release_ts = time.perf_counter_ns() // 1000
        deadline = time.monotonic() + args.timeout
        replay_deadline = deadline
        set_replay_deadline = getattr(runtime, "set_replay_deadline", None)
        if callable(set_replay_deadline):
            set_replay_deadline(deadline)
        validation_records: list[tuple[dict[str, Any], torch.Tensor, Any]] = []
        if dag is not None:
            local_jobs = _local_dag_jobs(dag, rank)
            group_ranks = dag.group_ranks
            all_comms = [(job, node) for job in dag.graph.jobs for node in job.nodes
                         if isinstance(node, CommNode)]
            missing_id = f"{all_comms[0][0].job_id}/{all_comms[0][1].node_id}"
            run_one = lambda job: _run_dag_job(
                job, dag=dag, args=args, runtime=runtime, groups=groups, rank=rank,
                device=device, deadline=deadline, stop_event=stop_event,
                dag_events=dag_events, group_ranks=group_ranks, missing_id=missing_id,
                tails=dag_tails, validation_records=validation_records,
                gpu_compute_programs=gpu_compute_programs,
                gpu_dag_resources=gpu_dag_resources,
            )
            jobs = _run_jobs(local_jobs, run_one, runtime=runtime, deadline=deadline,
                             stop_event=stop_event, thread_name_prefix="dag")
        else:
            binding_events_lock = threading.Lock()
            run_one = lambda job: _run_linear_job(
                job, workload=workload, args=args, runtime=runtime, groups=groups, rank=rank, world_size=world_size,
                device=device, deadline=deadline, stop_event=stop_event,
                validation_records=validation_records, precreated_bindings=precreated_bindings,
                binding_creation_events=binding_creation_events,
                binding_events_lock=binding_events_lock,
            )
            jobs = _run_jobs(local_jobs, run_one, runtime=runtime, deadline=deadline,
                             stop_event=stop_event, thread_name_prefix="job")
        application_end_ts = max(job["job_end_ts"] for job in jobs) if dag is not None else time.perf_counter_ns() // 1000
        (communication_drain_end_ts, cpu_drain_end_s, usage_end,
         validation_start_ts, validation_end_ts) = _finish_epoch_then_validate(
            runtime, validation_records, deadline=deadline)
        gpu_dag_validation: dict[str, dict[str, bool]] = {}
        if gpu_program_dag:
            for job_id, resources in gpu_dag_resources.items():
                checks = resources.validate()
                gpu_dag_validation[job_id] = checks
                job_result = next(item for item in jobs if item["job_id"] == job_id)
                job_result["gpu_buffer_validation"] = checks
                if not all(checks.values()):
                    raise AssertionError(f"GPU DAG numerical validation failed for {job_id}: {checks}")
            validation_end_ts = time.perf_counter_ns() // 1000
        harness_start_ts = time.perf_counter_ns() // 1000
        output = {
            "rank": rank, "status": "ok", "mode": "runtime",
            "comm_engine": comm_engine,
            "communication_adapter": comm_engine,
            "observation_mode": observation_mode,
            "input_mode": "dag" if dag is not None else "linear",
            "policy": args.policy, "backend": args.backend, "jobs": jobs,
            "device": str(torch.cuda.current_device()) if args.backend == "nccl" else "cpu",
            "device_name": torch.cuda.get_device_name() if args.backend == "nccl" else None,
            "device_uuid": (str(getattr(torch.cuda.get_device_properties(torch.cuda.current_device()),
                                       "uuid", "")) or None) if args.backend == "nccl" else None,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_version": torch.__version__,
            "torch_git_version": torch.version.git_version,
            "cuda_version": torch.version.cuda,
            "nccl_version": torch.cuda.nccl.version() if args.backend == "nccl" else None,
            "executor_type": type(executor).__name__,
            "completion_probe_type": type(getattr(runtime, "completion_probe", None)).__name__,
            "completion_source": ("cuda_event_query_after_backend_work_wait" if args.backend == "nccl"
                                  else "work_is_completed"),
            "nccl_launch_order_implicit": (
                os.environ.get("NCCL_LAUNCH_ORDER_IMPLICIT") if comm_engine == "bare" else None),
            "offer_readiness_mode": "physical_ready",
            "compute_mode": compute_mode,
            "compute_matrix_size": compute_matrix_size if compute_mode == "cuda-matmul" else None,
            "compute_repeats": compute_repeats if compute_mode == "cuda-matmul" else None,
            "gpu_compute_program_count": len(gpu_compute_programs)
            + sum(len(item.node_programs) for item in gpu_dag_resources.values()),
            "gpu_compute_preparation_start_ts": gpu_compute_preparation_start_ts,
            "gpu_compute_preparation_end_ts": gpu_compute_preparation_end_ts,
            "gpu_compute_preparation_total_us": (
                gpu_compute_preparation_end_ts - gpu_compute_preparation_start_ts
                if gpu_compute_preparation_start_ts is not None else None),
            "application_end_definition": (
                "all local DAG nodes have physical completion evidence" if dag is not None else
                "all local host waits returned"),
            "epoch": args.epoch, "world_size": world_size,
            "max_inflight": None if comm_engine == "bare" else 1,
            "process_group_timeout_s": (min(20.0, args.timeout)
                                        if args.backend == "nccl" else None),
            "completion_poll_interval_s": args.poll_interval,
            "wake_completion_on_submit": getattr(args, "wake_completion_on_submit", False),
            "dag_poll_interval_s": args.dag_poll_interval,
            "compute_jitter": args.compute_jitter,
            "warmup_iterations": getattr(args, "warmup_iterations", 0),
            "warmup_collective_count": warmup_collective_count,
            "estimate_source": "offline_profile" if profile else "manifest",
            "compute_profile": ({"path": str(compute_profile_path),
                                 "digest": compute_profile.digest,
                                 "schema_version": compute_profile.raw.get("schema_version"),
                                 "estimator_version": "gpu-dag-profile-v1"}
                                if compute_profile is not None else None),
            "communication_profile": {
                "path": str(comm_profile) if comm_profile else None,
                "digest": profile.digest() if profile else None,
                "schema_version": profile.schema_version if profile else None,
                "strict": profile_strict if profile else None,
                "environment": dict(profile.environment) if profile else None,
            },
            "wait_budget_s": wait_budget_s,
            "binding_preparation": ("unchanged_dag" if dag is not None else
                                    binding_preparation),
            "declaration_mode": ("unchanged_dag" if dag is not None else
                                 getattr(args, "declaration_mode", "before-producer")),
            "preparation_start_ts": preparation_start_ts,
            "preparation_end_ts": preparation_end_ts,
            "preparation_total_us": (preparation_end_ts - preparation_start_ts
                                     if preparation_start_ts is not None else None),
            "preparation_scope": (
                "binding construction and cross-rank preparation status agreement"
                if preparation_start_ts is not None else
                "no separate pre-release linear binding stage; per-task creation is recorded"
                if dag is None else "DAG preparation path unchanged"
            ),
            "binding_creation_total_us": sum(item["binding_create_duration_us"]
                                              for item in binding_creation_events),
            "binding_creation_events": binding_creation_events,
            "application_release_ts": application_release_ts,
            "application_end_ts": application_end_ts,
            "communication_drain_end_ts": communication_drain_end_ts,
            "dependency_mode": "physical-completion" if dag is not None else None,
            "compute_model": dag.execution.compute_model if gpu_program_dag else None,
            "execution_mode": dag.execution.mode if dag is not None else None,
            "execution_contract": ({
                "dependency_mode": "physical-completion",
                "compute_model": dag.execution.compute_model,
                "max_inflight": None if comm_engine == "bare" else 1,
                "communication_capacity": "not_admission_limited" if comm_engine == "bare" else 1,
                "communication_order": ("layered_topological_job_round_robin_projection"
                                         if comm_engine == "bare"
                                         else None),
                "communication_contract_version": (bare_order.contract_version
                                                    if comm_engine == "bare" else None),
                "communication_order_digest": (bare_order.digest if comm_engine == "bare" else None),
                "process_group_timeout_s": process_group_timeout_s,
                "seed_derivation_version": "sha256-torch-generator-v1",
            } if gpu_program_dag else None),
            "input_hash": dag.input_hash if dag is not None else None,
            "estimate_view_hash": dag.estimate_view_hash if dag is not None else None,
            "tensor_seed_derivation_version": "sha256-torch-generator-v1" if gpu_program_dag else None,
            "gpu_dag_validation": gpu_dag_validation if gpu_program_dag else None,
            "validation_start_ts": validation_start_ts,
            "validation_end_ts": validation_end_ts,
            "harness_start_ts": harness_start_ts,
            "harness_end_ts": None,
            "application_makespan_us": application_end_ts - application_release_ts,
            "communication_drain_makespan_us": communication_drain_end_ts - application_release_ts,
            "process_cpu_time_s": cpu_drain_end_s - cpu_start_s,
            "voluntary_context_switches": usage_end.ru_nvcsw - usage_start.ru_nvcsw,
            "involuntary_context_switches": usage_end.ru_nivcsw - usage_start.ru_nivcsw,
            "validation_total_us": validation_end_ts - validation_start_ts,
            "task_sequence": [task["task_id"] for job in jobs for task in job["tasks"]],
            "grant_sequence": runtime.grant_order, "launch_sequence": runtime.launch_order,
            "protocol_send_call_counts": {
                "declare": (sum(len(job["tasks"]) for job in jobs)
                            if dag is None and getattr(args, "declaration_mode", "before-producer")
                            == "before-producer" else 0),
                "offer": (sum(len(job["tasks"]) for job in jobs) if comm_engine == "new" else 0),
            },
            "runtime_events": runtime.event_log.as_dict()["events"],
            "protocol_transition_counts": ({
                "submitted_reports": getattr(server.coordinator, "submitted_report_count", None),
                "completed_reports": getattr(server.coordinator, "completed_report_count", None),
                "submitted_before_completed_enforced": True,
            } if server is not None and rank == 0 else None),
            "coordinator_instrumentation": (
                list(getattr(server.coordinator, "instrumentation_records", []))
                if server is not None and rank == 0 else []
            ),
        }
        output["linear_ltf_estimates"] = linear_ltf_estimates(workload) if workload is not None else []
        output["linear_static_ltf_order"] = (
            list(linear_static_order(workload, "static_ltf")) if workload is not None else []
        )
        if dag is not None:
            local_jobs = _local_dag_jobs(dag, rank)
            task_groups = {f"{job.job_id}/{node.node_id}": node.group_id
                           for job in dag.graph.jobs for node in job.nodes if isinstance(node, CommNode)}
            output.update({
                "dag_name": dag.name,
                "dag_seed": dag.seed,
                "manifest_digest": dag.manifest_digest,
                "canonical_dag": json.loads(dag.canonical_json),
                "dag_tails_s": {task_id: dag_tails[task_id] for task_id in dag.expected_task_ids},
                "groups": [{"group_id": group.group_id, "ranks": list(group.ranks)}
                           for group in dag.graph.groups],
                "expected_task_ids": [task_id for task_id in dag.expected_task_ids
                                      if rank in dag.group_ranks[task_groups[task_id]]],
                "expected_node_ids": [f"{job.job_id}/{node.node_id}" for job in local_jobs for node in job.nodes],
                "dag_events": dag_events.as_dict()["events"],
            })
            if comm_engine == "old":
                output["shared_plan_digest"] = runtime.plan.digest()
                output["shared_plan_sequence"] = list(runtime.launch_order)
                output["plan_origin"] = "gpu-shared-estimator"
            elif comm_engine == "bare":
                output["bare_launch_sequence"] = list(runtime.launch_order)
                output["bare_contract"] = bare_order.contract_version
                output["bare_default_order"] = list(bare_order.sequence)
                output["bare_job_tie_order"] = list(bare_order.job_order)
                output["bare_default_order_digest"] = bare_order.digest
                output["bare_input_hash"] = bare_order.input_hash
                output["bare_local_order"] = list(runtime.local_order)
                output["bare_peak_inflight"] = runtime.peak_inflight
        if server is not None:
            output["decision_records"] = list(server.coordinator.records)
            output["mechanism_counts"] = dict(server.coordinator.mechanism_counts)
        output["harness_end_ts"] = time.perf_counter_ns() // 1000
        output["harness_total_us"] = output["harness_end_ts"] - harness_start_ts
        return output
    finally:
        try:
            if runtime is not None:
                runtime.close()
        finally:
            try:
                if server is not None:
                    server.close()
            finally:
                peer_or_local_failure = (failure_signal.failure()
                                         if failure_signal is not None else None)
                if peer_or_local_failure is not None:
                    # The default TCPStore is hosted by rank 0. A failing
                    # rank 0 must keep it alive until peers have read the
                    # first failure and acknowledged their own teardown.
                    if comm_engine == "bare" and failure_signal is not None:
                        try:
                            failure_signal.publish_teardown_ready()
                            failure_signal.wait_for_teardown_ready(time.monotonic() + 2.0)
                        except BaseException:
                            pass
                    # NCCL has no safe cancellation for a host launch blocked
                    # without its peer. Exit this rank process and let the
                    # parent harness bound sibling cleanup instead of entering
                    # a communicator destroy path that can wait for watchdog
                    # timeout. Only this replay child is terminated here.
                    sys.stderr.write(f"rank {rank} fail-stop after collective error: "
                                     f"{peer_or_local_failure}\n")
                    sys.stderr.flush()
                    os._exit(1)
                for group in reversed(list(groups.values())):
                    try:
                        dist.destroy_process_group(group)
                    except Exception:
                        pass
                if dist.is_initialized():
                    dist.destroy_process_group()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=("static_fifo", "static_ltf", "fifo", "ltf", "lookahead", "bare"), default="fifo")
    parser.add_argument("--comm-engine", choices=("new", "old", "bare"), default="new")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--workload")
    source.add_argument("--dag", type=Path)
    parser.add_argument("--static-order", type=Path)
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--setup-timeout", type=float, default=20.0)
    parser.add_argument("--poll-interval", type=float, default=0.001)
    parser.add_argument("--wake-completion-on-submit", action="store_true",
                        help="wake the periodic completion probe when new work becomes probeable")
    parser.add_argument("--dag-poll-interval", type=float, default=0.001)
    parser.add_argument("--compute-jitter", type=float, default=0.0)
    parser.add_argument("--compute-mode", choices=("host-sleep", "cuda-matmul"), default="host-sleep")
    parser.add_argument("--compute-matrix-size", type=int, default=256)
    parser.add_argument("--compute-repeats", type=int, default=1)
    parser.add_argument("--matmul-precision", choices=("highest", "high", "medium"), default="highest")
    parser.add_argument("--wait-budget-s", type=float, default=0.02)
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--binding-preparation", choices=("precreate", "on-ready"), default="precreate")
    parser.add_argument("--declaration-mode", choices=("before-producer", "on-submit"),
                        default="before-producer")
    parser.add_argument("--observation-mode", choices=("minimal", "diagnostic", "full"), default="full")
    parser.add_argument("--comm-profile", type=Path)
    parser.add_argument("--compute-profile", type=Path)
    parser.add_argument("--profile-strict", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--control-port", type=int, required=True)
    parser.add_argument("--cuda-profiler-dir", type=Path)
    parser.add_argument("--fault", choices=("none", "missing_task", "metadata_mismatch", "launch_failure",
                                               "completion_probe_failure", "compute_failure", "binding_failure"), default="none")
    args = parser.parse_args()
    if args.timeout <= 0 or args.setup_timeout <= 0 or args.wait_budget_s < 0 or args.warmup_iterations < 0:
        parser.error("setup-timeout and timeout must be positive; wait-budget-s must be non-negative")
    if args.declaration_mode == "on-submit" and (args.dag or args.policy == "lookahead"):
        parser.error("--declaration-mode on-submit supports Gloo linear non-Lookahead replay only")
    try:
        if args.cuda_profiler_dir:
            args.cuda_profiler_dir.mkdir(parents=True, exist_ok=True)
            rank = int(os.environ.get("RANK", "0"))
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=False,
                with_stack=False,
            ) as profiler:
                output = run_rank(args)
            trace_path = args.cuda_profiler_dir / f"rank{rank}.json"
            profiler.export_chrome_trace(str(trace_path))
            output["cuda_profiler_trace"] = str(trace_path)
        else:
            output = run_rank(args)
    except BaseException as exc:  # noqa: BLE001
        output = {"rank": int(os.environ.get("RANK", -1)), "status": "error",
                  "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(output, sort_keys=True), flush=True)
        return 1
    print(json.dumps(output, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
