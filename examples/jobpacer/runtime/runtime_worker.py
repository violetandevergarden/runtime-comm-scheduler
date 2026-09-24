"""One-rank JobPacer linear or validated-DAG replay worker."""

from __future__ import annotations

import argparse
import json
import os
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
from runtime_comm_scheduler.runtime import CollectiveSpec, DirectExecutor, EventLog, LocalBinding, RankRuntime
from runtime_comm_scheduler.runtime.transport import ControlClient
from examples.jobpacer.runtime.runtime_adapter import (
    DagInput,
    apply_dag_profile,
    group_spec,
    linear_static_order,
    load_dag,
    load_static_order,
    make_collective_binding,
    make_replay_compute,
    task_hint,
    task_spec,
)
from examples.jobpacer.workloads import Job, Workload, linear_execution_duration, load_workload, ranks_for_job
from examples.jobpacer.comm_profile import apply_profile, load_profile


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


def _free_device(backend: str) -> str:
    if backend == "nccl":
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        return "cuda"
    return "cpu"


def _new_groups(workload: Workload | DagInput, world_size: int,
                setup_deadline: float | None = None) -> dict[str, Any]:
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
            timeout = {} if remaining is None else {"timeout": timedelta(seconds=remaining)}
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


def _validate_deferred(
    validation_records: list[tuple[dict[str, Any], torch.Tensor, int]],
) -> tuple[int, int]:
    """Validate tensors after communication drain, outside the application path."""
    start_ts = time.perf_counter_ns() // 1000
    for task_result, tensor, expected in validation_records:
        correct = bool(torch.all(tensor == expected).item())
        task_result["correct"] = correct
        if not correct:
            raise AssertionError(f"incorrect all_reduce result for {task_result['task_id']}")
    return start_ts, time.perf_counter_ns() // 1000


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
            runtime.abort(exc, stage="job", job_id=job_id)

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
                    validation_records: list[tuple[dict[str, Any], torch.Tensor, int]]) -> dict[str, Any]:
    result: dict[str, Any] = {"job_id": job.job_id, "status": "ok", "tasks": [],
                              "compute_samples_s": {},
                              "job_start_ts": time.perf_counter_ns() // 1000}
    for index, comm in enumerate(job.communications):
        if args.fault == "missing_task" and rank == 1 and job.job_id == "job-1" and index == len(job.communications) - 1:
            continue
        if stop_event.is_set():
            raise RuntimeError("another job failed")
        spec = task_spec(job, comm, epoch=args.epoch)
        hint = task_hint(job, index)
        runtime.declare(spec, hint)
        producer_start = time.perf_counter_ns() // 1000
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
        ready_ts = time.perf_counter_ns() // 1000
        group = groups[job.job_id]
        binding = make_collective_binding(spec, group, rank=rank, device=device)
        if args.fault == "metadata_mismatch" and rank == 1 and job.job_id == "job-1" and index == 0:
            spec = replace(spec, collective=CollectiveSpec("all_reduce", spec.collective.numel + 1,
                                                           spec.collective.num_bytes + 4, "float32",
                                                           (spec.collective.numel + 1,)))
        if args.fault == "launch_failure" and rank == 0 and job.job_id == "job-0" and index == 0:
            def fail_launch():
                raise RuntimeError("injected launch failure")
            binding = replace(binding, launch=fail_launch)
        submit_call_ts = time.perf_counter_ns() // 1000
        handle = runtime.submit(spec, binding, hint)
        submit_return_ts = time.perf_counter_ns() // 1000
        consume_start = time.perf_counter_ns() // 1000
        stop_event.wait(consumer_s)
        first_wait_ts = time.perf_counter_ns() // 1000
        if not handle.wait_host(max(0.0, deadline - time.monotonic())):
            raise TimeoutError(f"wait timed out for {spec.task_id}")
        consumer_end_ts = time.perf_counter_ns() // 1000
        expected = sum(item + 1 for item in ranks_for_job(job, world_size))
        task_result = {
            "task_id": spec.task_id, "job_id": job.job_id, "ordinal": comm.id,
            "correct": None, "decision_seq": handle.decision_seq,
            "group_id": spec.group_id, "group_seq": spec.group_seq,
            "producer_start_ts": producer_start, "ready_ts": ready_ts,
            "submit_call_ts": submit_call_ts, "submit_return_ts": submit_return_ts,
            "consumer_start_ts": consume_start, "first_wait_ts": first_wait_ts,
            "consumer_end_ts": consumer_end_ts,
        }
        result["tasks"].append(task_result)
        validation_records.append((task_result, binding.tensor, expected))
    result["job_end_ts"] = time.perf_counter_ns() // 1000
    return result


def _run_dag_job(job: DagJob, *, dag: DagInput, args, runtime, groups, rank: int,
                 device: str, deadline: float, stop_event: threading.Event,
                 dag_events: EventLog, group_ranks: dict[str, tuple[int, ...]],
                 missing_id: str, tails: dict[str, float],
                 validation_records: list[tuple[dict[str, Any], torch.Tensor, int]]) -> dict[str, Any]:
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
        binding = make_collective_binding(dag_task_spec(job, comm, epoch=args.epoch),
                                          groups[comm.group_id], rank=rank, device=device)
        if args.fault == "launch_failure" and rank == 0 and job.job_id == "job-0" and comm_index == 0:
            def fail_launch():
                raise RuntimeError("injected launch failure")
            binding = replace(binding, launch=fail_launch)
        return binding

    samples: dict[str, float] = {}
    compute = make_replay_compute(job, dag.execution, seed=dag.seed, epoch=args.epoch,
                                  rank=rank, jitter=args.compute_jitter, samples=samples,
                                  event_log=dag_events)
    first_compute = next((node.node_id for node in job.nodes if isinstance(node, ComputeNode)), None)
    def run_compute(node, cancellation):
        if args.fault == "compute_failure" and rank == 0 and node.node_id == first_compute:
            raise RuntimeError("injected DAG compute failure")
        compute(node, cancellation)

    runner = DagRunner(dag.graph, job, runner_runtime, epoch=args.epoch, rank=rank,
                       compute_fn=run_compute, make_binding=make_binding,
                       deadline=deadline, poll_interval=args.dag_poll_interval, tails=tails,
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
        validation_records.append((task_result, binding.tensor, expected))
    result["completed_node_ids"] = [f"{job.job_id}/{node_id}" for node_id in runner.completed_node_ids]
    result["job_end_ts"] = time.perf_counter_ns() // 1000
    return result


def run_rank(args: argparse.Namespace) -> dict[str, Any]:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if args.timeout <= 0 or args.setup_timeout <= 0 or args.poll_interval <= 0 or args.dag_poll_interval <= 0:
        raise ValueError("setup/replay timeouts and poll intervals must be positive")
    if not 0 <= args.compute_jitter < 1:
        raise ValueError("compute_jitter must be in [0, 1)")
    comm_profile = getattr(args, "comm_profile", None)
    profile_strict = getattr(args, "profile_strict", True)
    wait_budget_s = getattr(args, "wait_budget_s", 0.02)
    if wait_budget_s < 0:
        raise ValueError("wait_budget_s must be non-negative")
    setup_deadline = time.monotonic() + args.setup_timeout
    dag = load_dag(args.dag, epoch=args.epoch, world_size=world_size) if args.dag else None
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
    dag_tails = compute_tails(dag.graph) if dag is not None else None
    if args.static_order and (dag is None or args.policy not in {"static_fifo", "static_ltf"}):
        raise ValueError("--static-order is only valid with a DAG and static policy")
    if dag is not None:
        order = (load_static_order(args.static_order, dag.graph) if args.static_order
                 else build_static_order(dag.graph, args.policy, tails=dag_tails)
                 if args.policy in {"static_fifo", "static_ltf"} else ())
    else:
        order = ()
    device = _free_device(args.backend)
    inputs = dag if dag is not None else workload
    from runtime_comm_scheduler.runtime.coordinator import CoordinatorState
    from runtime_comm_scheduler.runtime.transport import CoordinatorServer
    server = None
    runtime = None
    groups: dict[str, Any] = {}
    try:
        dist.init_process_group(args.backend,
                                init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
                                rank=rank, world_size=world_size,
                                timeout=timedelta(seconds=_remaining(setup_deadline)))
        groups = _new_groups(inputs, world_size, setup_deadline)
        dist.barrier()
        warmup_collective_count = _warmup_collectives(
            inputs, groups, rank=rank, world_size=world_size, device=device,
            iterations=getattr(args, "warmup_iterations", 0),
        )

        runtime_policy = "static" if args.policy in {"static_fifo", "static_ltf"} else args.policy
        if dag is None and runtime_policy == "static":
            order = linear_static_order(workload, args.policy)
        if rank == 0:
            # The coordinator starts its watchdog at the first group event, before local setup ends.
            coordinator = CoordinatorState(tuple(range(world_size)), epoch=args.epoch, policy=runtime_policy,
                                          static_order=order,
                                          wait_budget_s=wait_budget_s,
                                          epoch_timeout_s=args.setup_timeout + args.timeout)
            server = CoordinatorServer(coordinator, "127.0.0.1", args.control_port)
            server.start()

        client = ControlClient(rank, args.epoch, "127.0.0.1", args.control_port,
                               _remaining(setup_deadline))
        probe = _FailingProbe() if args.fault == "completion_probe_failure" and rank == 0 else None
        runtime = RankRuntime(rank, args.epoch, client, executor=DirectExecutor(),
                              completion_poll_interval_s=args.poll_interval,
                              event_log=EventLog("runtime", rank), completion_probe=probe)
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

        dag_events = EventLog("dag", rank)
        stop_event = threading.Event()
        # This barrier is the common application release boundary.  It is the
        # default process group, never a job collective controlled by the
        # coordinator.
        if server is not None:
            server.wait_ready(_remaining(setup_deadline))
        dist.barrier()
        application_release_ts = time.perf_counter_ns() // 1000
        deadline = time.monotonic() + args.timeout
        validation_records: list[tuple[dict[str, Any], torch.Tensor, int]] = []
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
            )
            jobs = _run_jobs(local_jobs, run_one, runtime=runtime, deadline=deadline,
                             stop_event=stop_event, thread_name_prefix="dag")
        else:
            local_jobs = [job for job in workload.jobs if rank in ranks_for_job(job, world_size)]
            run_one = lambda job: _run_linear_job(
                job, workload=workload, args=args, runtime=runtime, groups=groups, rank=rank, world_size=world_size,
                device=device, deadline=deadline, stop_event=stop_event,
                validation_records=validation_records,
            )
            jobs = _run_jobs(local_jobs, run_one, runtime=runtime, deadline=deadline,
                             stop_event=stop_event, thread_name_prefix="job")
        runtime.finish_epoch(_remaining(deadline))
        communication_drain_end_ts = time.perf_counter_ns() // 1000
        validation_start_ts, validation_end_ts = _validate_deferred(validation_records)
        application_end_ts = max(
            (job["job_end_ts"] for job in jobs), default=application_release_ts
        )
        harness_start_ts = time.perf_counter_ns() // 1000
        output = {
            "rank": rank, "status": "ok", "mode": "runtime",
            "input_mode": "dag" if dag is not None else "linear",
            "policy": args.policy, "backend": args.backend, "jobs": jobs,
            "epoch": args.epoch, "world_size": world_size, "max_inflight": 1,
            "completion_poll_interval_s": args.poll_interval,
            "dag_poll_interval_s": args.dag_poll_interval,
            "compute_jitter": args.compute_jitter,
            "warmup_iterations": getattr(args, "warmup_iterations", 0),
            "warmup_collective_count": warmup_collective_count,
            "estimate_source": "offline_profile" if profile else "manifest",
            "communication_profile": {
                "path": str(comm_profile) if comm_profile else None,
                "digest": profile.digest() if profile else None,
                "schema_version": profile.schema_version if profile else None,
                "strict": profile_strict if profile else None,
                "environment": dict(profile.environment) if profile else None,
            },
            "wait_budget_s": wait_budget_s,
            "application_release_ts": application_release_ts,
            "application_end_ts": application_end_ts,
            "communication_drain_end_ts": communication_drain_end_ts,
            "validation_start_ts": validation_start_ts,
            "validation_end_ts": validation_end_ts,
            "harness_start_ts": harness_start_ts,
            "harness_end_ts": None,
            "application_makespan_us": application_end_ts - application_release_ts,
            "communication_drain_makespan_us": communication_drain_end_ts - application_release_ts,
            "validation_total_us": validation_end_ts - validation_start_ts,
            "task_sequence": [task["task_id"] for job in jobs for task in job["tasks"]],
            "grant_sequence": runtime.grant_order, "launch_sequence": runtime.launch_order,
            "runtime_events": runtime.event_log.as_dict()["events"],
        }
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
        if server is not None:
            output["decision_records"] = list(server.coordinator.records)
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
                for group in reversed(list(groups.values())):
                    try:
                        dist.destroy_process_group(group)
                    except Exception:
                        pass
                if dist.is_initialized():
                    dist.destroy_process_group()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=("static_fifo", "static_ltf", "fifo", "ltf", "lookahead"), default="fifo")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--workload")
    source.add_argument("--dag", type=Path)
    parser.add_argument("--static-order", type=Path)
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--setup-timeout", type=float, default=20.0)
    parser.add_argument("--poll-interval", type=float, default=0.001)
    parser.add_argument("--dag-poll-interval", type=float, default=0.001)
    parser.add_argument("--compute-jitter", type=float, default=0.0)
    parser.add_argument("--wait-budget-s", type=float, default=0.02)
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--comm-profile", type=Path)
    parser.add_argument("--profile-strict", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--control-port", type=int, required=True)
    parser.add_argument("--fault", choices=("none", "missing_task", "metadata_mismatch", "launch_failure",
                                               "completion_probe_failure", "compute_failure", "binding_failure"), default="none")
    args = parser.parse_args()
    if args.timeout <= 0 or args.setup_timeout <= 0 or args.wait_budget_s < 0 or args.warmup_iterations < 0:
        parser.error("setup-timeout and timeout must be positive; wait-budget-s must be non-negative")
    try:
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
