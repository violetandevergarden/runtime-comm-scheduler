"""One-rank JobPacer linear or validated-DAG replay worker."""

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

from runtime_comm_scheduler.runtime import (
    CollectiveSpec, DirectExecutor, EventLog, LocalBinding, RankRuntime,
)
from runtime_comm_scheduler.runtime.transport import ControlClient
from runtime_comm_scheduler.dag import (
    CommNode,
    DagInput,
    DagJob,
    DagRunner,
    build_static_order,
    dag_group_specs,
    dag_task_spec,
    load_dag,
    load_static_order,
)

try:
    from .runtime_adapter import group_spec, task_hint, task_spec
    from .workloads import Job, Workload, load_workload, ranks_for_job
except ImportError:  # pragma: no cover - direct worker execution
    from runtime_adapter import group_spec, task_hint, task_spec
    from workloads import Job, Workload, load_workload, ranks_for_job


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


def _new_groups(workload: Workload | DagInput, world_size: int) -> dict[str, Any]:
    if isinstance(workload, DagInput):
        group_defs = ((group.group_id, group.ranks) for group in workload.groups)
    else:
        group_defs = ((job.job_id, ranks_for_job(job, world_size)) for job in workload.jobs)
    # Every process creates communicators in the same manifest order.
    return {group_id: dist.new_group(list(ranks)) for group_id, ranks in group_defs}


def _local_dag_jobs(dag: DagInput, rank: int) -> list[DagJob]:
    return [job for job in dag.jobs
            if rank in dag.group_ranks[next(node.group_id for node in job.nodes if isinstance(node, CommNode))]]


def _make_dag_binding(comm: CommNode, group: Any, rank: int, *, fault: str, job_id: str,
                      comm_index: int, device: str) -> LocalBinding:
    if fault == "binding_failure" and rank == 0 and comm_index == 0:
        raise RuntimeError("injected DAG binding failure")
    dtype = getattr(torch, comm.collective.dtype)
    tensor = torch.full((comm.collective.numel,), float(rank + 1), dtype=dtype, device=device)

    def launch():
        if fault == "launch_failure" and rank == 0 and job_id == "job-0" and comm_index == 0:
            raise RuntimeError("injected launch failure")
        return dist.all_reduce(tensor, group=group, async_op=True)

    return LocalBinding(tensor, group, launch, device=device, keepalive=(tensor,))


def _run_linear(args, runtime, workload: Workload, groups, rank: int, world_size: int, device: str):
    local_jobs = [job for job in workload.jobs if rank in ranks_for_job(job, world_size)]
    result_by_job: dict[str, dict[str, Any]] = {}
    errors: list[BaseException] = []
    stop = threading.Event()
    barrier = threading.Barrier(len(local_jobs)) if local_jobs else None
    deadline = time.monotonic() + args.timeout

    def run_job(job: Job) -> None:
        result = {"job_id": job.job_id, "status": "ok", "tasks": []}
        try:
            if barrier is not None:
                barrier.wait(max(0.0, deadline - time.monotonic()))
            result["job_start_ts"] = time.perf_counter_ns() // 1000
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
                tensor = torch.full((comm.num_bytes // 4,), float(rank + 1), dtype=torch.float32, device=device)
                if args.fault == "metadata_mismatch" and rank == 1 and job.job_id == "job-1" and index == 0:
                    spec = replace(spec, collective=CollectiveSpec("all_reduce", spec.collective.numel + 1,
                                                                   spec.collective.num_bytes + 4, "float32",
                                                                   (spec.collective.numel + 1,)))
                group = groups[job.job_id]

                def launch(tensor=tensor, group=group, job=job, index=index):
                    if args.fault == "launch_failure" and rank == 0 and job.job_id == "job-0" and index == 0:
                        raise RuntimeError("injected launch failure")
                    return dist.all_reduce(tensor, group=group, async_op=True)

                submit_call_ts = time.perf_counter_ns() // 1000
                handle = runtime.submit(spec, LocalBinding(tensor, group, launch, device=device, keepalive=(tensor,)), hint)
                submit_return_ts = time.perf_counter_ns() // 1000
                consume_start = time.perf_counter_ns() // 1000
                time.sleep(comm.consumer_compute_s)
                first_wait_ts = time.perf_counter_ns() // 1000
                if not handle.wait_host(max(0.0, deadline - time.monotonic())):
                    raise TimeoutError(f"wait timed out for {spec.task_id}")
                consumer_end_ts = time.perf_counter_ns() // 1000
                expected = sum(item + 1 for item in ranks_for_job(job, world_size))
                correct = bool(torch.all(tensor == expected).item())
                if not correct:
                    raise AssertionError(f"incorrect result for {spec.task_id}")
                result["tasks"].append({
                    "task_id": spec.task_id, "job_id": job.job_id, "ordinal": comm.id,
                    "correct": correct, "decision_seq": handle.decision_seq,
                    "group_id": spec.group_id, "group_seq": spec.group_seq,
                    "producer_start_ts": producer_start, "ready_ts": ready_ts,
                    "submit_call_ts": submit_call_ts, "submit_return_ts": submit_return_ts,
                    "consumer_start_ts": consume_start, "first_wait_ts": first_wait_ts,
                    "consumer_end_ts": consumer_end_ts,
                })
        except BaseException as exc:  # noqa: BLE001
            result["status"] = "error"
            result["error"] = f"{type(exc).__name__}: {exc}"
            errors.append(exc)
            stop.set()
            runtime.abort(exc, stage="job", job_id=job.job_id)
        result["job_end_ts"] = time.perf_counter_ns() // 1000
        result_by_job[job.job_id] = result

    threads = [threading.Thread(target=run_job, args=(job,), name=f"job-{job.job_id}") for job in local_jobs]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(max(0.0, deadline - time.monotonic()))
    if any(thread.is_alive() for thread in threads):
        stop.set()
        error = TimeoutError("linear job exceeded replay deadline")
        runtime.abort(error, stage="job_timeout")
        raise error
    if errors:
        raise errors[0]
    return [result_by_job[job.job_id] for job in local_jobs]


def _run_dag(args, runtime, dag: DagInput, groups, rank: int, device: str, dag_events: EventLog):
    local_jobs = _local_dag_jobs(dag, rank)
    result_by_job: dict[str, dict[str, Any]] = {}
    errors: list[BaseException] = []
    stop = threading.Event()
    barrier = threading.Barrier(len(local_jobs)) if local_jobs else None
    deadline = time.monotonic() + args.timeout
    all_comms = [(job, node) for job in dag.jobs for node in job.nodes if isinstance(node, CommNode)]
    missing_id = f"{all_comms[0][0].job_id}/{all_comms[0][1].node_id}" if all_comms else ""

    def run_job(job: DagJob) -> None:
        result: dict[str, Any] = {"job_id": job.job_id, "status": "ok", "tasks": []}
        try:
            dag_events.record("job_barrier_wait", job_id=job.job_id)
            if barrier is not None:
                barrier.wait(max(0.0, deadline - time.monotonic()))
            result["job_start_ts"] = time.perf_counter_ns() // 1000
            comm_nodes = [node for node in job.nodes if isinstance(node, CommNode)]
            runner_runtime = runtime
            if args.fault == "missing_task" and rank == 1 and missing_id in {
                f"{job.job_id}/{node.node_id}" for node in comm_nodes
            }:
                runner_runtime = _DropOneSubmit(runtime, missing_id)

            def make_binding(comm: CommNode) -> LocalBinding:
                comm_index = next(index for index, item in enumerate(comm_nodes) if item.node_id == comm.node_id)
                return _make_dag_binding(comm, groups[comm.group_id], rank, fault=args.fault,
                                         job_id=job.job_id, comm_index=comm_index, device=device)

            def compute(node, duration, cancellation):
                if args.fault == "compute_failure" and rank == 0 and node.node_id == next(
                    item.node_id for item in job.nodes if hasattr(item, "estimated_duration_s")
                ):
                    raise RuntimeError("injected DAG compute failure")
                cancellation.wait(duration)

            runner = DagRunner(dag, job, runner_runtime, epoch=args.epoch, rank=rank,
                               make_binding=make_binding, timeout=args.timeout,
                               poll_interval=args.dag_poll_interval, compute_jitter=args.compute_jitter,
                               enable_lookahead=args.policy == "lookahead",
                               stop_event=stop, compute_fn=compute, event_log=dag_events, deadline=deadline)
            runner_result = runner.run()
            result.update(runner_result)
            result["job_start_ts"] = result.get("job_start_ts") or time.perf_counter_ns() // 1000
            expected = sum(rank_id + 1 for rank_id in dag.group_ranks[
                next(comm.group_id for comm in comm_nodes)
            ])
            for node in job.nodes:
                if not isinstance(node, CommNode):
                    continue
                binding = runner.bindings[node.node_id]
                tensor = binding.tensor
                correct = bool(torch.all(tensor == expected).item())
                if not correct:
                    raise AssertionError(f"incorrect all_reduce result for {job.job_id}/{node.node_id}")
                spec = dag_task_spec(job, node, epoch=args.epoch)
                result["tasks"].append({"task_id": spec.task_id, "node_id": node.node_id,
                                        "job_id": job.job_id, "group_id": spec.group_id,
                                        "group_seq": spec.group_seq, "decision_seq": runner.handles[node.node_id].decision_seq,
                                        "correct": correct})
            result["completed_node_ids"] = [f"{job.job_id}/{node_id}" for node_id in runner.completed_node_ids]
            result["job_end_ts"] = time.perf_counter_ns() // 1000
        except BaseException as exc:  # noqa: BLE001
            result["status"] = "error"
            result["error"] = f"{type(exc).__name__}: {exc}"
            errors.append(exc)
            stop.set()
            runtime.abort(exc, stage="dag_runner", job_id=job.job_id)
        result_by_job[job.job_id] = result

    threads = [threading.Thread(target=run_job, args=(job,), name=f"dag-{job.job_id}") for job in local_jobs]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(max(0.0, deadline - time.monotonic()))
    if any(thread.is_alive() for thread in threads):
        stop.set()
        error = TimeoutError("DAG job runners exceeded shared replay deadline")
        runtime.abort(error, stage="dag_timeout")
        raise error
    if errors:
        raise errors[0]
    return [result_by_job[job.job_id] for job in local_jobs]


def run_rank(args: argparse.Namespace) -> dict[str, Any]:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if args.timeout <= 0 or args.poll_interval <= 0 or args.dag_poll_interval <= 0:
        raise ValueError("timeout and poll intervals must be positive")
    if not 0 <= args.compute_jitter < 1:
        raise ValueError("compute_jitter must be in [0, 1)")
    dag = load_dag(args.dag, epoch=args.epoch, world_size=world_size) if args.dag else None
    workload = None if dag else load_workload(args.workload or "balanced")
    if args.static_order and (dag is None or args.policy not in {"static_fifo", "static_ltf"}):
        raise ValueError("--static-order is only valid with a DAG and static policy")
    if dag is not None:
        order = load_static_order(args.static_order, dag) if args.static_order else build_static_order(dag, args.policy) if args.policy in {"static_fifo", "static_ltf"} else ()
    else:
        order = ()
    device = _free_device(args.backend)
    dist.init_process_group(args.backend, init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}", rank=rank, world_size=world_size)
    inputs = dag if dag is not None else workload
    groups = _new_groups(inputs, world_size)
    dist.barrier()

    from runtime_comm_scheduler.runtime.coordinator import CoordinatorState
    from runtime_comm_scheduler.runtime.transport import CoordinatorServer
    server = None
    runtime_policy = "static" if args.policy in {"static_fifo", "static_ltf"} else args.policy
    if dag is None and runtime_policy == "static":
        try:
            from .plan_builder import build_plan
        except ImportError:  # pragma: no cover
            from plan_builder import build_plan
        plan = build_plan(workload, "fifo" if args.policy == "static_fifo" else "ltf")
        order = tuple(f"{key.process_group_id}/comm-{key.ordinal}" for key in plan.keys)
    if rank == 0:
        coordinator = CoordinatorState(tuple(range(world_size)), epoch=args.epoch, policy=runtime_policy,
                                      static_order=order, epoch_timeout_s=args.timeout)
        server = CoordinatorServer(coordinator, "127.0.0.1", args.control_port)
        server.start()

    client = ControlClient(rank, args.epoch, "127.0.0.1", args.control_port, args.timeout)
    probe = _FailingProbe() if args.fault == "completion_probe_failure" and rank == 0 else None
    runtime = RankRuntime(rank, args.epoch, client, executor=DirectExecutor(),
                          completion_poll_interval_s=args.poll_interval,
                          event_log=EventLog("runtime", rank), completion_probe=probe)
    if dag is not None:
        used_groups = {node.group_id for job in dag.jobs for node in job.nodes
                       if isinstance(node, CommNode) and rank in dag.group_ranks[node.group_id]}
        specs = {group.group_id: group for group in dag_group_specs(dag, epoch=args.epoch)}
        for group_id in sorted(used_groups):
            runtime.register_group(specs[group_id], groups[group_id])
    else:
        for job in workload.jobs:
            if rank in ranks_for_job(job, world_size):
                runtime.register_group(group_spec(job, world_size, epoch=args.epoch), groups[job.job_id])
    runtime.start(args.timeout)
    if server is not None:
        server.wait_ready(args.timeout)
    dag_events = EventLog("dag", rank)
    try:
        jobs = (_run_dag(args, runtime, dag, groups, rank, device, dag_events)
                if dag is not None else _run_linear(args, runtime, workload, groups, rank, world_size, device))
        runtime.finish_epoch(max(0.0, args.timeout))
        output = {
            "rank": rank, "status": "ok", "mode": "runtime",
            "input_mode": "dag" if dag is not None else "linear",
            "policy": args.policy, "backend": args.backend, "jobs": jobs,
            "epoch": args.epoch, "world_size": world_size, "max_inflight": 1,
            "completion_poll_interval_s": args.poll_interval,
            "dag_poll_interval_s": args.dag_poll_interval,
            "compute_jitter": args.compute_jitter,
            "task_sequence": [task["task_id"] for job in jobs for task in job["tasks"]],
            "grant_sequence": runtime.grant_order, "launch_sequence": runtime.launch_order,
            "runtime_events": runtime.event_log.as_dict()["events"],
        }
        if dag is not None:
            local_jobs = _local_dag_jobs(dag, rank)
            task_groups = {f"{job.job_id}/{node.node_id}": node.group_id
                           for job in dag.jobs for node in job.nodes if isinstance(node, CommNode)}
            output.update({
                "dag_name": dag.name,
                "dag_seed": dag.seed,
                "manifest_digest": dag.manifest_digest,
                "canonical_dag": json.loads(dag.canonical_json),
                "dag_tails_s": {task_id: dag.tails[task_id] for task_id in dag.expected_task_ids},
                "groups": [{"group_id": group.group_id, "ranks": list(group.ranks)} for group in dag.groups],
                "expected_task_ids": [task_id for task_id in dag.expected_task_ids
                                      if rank in dag.group_ranks[task_groups[task_id]]],
                "expected_node_ids": [f"{job.job_id}/{node.node_id}" for job in local_jobs for node in job.nodes],
                "dag_events": dag_events.as_dict()["events"],
            })
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
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--workload")
    source.add_argument("--dag", type=Path)
    parser.add_argument("--static-order", type=Path)
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--poll-interval", type=float, default=0.001)
    parser.add_argument("--dag-poll-interval", type=float, default=0.001)
    parser.add_argument("--compute-jitter", type=float, default=0.0)
    parser.add_argument("--control-port", type=int, required=True)
    parser.add_argument("--fault", choices=("none", "missing_task", "metadata_mismatch", "launch_failure",
                                               "completion_probe_failure", "compute_failure", "binding_failure"), default="none")
    args = parser.parse_args()
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
