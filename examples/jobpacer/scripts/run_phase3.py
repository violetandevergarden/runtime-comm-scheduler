"""Launch and collect the Phase 3 JobPacer runtime replay."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from runtime_comm_scheduler.dag import CommNode, build_static_order
from examples.jobpacer.analysis.benchmark_paths import (
    is_formal_experiment_input, repository_path, resolve_migrated_path,
)
from examples.jobpacer.analysis.runtime_results import expected_dag_results, metrics, performance, validate_results
from examples.jobpacer.runtime.runtime_adapter import (
    all_specs, apply_dag_compute_profile, apply_dag_profile, load_dag, load_static_order,
)
from examples.jobpacer.workloads import load_workload, ranks_for_job
from examples.jobpacer.comm_profile import apply_profile, load_profile
from examples.jobpacer.gpu.cuda_devices import (
    compute_profile_software, validate_visible_cuda_devices, visible_cuda_uuids,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FORMAL_INPUT_ROOT = (ROOT / "benchmark/phase3/experiments").resolve()
PHASE3_MIGRATION_MAP = ROOT / "benchmark/phase3/results/migration-map.json"
BARE_FAILURE_CLEANUP_GRACE_S = 5.0
RENDEZVOUS_STARTUP_MAX_ATTEMPTS = 5
RENDEZVOUS_BIND_ERROR_MARKER = "The server socket has failed to listen"


def _free_port(exclude: set[int] | None = None) -> int:
    excluded = exclude or set()
    for _ in range(32):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            port = int(sock.getsockname()[1])
        if port not in excluded:
            return port
    raise RuntimeError("could not allocate distinct local rendezvous and control ports")


def _is_rendezvous_bind_conflict(errors: list[str]) -> bool:
    return any(
        RENDEZVOUS_BIND_ERROR_MARKER in error
        and "EADDRINUSE" in error
        and "address already in use" in error.lower()
        for error in errors
    )


def _should_retry_rendezvous_startup(
    results: list[dict[str, Any]], errors: list[str], attempt_index: int,
) -> bool:
    """Retry only a pre-task TCPStore bind collision, never a partial rank result."""
    return (
        not results
        and attempt_index + 1 < RENDEZVOUS_STARTUP_MAX_ATTEMPTS
        and _is_rendezvous_bind_conflict(errors)
    )


def _start(rank: int, args: argparse.Namespace, rendezvous_port: int, control_port: int) -> subprocess.Popen[str]:
    env = dict(os.environ)
    env.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(rendezvous_port), RANK=str(rank),
               WORLD_SIZE=str(args.world_size), LOCAL_RANK=str(rank))
    if args.comm_engine == "bare":
        # NCCL 2.26+ implicit ordering is disabled by default. Each worker
        # still uses the same frozen host call order; this enables NCCL's
        # supported cross-communicator ordering and possible device overlap.
        env["NCCL_LAUNCH_ORDER_IMPLICIT"] = "1"
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(ROOT), env.get("PYTHONPATH", "")))
    command = [sys.executable, "-m", "examples.jobpacer.runtime.runtime_worker",
               "--policy", args.policy, "--backend", args.backend,
               "--comm-engine", args.comm_engine,
               "--epoch", str(args.epoch), "--timeout", str(args.timeout),
               "--setup-timeout", str(args.setup_timeout),
               "--poll-interval", str(args.poll_interval), "--dag-poll-interval", str(args.dag_poll_interval),
               "--wait-budget-s", str(args.wait_budget_s),
               "--warmup-iterations", str(args.warmup_iterations),
               "--observation-mode", args.observation_mode,
               "--control-port", str(control_port),
               "--fault", args.fault]
    command.extend(("--compute-jitter", str(args.compute_jitter),
                    "--compute-mode", args.compute_mode,
                    "--lane", args.lane,
                    "--compute-matrix-size", str(args.compute_matrix_size),
                    "--compute-repeats", str(args.compute_repeats),
                    "--binding-preparation", args.binding_preparation,
                    "--declaration-mode", args.declaration_mode))
    if args.cuda_profiler_dir:
        command.extend(["--cuda-profiler-dir", str(args.cuda_profiler_dir)])
    if args.wake_completion_on_submit:
        command.append("--wake-completion-on-submit")
    if args.dag:
        command.extend(("--dag", str(args.dag)))
    else:
        command.extend(("--workload", args.workload or "balanced"))
    if args.static_order:
        command.extend(("--static-order", str(args.static_order)))
    if args.comm_profile:
        command.extend(("--comm-profile", str(args.comm_profile)))
    if args.compute_profile:
        command.extend(("--compute-profile", str(args.compute_profile)))
    command.extend(("--matmul-precision", args.matmul_precision))
    command.append("--profile-strict" if args.profile_strict else "--no-profile-strict")
    return subprocess.Popen(
        command,
        cwd=str(ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )


def _collect(rank: int, process: subprocess.Popen[str], timeout: float) -> tuple[dict[str, Any] | None, str | None]:
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        return None, f"rank {rank} timed out: {stderr[-2000:]}"
    if process.returncode != 0:
        return None, f"rank {rank} exited {process.returncode}: {stderr[-3000:]} {stdout[-1000:]}"
    try:
        return json.loads(stdout.strip().splitlines()[-1]), None
    except (IndexError, json.JSONDecodeError) as exc:
        return None, f"rank {rank} emitted invalid JSON ({exc}): {stdout[-2000:]}"


def _stop_failed_siblings(processes: list[subprocess.Popen[str]], *, comm_engine: str) -> None:
    """Stop replay children after a failure, allowing bare peers to observe fail-stop first."""
    if comm_engine == "bare":
        # Bare workers publish failures through the rendezvous Store and exit
        # themselves instead of destroying an NCCL communicator with pending
        # work. Give that notification a short window to reach the peer; retain
        # a hard parent-side bound if a worker is stuck in a backend call.
        for process in processes:
            if process.poll() is not None:
                continue
            try:
                process.wait(timeout=BARE_FAILURE_CLEANUP_GRACE_S)
            except subprocess.TimeoutExpired:
                process.kill()
        return
    for process in processes:
        if process.poll() is None:
            process.kill()


def _option_was_set(name: str) -> bool:
    return any(token == name or token.startswith(name + "=") for token in sys.argv[1:])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("static_fifo", "static_ltf", "fifo", "ltf", "lookahead", "bare"), default="fifo")
    parser.add_argument("--comm-engine", choices=("new", "old", "raw-ordered", "bare"), default="new")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--workload")
    source.add_argument("--dag", type=Path)
    parser.add_argument("--static-order", type=Path)
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--startup-attempts", type=int, choices=(1, 2, 3), default=RENDEZVOUS_STARTUP_MAX_ATTEMPTS)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--setup-timeout", type=float, default=20.0)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--poll-interval", type=float, default=0.001)
    parser.add_argument("--wake-completion-on-submit", action="store_true",
                        help="wake the periodic completion probe when new work becomes probeable")
    parser.add_argument("--dag-poll-interval", type=float, default=0.001)
    parser.add_argument("--compute-jitter", type=float, default=0.0)
    parser.add_argument("--compute-mode", choices=("host-sleep", "cuda-matmul"), default="host-sleep")
    parser.add_argument("--lane", choices=("H", "S"), default="H")
    parser.add_argument("--compute-matrix-size", type=int, default=256)
    parser.add_argument("--compute-repeats", type=int, default=1)
    parser.add_argument("--matmul-precision", choices=("highest", "high", "medium"), default="highest")
    parser.add_argument("--wait-budget-s", type=float, default=0.02)
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--binding-preparation", choices=("precreate", "on-ready"), default="precreate",
                        help="linear replay only; DAG preparation remains unchanged")
    parser.add_argument("--declaration-mode", choices=("before-producer", "on-submit"),
                        default="before-producer",
                        help="linear replay only; on-submit is unavailable to DAG/Lookahead")
    parser.add_argument("--observation-mode", choices=("minimal", "diagnostic", "full"), default="full",
                        help="minimal; control-path diagnostic without policy snapshots; or full legacy trace")
    parser.add_argument("--comm-profile", type=Path)
    parser.add_argument("--compute-profile", type=Path)
    parser.add_argument("--profile-strict", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cuda-profiler-dir", type=Path)
    parser.add_argument("--fault", choices=("none", "missing_task", "metadata_mismatch", "launch_failure",
                                              "completion_probe_failure", "compute_failure", "binding_failure"),
                        default="none")
    args = parser.parse_args(argv)
    if args.timeout <= 0 or args.setup_timeout <= 0 or args.poll_interval <= 0 or args.dag_poll_interval <= 0:
        parser.error("setup/replay timeouts and poll intervals must be positive")
    if args.wait_budget_s < 0 or args.warmup_iterations < 0:
        parser.error("wait-budget-s must be non-negative")
    if not 0 <= args.compute_jitter < 1:
        parser.error("compute-jitter must be in [0, 1)")
    if args.compute_matrix_size < 16 or args.compute_repeats <= 0:
        parser.error("compute-matrix-size must be >= 16 and compute-repeats positive")
    if args.compute_mode == "cuda-matmul" and args.backend != "nccl":
        parser.error("cuda-matmul compute requires --backend nccl")
    if args.compute_mode == "cuda-matmul" and not args.dag and args.lane != "S":
        parser.error("linear cuda-matmul requires --lane S")
    if args.lane == "S" and (args.backend != "nccl" or args.dag
                              or args.compute_mode != "cuda-matmul"
                              or args.binding_preparation != "precreate"):
        parser.error("S lane requires linear NCCL, cuda-matmul, and precreate bindings")
    if args.epoch < 0 or args.world_size <= 0:
        parser.error("epoch must be non-negative and world-size positive")
    if args.comm_engine == "bare" and (not args.dag or args.policy != "bare"):
        parser.error("bare requires a schema-v2 DAG, --comm-engine bare, and --policy bare")
    if args.comm_engine == "bare" and args.world_size != 2:
        parser.error("bare-ordered-v2-layered-round-robin is validated only for a two-rank NCCL world")
    if args.policy == "bare" and args.comm_engine != "bare":
        parser.error("--policy bare requires --comm-engine bare")
    if args.backend == "nccl":
        try:
            validate_visible_cuda_devices(args.world_size)
        except (RuntimeError, ValueError) as exc:
            parser.error(str(exc))
    if args.declaration_mode == "on-submit" and (args.dag or args.policy == "lookahead"):
        parser.error("--declaration-mode on-submit supports linear non-Lookahead replay only")
    if args.dag:
        args.dag = resolve_migrated_path(repository_path(args.dag), PHASE3_MIGRATION_MAP)
    elif args.workload:
        workload_path = Path(args.workload)
        if workload_path.exists() or workload_path.suffix.lower() == ".json":
            args.workload = str(resolve_migrated_path(repository_path(workload_path), PHASE3_MIGRATION_MAP))
    if args.static_order:
        args.static_order = resolve_migrated_path(repository_path(args.static_order), PHASE3_MIGRATION_MAP)
    if args.comm_profile:
        args.comm_profile = resolve_migrated_path(repository_path(args.comm_profile), PHASE3_MIGRATION_MAP)
    if args.compute_profile:
        args.compute_profile = resolve_migrated_path(repository_path(args.compute_profile), PHASE3_MIGRATION_MAP)
    if args.output:
        args.output = repository_path(args.output)
    if args.cuda_profiler_dir:
        args.cuda_profiler_dir = repository_path(args.cuda_profiler_dir)
        args.cuda_profiler_dir.mkdir(parents=True, exist_ok=True)
    source_path = args.dag or (Path(args.workload) if args.workload else None)
    if (source_path is not None and source_path.exists()
            and is_formal_experiment_input(source_path, FORMAL_INPUT_ROOT) and not args.comm_profile):
        parser.error("formal Phase 3 experiment inputs require --comm-profile")
    expected: dict[str, dict[str, Any]] = {str(rank): {} for rank in range(args.world_size)}
    expected["__all__"] = {}
    expected["__groups__"] = {}
    expected_nodes: dict[int, set[str]] | None = None
    digests: set[str] | None = None
    order: tuple[str, ...] = ()
    bare_order = None
    if args.dag:
        try:
            dag = load_dag(args.dag, epoch=args.epoch, world_size=args.world_size)
            if dag.execution.schema_version == 2:
                if args.backend != "nccl":
                    parser.error("DAG schema-v2 cuda-program inputs require --backend nccl")
                if (_option_was_set("--compute-matrix-size") or _option_was_set("--compute-repeats")
                        or _option_was_set("--compute-mode")):
                    parser.error("DAG schema-v2 programs define their own operators and reject global compute overrides")
                if args.compute_jitter != 0:
                    parser.error("DAG schema-v2 uses frozen workload inputs; --compute-jitter is unsupported")
            if args.comm_engine != "new":
                if dag.execution.schema_version != 2 or args.backend != "nccl":
                    parser.error("old/raw-ordered/bare adapters require a schema-v2 NCCL DAG")
                if args.comm_engine == "old" and args.policy not in {"static_fifo", "static_ltf"}:
                    parser.error("old scheduler arms require static_fifo or static_ltf")
                if args.comm_engine == "raw-ordered" and args.policy != "static_fifo":
                    parser.error("raw-ordered uses the common static_fifo sequence")
                if args.comm_engine == "bare" and args.policy != "bare":
                    parser.error("bare uses direct DAG readiness and requires --policy bare")
            if args.policy == "bare" and args.comm_engine != "bare":
                parser.error("--policy bare requires --comm-engine bare")
            if args.comm_profile:
                dag = apply_dag_profile(
                    dag, load_profile(args.comm_profile),
                    {"backend": args.backend,
                     "device_type": "cuda" if args.backend == "nccl" else "cpu",
                     "world_size": args.world_size},
                    strict=args.profile_strict,
                )
            if args.compute_profile:
                from examples.jobpacer.gpu.gpu_compute_profile import load_gpu_compute_profile
                dag = apply_dag_compute_profile(
                    dag, load_gpu_compute_profile(args.compute_profile),
                    device_uuids=visible_cuda_uuids(args.world_size),
                    software=compute_profile_software(args.matmul_precision),
                    strict=args.profile_strict,
                )
            elif dag.execution.schema_version == 2 and args.policy in {"static_ltf", "ltf"}:
                parser.error("schema-v2 static_ltf/ltf requires --compute-profile")
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        if args.static_order and args.policy not in {"static_fifo", "static_ltf", "bare"}:
            parser.error("--static-order requires a static policy")
        if args.static_order:
            try:
                order = load_static_order(args.static_order, dag.graph, extra_predecessors=dag.execution.submit_after)
            except (OSError, ValueError) as exc:
                parser.error(str(exc))
        elif args.policy in {"static_fifo", "static_ltf"}:
            order = build_static_order(
                dag.graph, args.policy,
                extra_predecessors=dag.execution.submit_after,
            )
        if args.comm_engine == "bare":
            try:
                from examples.jobpacer.runtime.dag_comm_adapters import build_bare_order
                bare_order = build_bare_order(
                    dag.graph, submit_after=dag.execution.submit_after, input_hash=dag.input_hash,
                )
                if args.static_order and tuple(order) != bare_order.sequence:
                    raise ValueError("frozen bare order differs from the versioned default sequence")
            except ValueError as exc:
                parser.error(str(exc))
        expected_dag = expected_dag_results(dag.graph, args.world_size)
        expected = expected_dag["expected"]
        expected_nodes = expected_dag["expected_nodes"]
        digests = {dag.manifest_digest}
    else:
        workload = load_workload(args.workload or "balanced")
        profile = None
        if args.comm_profile:
            profile = load_profile(args.comm_profile)
            workload = apply_profile(
                workload,
                profile,
                {"backend": args.backend, "device_type": "cuda" if args.backend == "nccl" else "cpu",
                 "world_size": args.world_size},
                strict=args.profile_strict,
            )
        for job, _index, spec, _hint in all_specs(workload, epoch=args.epoch):
            meta = {"group_id": spec.group_id, "group_seq": spec.group_seq}
            expected["__all__"][spec.task_id] = meta
            expected["__groups__"].setdefault(spec.group_id, ranks_for_job(job, args.world_size))
            for rank in ranks_for_job(job, args.world_size):
                expected[str(rank)][spec.task_id] = meta
        if args.static_order:
            parser.error("--static-order requires --dag")

    results: list[dict[str, Any]] = []
    errors: list[str] = []
    rendezvous_startup_attempts = []
    for attempt_index in range(args.startup_attempts):
        rendezvous_port = _free_port()
        control_port = _free_port({rendezvous_port})
        processes = [_start(rank, args, rendezvous_port, control_port)
                     for rank in range(args.world_size)]
        results = []
        errors = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(processes)) as pool:
            futures = [pool.submit(_collect, rank, process, args.setup_timeout + args.timeout + 5)
                       for rank, process in enumerate(processes)]
            while not all(future.done() for future in futures):
                failed_children = [process for process in processes
                                   if process.poll() is not None and process.returncode != 0]
                if failed_children:
                    _stop_failed_siblings(processes, comm_engine=args.comm_engine)
                    break
                time.sleep(0.02)
            for future in futures:
                result, error = future.result()
                if error:
                    errors.append(error)
                elif result is not None:
                    results.append(result)
        rendezvous_startup_attempts.append({
            "attempt": attempt_index + 1,
            "rendezvous_port": rendezvous_port,
            "control_port": control_port,
            "errors": list(errors),
        })
        if attempt_index + 1 < args.startup_attempts and _should_retry_rendezvous_startup(results, errors, attempt_index):
            continue
        break
    results.sort(key=lambda item: item.get("rank", -1))
    expected_config = {
            "epoch": args.epoch, "world_size": args.world_size,
            "max_inflight": None if args.comm_engine == "bare" else 1,
            "process_group_timeout_s": (min(20.0, args.timeout)
                                        if args.backend == "nccl" else None),
            "backend": args.backend, "completion_poll_interval_s": args.poll_interval,
            "comm_engine": args.comm_engine,
            "wake_completion_on_submit": args.wake_completion_on_submit,
            "dag_poll_interval_s": args.dag_poll_interval, "compute_jitter": args.compute_jitter,
            "compute_mode": args.compute_mode,
            "measurement_lane": args.lane,
            "compute_matrix_size": args.compute_matrix_size if args.compute_mode == "cuda-matmul" else None,
            "compute_repeats": args.compute_repeats if args.compute_mode == "cuda-matmul" else None,
            "warmup_iterations": args.warmup_iterations,
            "observation_mode": args.observation_mode,
        }
    if not args.dag:
        expected_config["binding_preparation"] = args.binding_preparation
        expected_config["declaration_mode"] = args.declaration_mode
    validation = validate_results(
        results, args.world_size, expected=expected, expected_nodes=expected_nodes, digests=digests,
        expected_config=expected_config,
    )
    if args.dag and args.policy in {"static_fifo", "static_ltf"}:
        for result in results:
            rank = int(result.get("rank", -1))
            local_tasks = set(result.get("expected_task_ids", ()))
            expected_projection = tuple(task_id for task_id in order if task_id in local_tasks)
            actual_projection = tuple(result.get("launch_sequence", ()))
            if actual_projection != expected_projection:
                validation["errors"].append(
                    f"rank {rank} static sequence mismatch: expected={expected_projection}, "
                    f"actual={actual_projection}"
                )
                validation["status"] = "failed"
    if args.dag and args.comm_engine == "bare" and len(results) == args.world_size:
        assert bare_order is not None
        task_group = {
            f"{job.job_id}/{node.node_id}": node.group_id
            for job in dag.graph.jobs for node in job.nodes
            if isinstance(node, CommNode)
        }
        for result in results:
            rank = int(result.get("rank", -1))
            launch_order = tuple(result.get("launch_sequence", ()))
            expected_tasks = tuple(result.get("expected_task_ids", ()))
            expected_projection = tuple(
                task_id for task_id in bare_order.sequence
                if rank in dag.group_ranks[task_group[task_id]]
            )
            if launch_order != expected_projection:
                validation["errors"].append(
                    f"rank {rank} bare launch order differs from its common-order projection: "
                    f"expected={expected_projection}, actual={launch_order}"
                )
                validation["status"] = "failed"
            if len(launch_order) != len(expected_tasks) or set(launch_order) != set(expected_tasks):
                validation["errors"].append(
                    f"rank {rank} bare launch task set differs from its group membership"
                )
                validation["status"] = "failed"
            if (result.get("bare_contract") != bare_order.contract_version
                    or tuple(result.get("bare_default_order", ())) != bare_order.sequence
                    or result.get("bare_default_order_digest") != bare_order.digest
                    or result.get("bare_input_hash") != bare_order.input_hash):
                validation["errors"].append(
                    f"rank {rank} loaded a different bare order contract or digest"
                )
                validation["status"] = "failed"
            if result.get("nccl_launch_order_implicit") != "1":
                validation["errors"].append(
                    f"rank {rank} did not enable NCCL_LAUNCH_ORDER_IMPLICIT=1"
                )
                validation["status"] = "failed"
        group_order = {
            group_id: tuple(task_id for _seq, task_id in sorted(
                (node.group_seq, f"{job.job_id}/{node.node_id}")
                for job in dag.graph.jobs for node in job.nodes
                if isinstance(node, CommNode) and node.group_id == group_id
            ))
            for group_id in dag.group_ranks
        }
        for result in results:
            launch_order = tuple(result.get("launch_sequence", ()))
            for group_id, members in dag.group_ranks.items():
                if result.get("rank") not in members:
                    continue
                projection = tuple(task_id for task_id in launch_order
                                   if task_group.get(task_id) == group_id)
                if projection != group_order[group_id]:
                    validation["errors"].append(
                        f"rank {result.get('rank')} bare group {group_id} order mismatch: "
                        f"expected={group_order[group_id]}, actual={projection}"
                    )
                    validation["status"] = "failed"
    if args.backend == "nccl":
        uuids = [result.get("device_uuid") for result in results]
        if len(uuids) != args.world_size or any(not item for item in uuids) or len(set(uuids)) != len(uuids):
            validation["errors"].append(f"CUDA device UUID mapping is missing or duplicated: {uuids}")
            validation["status"] = "failed"
    diagnostic_present = bool(
        results and all(result.get("runtime_events") for result in results)
        and any(result.get("coordinator_instrumentation") for result in results)
    )
    validation["observation_mode"] = args.observation_mode
    validation["diagnostic_timestamps_present"] = diagnostic_present
    if args.comm_engine == "new" and args.observation_mode != "minimal" and not diagnostic_present:
        validation["errors"].append("diagnostic observation requested but timestamp records are missing")
        validation["status"] = "failed"
    if any(result.get("comm_engine") != args.comm_engine for result in results):
        validation["errors"].append("communication engine differs across worker ranks")
        validation["status"] = "failed"
    validation["errors"] = errors + validation["errors"]
    if errors:
        validation["status"] = "failed"
    config = vars(args).copy()
    config["output"] = str(args.output) if args.output else None
    config["dag"] = str(args.dag) if args.dag else None
    config["static_order"] = str(args.static_order) if args.static_order else None
    config["cuda_profiler_dir"] = str(args.cuda_profiler_dir) if args.cuda_profiler_dir else None
    config["comm_profile"] = str(args.comm_profile) if args.comm_profile else None
    config["compute_profile"] = str(args.compute_profile) if args.compute_profile else None
    config["profile_strict"] = args.profile_strict if args.comm_profile else None
    config["wait_budget_s"] = args.wait_budget_s
    config["rendezvous_startup_attempts"] = rendezvous_startup_attempts
    config["communication_adapter"] = ("raw-ordered-static-fifo" if args.comm_engine == "raw-ordered"
                                       else args.comm_engine)
    config["backend_ordering"] = ({
        "nccl_launch_order_implicit": "1",
        "contract_version": bare_order.contract_version,
        "default_order_digest": bare_order.digest,
    } if bare_order is not None else None)
    if args.dag:
        config["dag_input_hash"] = dag.input_hash
        config["estimate_view_hash"] = dag.estimate_view_hash
        config["execution_contract"] = ({
            "dependency_mode": "physical-completion",
            "compute_model": dag.execution.compute_model,
            "max_inflight": None if args.comm_engine == "bare" else 1,
            "communication_capacity": "not_admission_limited" if args.comm_engine == "bare" else 1,
            "communication_order": ("layered_topological_job_round_robin_projection"
                                    if args.comm_engine == "bare" else None),
            "communication_contract_version": (bare_order.contract_version
                                                if bare_order is not None else None),
            "communication_order_digest": bare_order.digest if bare_order is not None else None,
            "mode": dag.execution.mode,
        } if dag.execution.schema_version == 2 else None)
        config["static_order_sequence"] = list(order) if args.policy in {"static_fifo", "static_ltf"} else None
        config["static_order_digest"] = (
            hashlib.sha256(json.dumps(list(order), separators=(",", ":")).encode()).hexdigest()
            if args.policy in {"static_fifo", "static_ltf"} else None
        )
        config["bare_default_order_sequence"] = (
            list(bare_order.sequence) if bare_order is not None else None)
    config["device_mapping"] = ({
        "inherited_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "rank_to_logical_device": list(range(args.world_size)),
        "rank_to_device_uuid": [result.get("device_uuid") for result in results],
    } if args.backend == "nccl" else {"backend_device": "cpu"})
    config["binding_preparation"] = args.binding_preparation if not args.dag else "unchanged_dag"
    config["declaration_mode"] = args.declaration_mode if not args.dag else "unchanged_dag"
    if args.comm_profile:
        config["profile_digest"] = load_profile(args.comm_profile).digest()
    if args.compute_profile:
        from examples.jobpacer.gpu.gpu_compute_profile import load_gpu_compute_profile
        config["compute_profile_digest"] = load_gpu_compute_profile(args.compute_profile).digest
    git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=False)
    git_status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                                capture_output=True, text=True, check=False)
    config["code_revision"] = git_head.stdout.strip() if git_head.returncode == 0 else None
    config["working_tree_dirty"] = bool(git_status.stdout.strip())
    payload = {"config": config, "validation": validation, "metrics": metrics(results),
               "performance": performance(results), "ranks": results}
    if args.output:
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if validation["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
