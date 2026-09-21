"""Launch and collect the Phase 3 JobPacer runtime replay."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

from runtime_comm_scheduler.dag import build_static_order

try:
    from .runtime_results import expected_dag_results, metrics, validate_results
except ImportError:  # pragma: no cover - direct script execution
    from runtime_results import expected_dag_results, metrics, validate_results


HERE = Path(__file__).resolve().parent
WORKER = HERE / "runtime_worker.py"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _start(rank: int, args: argparse.Namespace, rendezvous_port: int, control_port: int) -> subprocess.Popen[str]:
    env = dict(os.environ)
    env.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(rendezvous_port), RANK=str(rank),
               WORLD_SIZE=str(args.world_size), LOCAL_RANK="0")
    env["PYTHONPATH"] = str(HERE.parents[1] / "src") + os.pathsep + env.get("PYTHONPATH", "")
    if args.backend == "nccl":
        env["CUDA_VISIBLE_DEVICES"] = str(rank)
    command = [sys.executable, str(WORKER), "--policy", args.policy, "--backend", args.backend,
               "--epoch", str(args.epoch), "--timeout", str(args.timeout),
               "--setup-timeout", str(args.setup_timeout),
               "--poll-interval", str(args.poll_interval), "--dag-poll-interval", str(args.dag_poll_interval),
               "--compute-jitter", str(args.compute_jitter), "--control-port", str(control_port),
               "--fault", args.fault]
    if args.dag:
        command.extend(("--dag", str(args.dag)))
    else:
        command.extend(("--workload", args.workload or "balanced"))
    if args.static_order:
        command.extend(("--static-order", str(args.static_order)))
    return subprocess.Popen(
        command,
        cwd=str(HERE.parents[1]), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("static_fifo", "static_ltf", "fifo", "ltf", "lookahead"), default="fifo")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--workload")
    source.add_argument("--dag", type=Path)
    parser.add_argument("--static-order", type=Path)
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--setup-timeout", type=float, default=20.0)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--poll-interval", type=float, default=0.001)
    parser.add_argument("--dag-poll-interval", type=float, default=0.001)
    parser.add_argument("--compute-jitter", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fault", choices=("none", "missing_task", "metadata_mismatch", "launch_failure",
                                              "completion_probe_failure", "compute_failure", "binding_failure"),
                        default="none")
    args = parser.parse_args()
    if args.timeout <= 0 or args.setup_timeout <= 0 or args.poll_interval <= 0 or args.dag_poll_interval <= 0:
        parser.error("setup/replay timeouts and poll intervals must be positive")
    if not 0 <= args.compute_jitter < 1:
        parser.error("compute-jitter must be in [0, 1)")
    if args.epoch < 0 or args.world_size <= 0:
        parser.error("epoch must be non-negative and world-size positive")

    expected: dict[str, dict[str, Any]] = {str(rank): {} for rank in range(args.world_size)}
    expected["__all__"] = {}
    expected["__groups__"] = {}
    expected_nodes: dict[int, set[str]] | None = None
    digests: set[str] | None = None
    if args.dag:
        try:
            from .runtime_adapter import load_dag, load_static_order
        except ImportError:  # pragma: no cover - direct script execution
            from runtime_adapter import load_dag, load_static_order
        try:
            dag = load_dag(args.dag, epoch=args.epoch, world_size=args.world_size)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        if args.static_order and args.policy not in {"static_fifo", "static_ltf"}:
            parser.error("--static-order requires a static policy")
        if args.static_order:
            try:
                load_static_order(args.static_order, dag.graph)
            except (OSError, ValueError) as exc:
                parser.error(str(exc))
        elif args.policy in {"static_fifo", "static_ltf"}:
            build_static_order(dag.graph, args.policy)
        expected_dag = expected_dag_results(dag.graph, args.world_size)
        expected = expected_dag["expected"]
        expected_nodes = expected_dag["expected_nodes"]
        digests = {dag.manifest_digest}
    else:
        try:
            from .workloads import load_workload, ranks_for_job
            from .runtime_adapter import all_specs
        except ImportError:  # pragma: no cover - direct script execution
            from workloads import load_workload, ranks_for_job
            from runtime_adapter import all_specs
        workload = load_workload(args.workload or "balanced")
        for job, _index, spec, _hint in all_specs(workload, epoch=args.epoch):
            meta = {"group_id": spec.group_id, "group_seq": spec.group_seq}
            expected["__all__"][spec.task_id] = meta
            expected["__groups__"].setdefault(spec.group_id, ranks_for_job(job, args.world_size))
            for rank in ranks_for_job(job, args.world_size):
                expected[str(rank)][spec.task_id] = meta
        if args.static_order:
            parser.error("--static-order requires --dag")

    rendezvous_port, control_port = _free_port(), _free_port()
    processes = [_start(rank, args, rendezvous_port, control_port) for rank in range(args.world_size)]
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(processes)) as pool:
        futures = [pool.submit(_collect, rank, process, args.setup_timeout + args.timeout + 5)
                   for rank, process in enumerate(processes)]
        for future in futures:
            result, error = future.result()
            if error:
                errors.append(error)
            elif result is not None:
                results.append(result)
    results.sort(key=lambda item: item.get("rank", -1))
    validation = validate_results(
        results, args.world_size, expected=expected, expected_nodes=expected_nodes, digests=digests,
        expected_config={
            "epoch": args.epoch, "world_size": args.world_size, "max_inflight": 1,
            "backend": args.backend, "completion_poll_interval_s": args.poll_interval,
            "dag_poll_interval_s": args.dag_poll_interval, "compute_jitter": args.compute_jitter,
        },
    )
    validation["errors"] = errors + validation["errors"]
    if errors:
        validation["status"] = "failed"
    config = vars(args).copy()
    config["output"] = str(args.output) if args.output else None
    config["dag"] = str(args.dag) if args.dag else None
    config["static_order"] = str(args.static_order) if args.static_order else None
    git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE.parents[1],
                              capture_output=True, text=True, check=False)
    git_status = subprocess.run(["git", "status", "--porcelain"], cwd=HERE.parents[1],
                                capture_output=True, text=True, check=False)
    config["code_revision"] = git_head.stdout.strip() if git_head.returncode == 0 else None
    config["working_tree_dirty"] = bool(git_status.stdout.strip())
    payload = {"config": config, "validation": validation, "metrics": metrics(results), "ranks": results}
    if args.output:
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if validation["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
