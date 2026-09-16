"""Launch the independent Stage 3.1 runtime replay."""

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


HERE = Path(__file__).resolve().parent
WORKER = HERE / "runtime_worker.py"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _start(rank: int, args: argparse.Namespace, rendezvous_port: int, control_port: int) -> subprocess.Popen[str]:
    env = dict(os.environ)
    env.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(rendezvous_port), RANK=str(rank), WORLD_SIZE=str(args.world_size), LOCAL_RANK="0")
    env["PYTHONPATH"] = str(HERE.parents[1] / "src") + os.pathsep + env.get("PYTHONPATH", "")
    if args.backend == "nccl":
        env["CUDA_VISIBLE_DEVICES"] = str(rank)
    return subprocess.Popen(
        [sys.executable, str(WORKER), "--policy", args.policy, "--workload", args.workload, "--backend", args.backend, "--timeout", str(args.timeout), "--control-port", str(control_port), "--fault", args.fault],
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
    parser.add_argument("--workload", default="balanced")
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fault", choices=("none", "missing_task", "metadata_mismatch"), default="none")
    args = parser.parse_args()
    rendezvous_port, control_port = _free_port(), _free_port()
    processes = [_start(rank, args, rendezvous_port, control_port) for rank in range(args.world_size)]
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(processes)) as pool:
        futures = [pool.submit(_collect, rank, process, args.timeout) for rank, process in enumerate(processes)]
        for future in futures:
            result, error = future.result()
            if error:
                errors.append(error)
            elif result is not None:
                results.append(result)
    results.sort(key=lambda item: item.get("rank", -1))
    task_sequences = [tuple(item.get("grant_sequence", ())) for item in results]
    all_correct = all(task["correct"] for result in results for job in result.get("jobs", []) for task in job.get("tasks", []))
    # The rank-local task sequence is a projection; group order is checked by
    # every member in the runtime itself through the shared group_seq gate.
    validation = {
        "status": "ok" if not errors and len(results) == args.world_size and all_correct else "failed",
        "all_collectives_correct": all_correct,
        "rank_count": len(results),
        "errors": errors,
        "rank_task_sequences": task_sequences,
    }
    config = vars(args).copy()
    config["output"] = str(args.output) if args.output else None
    payload = {"config": config, "validation": validation, "ranks": results}
    if args.output:
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if validation["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
