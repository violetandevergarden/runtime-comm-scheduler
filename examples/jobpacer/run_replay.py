"""Launch and summarize the two-rank JobPacer replay."""

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

try:  # Support ``python examples/jobpacer/run_replay.py``.
    from .comm_profile import apply_profile, load_profile
    from .plan_builder import build_plan, key_labels, policy_names
    from .workloads import load_workload, ranks_for_job
except ImportError:  # pragma: no cover - direct script execution
    from comm_profile import apply_profile, load_profile
    from plan_builder import build_plan, key_labels, policy_names
    from workloads import load_workload, ranks_for_job


HERE = Path(__file__).resolve().parent
WORKER = HERE / "replay_worker.py"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _run_rank(rank: int, args: argparse.Namespace, port: int) -> subprocess.Popen[str]:
    env = dict(os.environ)
    env.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(args.world_size),
        # CUDA_VISIBLE_DEVICES remaps the exposed physical GPU to cuda:0.
        LOCAL_RANK="0",
    )
    source_root = str(HERE.parents[1] / "src")
    env["PYTHONPATH"] = source_root + os.pathsep + env.get("PYTHONPATH", "")
    if args.backend == "nccl":
        env["CUDA_VISIBLE_DEVICES"] = str(rank)
    command = [
        sys.executable,
        str(WORKER),
        "--mode",
        args.mode,
        "--policy",
        args.policy,
        "--selection",
        args.selection,
        "--scenario",
        args.scenario or "",
        "--workload",
        args.workload,
        "--backend",
        args.backend,
        "--max-outstanding",
        str(args.max_outstanding),
        "--finish-timeout",
        str(args.timeout),
        "--thread-timeout",
        str(args.timeout),
        "--completion-poll-interval-s",
        str(args.completion_poll_interval_s),
        "--fault",
        args.fault,
    ]
    if args.comm_profile:
        command.extend(["--comm-profile", str(args.comm_profile)])
    command.append("--profile-strict" if args.profile_strict else "--no-profile-strict")
    return subprocess.Popen(
        command,
        env=env,
        cwd=str(HERE.parents[1]),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _collect(
    rank: int, process: subprocess.Popen[str], timeout: float
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        return None, f"rank {rank} timed out: {stderr[-2000:]}"
    if process.returncode != 0:
        return (
            None,
            f"rank {rank} exited {process.returncode}: {stderr[-3000:]} {stdout[-1000:]}",
        )
    try:
        return json.loads(stdout.strip().splitlines()[-1]), None
    except (IndexError, json.JSONDecodeError) as exc:
        return None, f"rank {rank} emitted invalid JSON ({exc}): {stdout[-2000:]}"


def _validate_results(
    results: list[dict[str, Any]], args: argparse.Namespace
) -> dict[str, Any]:
    workload = load_workload(args.workload)
    if args.comm_profile:
        workload = apply_profile(
            workload,
            load_profile(args.comm_profile),
            {"backend": args.backend, "device_type": "cuda" if args.backend == "nccl" else "cpu", "world_size": args.world_size},
            strict=args.profile_strict,
        )
    plan = build_plan(workload, args.policy)
    digests = {result["plan"]["digest"] for result in results}
    all_correct = all(
        task["correct"]
        for result in results
        for job in result.get("jobs", [])
        for task in job.get("tasks", [])
    ) and all(
        len(job.get("tasks", [])) == len(next(
            item for item in workload.jobs if item.job_id == job["job_id"]
        ).communications)
        for result in results
        for job in result.get("jobs", [])
    )
    boundary_ok = True
    for result in results:
        if result.get("trace_schema_version") == 2:
            boundary_ok &= (
                result["application_release_ts"]
                <= result["application_end_ts"]
                <= result["communication_drain_end_ts"]
                <= result["validation_end_ts"]
                <= result["harness_end_ts"]
            )
            boundary_ok &= all(
                task["completion_observed_ts"]
                <= result["communication_drain_end_ts"]
                and task["validation_start_ts"]
                >= result["communication_drain_end_ts"]
                for job in result.get("jobs", [])
                for task in job.get("tasks", [])
            )
    scheduler_order_ok = None
    group_order_ok = None
    serial_admission_ok = None
    if args.mode == "scheduler":
        scheduler_order_ok = True
        group_order_ok = True
        for result in results:
            local_jobs = {
                job.job_id
                for job in workload.jobs
                if result["rank"] in ranks_for_job(job, args.world_size)
            }
            expected_keys = [
                key.as_list()
                for key in plan.keys
                if key.process_group_id in local_jobs
            ]
            expected_groups = {
                group_id: [
                    key.as_list() for key in plan.group_sequence(group_id)
                ]
                for group_id in local_jobs
            }
            if args.selection != "ready_first":
                scheduler_order_ok &= result.get("launch_sequence") == expected_keys
            group_order_ok &= result.get("group_sequence") == expected_groups
        serial_admission_ok = (
            all(
                all(
                    next_timing["admit_ts"]
                    >= (
                        timing.get("completion_observed_ts")
                        or timing.get("complete_ts")
                    )
                    for timing, next_timing in zip(
                        sorted(result.get("timings", []), key=lambda item: item["admit_ts"]),
                        sorted(result.get("timings", []), key=lambda item: item["admit_ts"])[1:],
                    )
                )
                for result in results
            )
            if args.max_outstanding == 1
            else None
        )
    return {
        "status": (
            "ok"
            if len(digests) == 1
            and all_correct
            and boundary_ok
            and scheduler_order_ok is not False
            and group_order_ok is not False
            and serial_admission_ok is not False
            and (
                args.selection != "ready_first"
                or args.max_outstanding != 1
                or all(
                    result.get("control", {}).get("global_completion_barrier_count", 0)
                    >= max(0, len(result.get("launch_sequence", [])) - 1)
                    for result in results
                )
            )
            else "failed"
        ),
        "plan_digest_equal": len(digests) == 1,
        "all_collectives_correct": all_correct,
        "trace_boundaries_closed": boundary_ok,
        "scheduler_sequence_matches_plan": scheduler_order_ok,
        "group_sequences_match_plan": group_order_ok,
        "strict_serial_admission_verified": serial_admission_ok,
        "rank_local_serial_admission_verified": serial_admission_ok,
        "global_completion_barrier_before_next_selection": (
            all(
                result.get("control", {}).get("global_completion_barrier_before_next_selection", False)
                and result.get("control", {}).get("global_completion_barrier_count", 0)
                >= max(0, len(result.get("launch_sequence", [])) - 1)
                for result in results
            )
            if args.selection == "ready_first" and args.max_outstanding == 1
            else None
        ),
        "expected_plan_labels": key_labels(plan),
        "plan_digest": plan.digest(),
    }


def _summarize_performance(
    results: list[dict[str, Any]], args: argparse.Namespace
) -> dict[str, Any]:
    """Aggregate per-rank durations without comparing absolute rank clocks."""
    explicit_boundaries = any(
        "application_makespan_us" in result for result in results
    )
    workload = load_workload(args.workload)
    jobs = []
    for job in workload.jobs:
        records_by_rank: dict[int, dict[str, Any]] = {}
        for result in results:
            rank = int(result["rank"])
            for job_result in result.get("jobs", []):
                if job_result["job_id"] != job.job_id:
                    continue
                if rank in records_by_rank:
                    raise ValueError(
                        f"job {job.job_id!r} has duplicate rank record for rank {rank}"
                    )
                records_by_rank[rank] = job_result
        expected_ranks = ranks_for_job(job, args.world_size)
        if set(records_by_rank) != set(expected_ranks):
            raise ValueError(
                f"job {job.job_id!r} expected rank records {expected_ranks}, "
                f"got {tuple(sorted(records_by_rank))}"
            )
        rank_makespans_us = [
            records_by_rank[rank]["makespan_us"] for rank in expected_ranks
        ]
        job_record = {
            "job_id": job.job_id,
            "participating_ranks": list(expected_ranks),
            "makespan_us": max(rank_makespans_us),
            "rank_makespans_us": rank_makespans_us,
            "tasks_per_rank": len(job.communications),
        }
        if explicit_boundaries:
            application_values = [
                records_by_rank[rank].get(
                    "application_makespan_us", records_by_rank[rank]["makespan_us"]
                )
                for rank in expected_ranks
            ]
            job_record["application_makespan_us"] = max(application_values)
            job_record["rank_application_makespans_us"] = application_values
            job_record["makespan_us"] = job_record["application_makespan_us"]
        jobs.append(job_record)
    summary = {
        "job_makespans": jobs,
        "workload_makespan_us": max(
            result.get("application_makespan_us", result["replay_makespan_us"])
            for result in results
        ),
        "rank_replay_makespans_us": [
            result.get("application_makespan_us", result["replay_makespan_us"])
            for result in sorted(results, key=lambda item: item["rank"])
        ],
    }
    if explicit_boundaries:
        summary.update(
            {
                "application_makespan_us": summary["workload_makespan_us"],
                "rank_application_makespans_us": summary[
                    "rank_replay_makespans_us"
                ],
                "communication_drain_makespan_us": max(
                    result.get(
                        "communication_drain_makespan_us",
                        result.get("application_makespan_us", 0),
                    )
                    for result in results
                ),
                "rank_communication_drain_makespans_us": [
                    result.get(
                        "communication_drain_makespan_us",
                        result.get("application_makespan_us", 0),
                    )
                    for result in sorted(results, key=lambda item: item["rank"])
                ],
                "validation_total_us": max(
                    result.get("validation_total_us", 0) for result in results
                ),
                "harness_total_us": max(
                    result.get("harness_total_us", 0) for result in results
                ),
            }
        )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("bare", "scheduler"), default="scheduler")
    parser.add_argument("--policy", choices=policy_names(), default="fifo")
    parser.add_argument(
        "--selection",
        choices=("runtime_arrival", "ready_first"),
        default="runtime_arrival",
    )
    parser.add_argument("--scenario")
    parser.add_argument("--workload", default="balanced")
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--max-outstanding", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--completion-poll-interval-s", type=float, default=0.001)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--comm-profile", type=Path)
    parser.add_argument("--profile-strict", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--fault", choices=("none", "missing_key", "metadata_mismatch"), default="none"
    )
    args = parser.parse_args(argv)
    if Path(args.workload).exists():
        args.workload = str(Path(args.workload).resolve())
    if args.comm_profile:
        args.comm_profile = args.comm_profile.resolve()
    if args.world_size < 2:
        parser.error("--world-size must be at least 2")
    workload = load_workload(args.workload)
    profile = None
    if args.comm_profile:
        profile = load_profile(args.comm_profile)
        apply_profile(
            workload,
            profile,
            {"backend": args.backend, "device_type": "cuda" if args.backend == "nccl" else "cpu", "world_size": args.world_size},
            strict=args.profile_strict,
        )
    port = _free_port()
    processes = [_run_rank(rank, args, port) for rank in range(args.world_size)]
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(processes)) as pool:
        futures = [
            pool.submit(_collect, rank, process, args.timeout)
            for rank, process in enumerate(processes)
        ]
        for future in futures:
            result, error = future.result()
            if error:
                errors.append(error)
            elif result is not None:
                results.append(result)
    if errors or len(results) != args.world_size:
        payload = {"status": "failed", "errors": errors, "results": results}
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1
    results.sort(key=lambda result: result["rank"])
    validation = _validate_results(results, args)
    performance = _summarize_performance(results, args)
    payload = {
        "trace_schema_version": max(
            result.get("trace_schema_version", 0) for result in results
        ),
        "config": {
            "mode": args.mode,
            "policy": args.policy,
            "selection": args.selection,
            "scenario": args.scenario,
            "workload": args.workload,
            "backend": args.backend,
            "world_size": args.world_size,
            "max_outstanding": args.max_outstanding,
            "completion_poll_interval_s": args.completion_poll_interval_s,
            "timeout_s": args.timeout,
            "torch": __import__("torch").__version__,
            "estimate_source": "offline_profile" if profile else "manifest",
            "comm_profile": str(args.comm_profile) if args.comm_profile else None,
            "profile_digest": profile.digest() if profile else None,
            "profile_schema_version": profile.schema_version if profile else None,
            "profile_strict": args.profile_strict if profile else None,
            "profile_environment": dict(profile.environment) if profile else None,
        },
        "validation": validation,
        "performance": performance,
        "ranks": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(validation, indent=2, sort_keys=True))
    return 0 if validation["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
