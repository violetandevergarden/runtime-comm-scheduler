#!/usr/bin/env python3
"""Run and summarize the Phase 1/2 Gloo comparison benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import socket
import subprocess
import sys
from shutil import copyfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import median, quantiles

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from examples.jobpacer.paths import repository_path, resolve_migrated_path

BENCHMARK_DIR = ROOT / "benchmark/phase1.2"
WORKLOAD_DIR = BENCHMARK_DIR / "experiments/shared/workloads"
RESULTS_DIR = BENCHMARK_DIR / "results"
PHASE12_WORKLOADS = (
    "comm-heavy",
    "mixed-message-sizes-large-first",
    "mixed-message-sizes-large-last",
    "long-short-chains",
    "staggered-compute",
    "overlap-window-short",
    "overlap-window-medium",
    "overlap-window-long",
)
SCENARIOS = {
    "phase1_bare": {"policy": "fifo", "selection": "runtime_arrival", "max_outstanding": 0, "mode": "bare"},
    "ready_first_unbounded": {"policy": "fifo", "selection": "ready_first", "max_outstanding": 0, "mode": "scheduler"},
    "ready_first_serial": {"policy": "fifo", "selection": "ready_first", "max_outstanding": 1, "mode": "scheduler"},
    "fifo_unbounded": {"policy": "fifo", "selection": "runtime_arrival", "max_outstanding": 0, "mode": "scheduler"},
    "fifo_serial": {"policy": "fifo", "selection": "runtime_arrival", "max_outstanding": 1, "mode": "scheduler"},
    "ltf_serial": {"policy": "ltf", "selection": "runtime_arrival", "max_outstanding": 1, "mode": "scheduler"},
}

from examples.jobpacer.gloo.comm_profile import apply_profile
from examples.jobpacer.runtime.comm_profile import load_profile
from examples.jobpacer.runtime.plan_builder import build_plan, policy_diagnostics
from examples.jobpacer.gloo.workloads import load_workload


def deterministic_scenario_order(
    scenarios: list[str] | tuple[str, ...], seed: int, repetition: int
) -> list[str]:
    """Return one reproducible permutation for a repetition round."""
    order = list(scenarios)
    random.Random(f"{seed}:{repetition}").shuffle(order)
    return order


def selected_workloads(names: list[str] | None) -> list[Path]:
    available = {name: WORKLOAD_DIR / f"{name}.json" for name in PHASE12_WORKLOADS}
    if not names:
        return [available[name] for name in PHASE12_WORKLOADS]
    unknown = sorted(set(names) - available.keys())
    if unknown:
        raise ValueError(
            f"unknown workload(s): {', '.join(unknown)}; choices={', '.join(sorted(available))}"
        )
    return [available[name] for name in dict.fromkeys(names)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_digest(path: Path) -> str:
    value = json.loads(path.read_text())
    canonical = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _batch_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + __import__("secrets").token_hex(3)


def _git_metadata() -> dict[str, object]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"git_commit": commit, "git_worktree_dirty": dirty}


def run(command: list[str]) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(ROOT), env.get("PYTHONPATH", "")))
    completed = subprocess.run(
        command, cwd=ROOT, env=env, text=True, capture_output=True, timeout=120
    )
    if completed.returncode:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stdout}\n{completed.stderr}"
        )


def stats(values: list[int]) -> dict[str, float]:
    p10, *_, p90 = quantiles(values, n=10, method="inclusive")
    return {
        "p10": p10,
        "median": median(values),
        "p90": p90,
        "min": min(values),
        "max": max(values),
    }


def sequence_label(key: list[object]) -> str:
    return f"{key[3]}:{key[6]}"


def summarize(paths: list[Path]) -> dict[str, object]:
    traces = [json.loads(path.read_text()) for path in paths]
    if not all(trace["validation"]["status"] == "ok" for trace in traces):
        raise RuntimeError(f"validation failed in {[str(path) for path in paths]}")

    job_ids = [item["job_id"] for item in traces[0]["performance"]["job_makespans"]]
    sequences = Counter(
        tuple(sequence_label(key) for key in trace["ranks"][0]["launch_sequence"])
        for trace in traces
    )
    rank0_overlap_runs = 0
    static_plan_matches = 0
    group_plan_matches = 0
    first_job1_admission_waits = []
    scheduler_states = []
    from examples.jobpacer.analysis.visualize import scheduler_state_summary

    for trace in traces:
        rank0 = next(rank for rank in trace["ranks"] if rank["rank"] == 0)
        if (
            trace["config"]["mode"] == "scheduler"
            and trace["config"]["selection"] != "ready_first"
            and trace["validation"].get("scheduler_sequence_matches_plan") is True
        ):
            static_plan_matches += 1
        if (
            trace["config"]["mode"] == "scheduler"
            and trace["validation"].get("group_sequences_match_plan") is True
        ):
            group_plan_matches += 1
        communications = [
            task
            for job in rank0["jobs"]
            for task in job["tasks"]
        ]
        if any(
            left["key"][3] != right["key"][3]
            and left["collective_call_start_ts"] < right["completion_observed_ts"]
            and right["collective_call_start_ts"] < left["completion_observed_ts"]
            for index, left in enumerate(communications)
            for right in communications[index + 1 :]
        ):
            rank0_overlap_runs += 1
        job1_first = next(
            (task for job in rank0["jobs"] if job["job_id"] == "job-1"
             for task in job["tasks"] if task["ordinal"] == 0),
            None,
        )
        if job1_first is not None:
            first_job1_admission_waits.append(
                job1_first["admit_ts"] - job1_first["ready_record_ts"]
            )
        if trace["config"]["mode"] == "scheduler":
            scheduler_states.append(scheduler_state_summary(rank0))
    duration_fields = (
        "submit_api_duration_us",
        "collective_call_duration_us",
        "post_return_completion_observation_us",
        "call_to_completion_observation_us",
        "deferred_binding_wait_us",
        "underlying_wait_duration_us",
        "validation_duration_us",
        "ready_to_job_end_observed_us",
        "completion_to_job_end_observed_us",
    )
    task_durations = {}
    for field in duration_fields:
        samples = [
            task[field]
            for trace in traces
            for rank in trace["ranks"]
            for job in rank["jobs"]
            for task in job["tasks"]
            if task.get(field) is not None
        ]
        if samples:
            task_durations[field] = stats(samples)
    run_boundary_fields = (
        "application_makespan_us",
        "communication_drain_makespan_us",
        "validation_total_us",
        "harness_total_us",
    )
    run_boundaries = {
        field: stats(
            [trace["performance"][field] for trace in traces]
        )
        for field in run_boundary_fields
        if field in traces[0]["performance"]
    }
    return {
        "runs": len(traces),
        "all_valid": True,
        "workload_makespan_us": stats(
            [trace["performance"]["workload_makespan_us"] for trace in traces]
        ),
        "job_makespan_us": {
            job_id: stats(
                [
                    next(
                        item["makespan_us"]
                        for item in trace["performance"]["job_makespans"]
                        if item["job_id"] == job_id
                    )
                    for trace in traces
                ]
            )
            for job_id in job_ids
        },
        "rank0_launch_sequences": [
            {"sequence": list(sequence), "count": count}
            for sequence, count in sequences.most_common()
        ],
        "rank0_distinct_launch_sequence_count": len(sequences),
        "static_plan_sequence_match_runs": static_plan_matches
        if traces[0]["config"]["mode"] == "scheduler"
        and traces[0]["config"]["selection"] != "ready_first"
        else None,
        "group_sequence_plan_match_runs": group_plan_matches
        if traces[0]["config"]["mode"] == "scheduler"
        else None,
        "rank0_cross_group_overlap_runs": rank0_overlap_runs,
        "rank0_job1_first_ready_to_admit_us": (
            stats(first_job1_admission_waits) if first_job1_admission_waits else None
        ),
        "rank0_scheduler_state_us": {
            name: stats([state[name] for state in scheduler_states])
            for name in (
                "work_conserving_idle_us",
                "head_of_line_idle_us",
                "capacity_wait_us",
                "capacity_occupied_us",
                "scheduler_delay_us",
                "coordination_wait_us",
                "pending_launch_us",
                "inflight_us",
                "no_ready_work_us",
                "unattributed_wait_us",
            )
        }
        if scheduler_states
        else None,
        "task_duration_us": task_durations,
        "run_boundary_us": run_boundaries,
        "expected_plan": traces[0]["validation"]["expected_plan_labels"],
        "strict_serial_admission_verified": all(
            trace["validation"]["strict_serial_admission_verified"] is True
            for trace in traces
        )
        if traces[0]["config"]["mode"] == "scheduler"
        and traces[0]["config"]["max_outstanding"] == 1
        else None,
        "rank_local_serial_admission_verified": all(
            trace["validation"]["rank_local_serial_admission_verified"] is True
            for trace in traces
        )
        if traces[0]["config"]["mode"] == "scheduler"
        and traces[0]["config"]["max_outstanding"] == 1
        else None,
        "global_completion_barrier_before_next_selection": all(
            trace["validation"]["global_completion_barrier_before_next_selection"] is True
            for trace in traces
        )
        if traces[0]["config"]["selection"] == "ready_first"
        and traces[0]["config"]["max_outstanding"] == 1
        else None,
        "rank_local_serial_admission_passes": sum(
            trace["validation"].get("rank_local_serial_admission_verified") is True
            for trace in traces
        )
        if traces[0]["config"]["mode"] == "scheduler"
        and traces[0]["config"]["max_outstanding"] == 1
        else None,
        "global_completion_barrier_passes": sum(
            trace["validation"].get("global_completion_barrier_before_next_selection") is True
            for trace in traces
        )
        if traces[0]["config"]["selection"] == "ready_first"
        and traces[0]["config"]["max_outstanding"] == 1
        else None,
    }


def percent_change(value: float, baseline: float) -> float:
    return (value / baseline - 1) * 100


def run_poll_sensitivity(profile_source: Path, repeats: int = 3) -> Path:
    """Compare Gloo completion-probe resolution without mixing the main matrix."""
    workload = WORKLOAD_DIR / "balanced-small.json"
    profile_source = profile_source.resolve()
    batch_root = RESULTS_DIR / "polling" / f"{_batch_id()}-poll-sensitivity"
    raw_dir = batch_root / "raw" / "balanced-small"
    profile_dir = batch_root / "profiles"
    raw_dir.mkdir(parents=True)
    profile_dir.mkdir()
    profile = profile_dir / "balanced-small-gloo.json"
    copyfile(profile_source, profile)
    scenarios = {"poll_1ms": 0.001, "poll_0p1ms": 0.0001}
    run_records: list[dict[str, object]] = []
    manifest = {
        "manifest_schema_version": 1,
        "trace_schema_version": 2,
        "complete": False,
        "batch_id": batch_root.name,
        "experiment_seed": 220914,
        "repetitions": repeats,
        "scenarios": scenarios,
        "runs": run_records,
        "metadata": {
            "backend": "gloo",
            "world_size": 2,
            "workload_digest": _sha256(workload),
            "profile_digest": _canonical_json_digest(profile),
            "profile_sha256": _sha256(profile),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "command_line": [sys.executable, *sys.argv],
            **_git_metadata(),
        },
    }

    def write_manifest() -> None:
        temporary = batch_root / "manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        temporary.replace(batch_root / "manifest.json")

    write_manifest()
    try:
        for repetition in range(repeats):
            order = deterministic_scenario_order(tuple(scenarios), 220914, repetition)
            for order_index, scenario in enumerate(order):
                interval = scenarios[scenario]
                output = raw_dir / scenario / f"run-{repetition:02d}.json"
                command = [
                    sys.executable,
                    str(ROOT / "examples/jobpacer/scripts/run_phase1.py"),
                    "--workload", str(workload),
                    "--backend", "gloo",
                    "--world-size", "2",
                    "--timeout", "30",
                    "--completion-poll-interval-s", str(interval),
                    "--comm-profile", str(profile),
                    "--output", str(output),
                ]
                record = {
                    "workload": "balanced-small",
                    "scenario": scenario,
                    "repetition": repetition,
                    "order_index": order_index,
                    "path": str(output.relative_to(batch_root)),
                    "workload_digest": manifest["metadata"]["workload_digest"],
                    "profile_path": str(profile.relative_to(batch_root)),
                    "profile_digest": manifest["metadata"]["profile_digest"],
                    "profile_sha256": manifest["metadata"]["profile_sha256"],
                    "status": "running",
                    "command": command,
                    "started_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                run_records.append(record)
                write_manifest()
                run(command)
                trace = json.loads(output.read_text())
                if trace.get("validation", {}).get("status") != "ok":
                    raise RuntimeError(f"poll sensitivity validation failed: {output}")
                record.update(
                    status="ok",
                    sha256=_sha256(output),
                    ended_at_utc=datetime.now(timezone.utc).isoformat(),
                )
                write_manifest()

        samples = {}
        by_repetition = {}
        for scenario in scenarios:
            paths = [batch_root / record["path"] for record in run_records if record["scenario"] == scenario]
            traces = [json.loads(path.read_text()) for path in paths]
            makespans = [trace["performance"]["workload_makespan_us"] for trace in traces]
            small_message_observations = [
                task["call_to_completion_observation_us"]
                for trace in traces
                for rank in trace["ranks"]
                for job in rank["jobs"]
                for task in job["tasks"]
                if task["num_bytes"] == 4096
            ]
            samples[scenario] = {
                "workload_makespan_us": stats(makespans),
                "4k_call_to_completion_observation_us": stats(small_message_observations),
                "runs": len(traces),
            }
            by_repetition[scenario] = {
                record["repetition"]: json.loads((batch_root / record["path"]).read_text())[
                    "performance"]["workload_makespan_us"]
                for record in run_records if record["scenario"] == scenario
            }
        differences = [
            by_repetition["poll_0p1ms"][index] - by_repetition["poll_1ms"][index]
            for index in range(repeats)
        ]
        summary = {
            "batch_id": batch_root.name,
            "scenarios": samples,
            "paired_workload_difference_0p1ms_minus_1ms_us": differences,
            "paired_difference_distribution_us": stats(differences),
        }
        (batch_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        (batch_root / "analysis.md").write_text(
            "# Gloo completion-poll sensitivity\n\n"
            f"Balanced-small, {repeats} paired repetitions; same profile and workload. "
            "Differences smaller than the 1 ms poll interval are resolution-sensitive; "
            "completion timestamps remain probe observations. These exploratory samples "
            "are not confidence intervals.\n\n"
            f"- 1 ms: workload median {samples['poll_1ms']['workload_makespan_us']['median'] / 1000:.3f} ms; "
            f"4 KiB call-to-observation median {samples['poll_1ms']['4k_call_to_completion_observation_us']['median'] / 1000:.3f} ms.\n"
            f"- 0.1 ms: workload median {samples['poll_0p1ms']['workload_makespan_us']['median'] / 1000:.3f} ms; "
            f"4 KiB call-to-observation median {samples['poll_0p1ms']['4k_call_to_completion_observation_us']['median'] / 1000:.3f} ms.\n"
            f"- Paired workload difference median: {summary['paired_difference_distribution_us']['median'] / 1000:.3f} ms.\n",
            encoding="utf-8",
        )
        manifest["summary_path"] = "summary.json"
        manifest["analysis_path"] = "analysis.md"
        manifest["complete"] = True
        write_manifest()
    except BaseException:
        write_manifest()
        raise
    return batch_root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--experiment-seed", type=int)
    parser.add_argument("--completion-poll-interval-s", type=float, default=0.001)
    parser.add_argument(
        "--workload",
        action="append",
        help="workload stem to run; repeat the option to select multiple (default: all)",
    )
    parser.add_argument(
        "--finalize-batch",
        type=Path,
        help="rebuild reports from a complete, hash-verified batch",
    )
    parser.add_argument("--poll-sensitivity-profile", type=Path)
    parser.add_argument(
        "--poll-sensitivity-source-batch",
        type=Path,
        help="run an isolated 0.1 ms sensitivity batch using a completed capacity-scan batch",
    )
    parser.add_argument("--poll-sensitivity-workload", default="overlap-window-long")
    parser.add_argument(
        "--poll-sensitivity-pair",
        action="append",
        metavar="CURRENT:BASELINE",
        help="paired scenario comparison for sensitivity; repeat as needed",
    )
    parser.add_argument("--experiment", choices=("capacity-scan", "srjf"))
    parser.add_argument("--capacity-selection", type=Path)
    args = parser.parse_args()
    for name in ("finalize_batch", "poll_sensitivity_profile", "poll_sensitivity_source_batch", "capacity_selection"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, resolve_migrated_path(
                repository_path(value), RESULTS_DIR / "migration-map.json"))
    if args.poll_sensitivity_source_batch:
        if args.finalize_batch or args.poll_sensitivity_profile or args.experiment or args.capacity_selection:
            parser.error("--poll-sensitivity-source-batch cannot be combined with other experiment modes")
        pairs = []
        for pair in args.poll_sensitivity_pair or (
            "fifo_k2:fifo_k1",
            "fifo_k3:fifo_k1",
            "fifo_unbounded:fifo_k1",
            "fifo_k1:fifo_unbounded",
            "fifo_k2:fifo_unbounded",
            "fifo_k3:fifo_unbounded",
            "ltf_k2:ltf_k1",
            "ltf_k3:ltf_k1",
            "ltf_unbounded:ltf_k1",
            "ltf_k1:ltf_unbounded",
            "ltf_k2:ltf_unbounded",
            "ltf_k3:ltf_unbounded",
        ):
            current, separator, baseline = pair.partition(":")
            if not separator or not current or not baseline:
                parser.error(f"invalid --poll-sensitivity-pair {pair!r}; expected CURRENT:BASELINE")
            pairs.append((current, baseline))
        try:
            from examples.jobpacer.experiments.runner_batch import run_poll_sensitivity_batch

            batch = run_poll_sensitivity_batch(
                source_batch=args.poll_sensitivity_source_batch,
                workload_name=args.poll_sensitivity_workload,
                pairs=pairs,
                repeats=args.repeats,
                seed=args.experiment_seed,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            parser.error(str(exc))
        print(batch)
        return 0
    if args.poll_sensitivity_profile:
        if args.finalize_batch:
            parser.error("--poll-sensitivity-profile cannot be combined with --finalize-batch")
        if not args.poll_sensitivity_profile.is_file():
            parser.error(f"profile not found: {args.poll_sensitivity_profile}")
        batch = run_poll_sensitivity(args.poll_sensitivity_profile)
        print(batch)
        return 0
    matrix_experiment = args.experiment
    if args.finalize_batch and matrix_experiment is None:
        manifest_file = args.finalize_batch / "manifest.json"
        if manifest_file.is_file():
            recorded_experiment = json.loads(manifest_file.read_text()).get("experiment")
            if recorded_experiment in {"capacity-scan", "srjf"}:
                matrix_experiment = recorded_experiment
    if matrix_experiment or args.capacity_selection:
        if not matrix_experiment:
            parser.error("--capacity-selection requires --experiment srjf")
        try:
            from examples.jobpacer.experiments.runner_batch import run_experiment

            batch = run_experiment(
                experiment=matrix_experiment,
                repeats=args.repeats,
                seed=args.experiment_seed,
                poll_interval_s=args.completion_poll_interval_s,
                workload_names=args.workload,
                capacity_selection=args.capacity_selection,
                finalize_batch=args.finalize_batch,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            parser.error(str(exc))
        print(batch)
        return 0
    args.repeats = args.repeats if args.repeats is not None else 10
    args.experiment_seed = args.experiment_seed if args.experiment_seed is not None else 120914
    if args.repeats < 2 or args.completion_poll_interval_s <= 0:
        parser.error("--repeats must be >= 2 and completion poll interval > 0")

    try:
        requested_workloads = selected_workloads(args.workload)
    except ValueError as exc:
        parser.error(str(exc))
    finalize_batch = args.finalize_batch is not None
    if finalize_batch:
        batch_root = args.finalize_batch.resolve()
        batches_root = (RESULTS_DIR / "baseline").resolve()
        if not batch_root.is_relative_to(batches_root):
            parser.error("--finalize-batch must point inside benchmark/phase1.2/results/baseline")
        manifest_path = batch_root / "manifest.json"
        if not manifest_path.is_file():
            parser.error(f"batch manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("repetitions") != args.repeats:
            parser.error("--repeats must match the batch")
        if manifest.get("experiment_seed") != args.experiment_seed:
            parser.error("--experiment-seed must match the batch")
        records = manifest.get("runs", [])
        recorded_names = manifest.get("workloads") or sorted(
            {item.get("workload") for item in records}
        )
        workload_paths = [WORKLOAD_DIR / f"{name}.json" for name in recorded_names]
        if args.workload and [path.stem for path in requested_workloads] != recorded_names:
            parser.error("--workload selection must match the batch manifest")
        expected_runs = len(workload_paths) * args.repeats * len(SCENARIOS)
        if len(records) != expected_runs or any(item.get("status") != "ok" for item in records):
            parser.error("finalization requires every expected run to be present and status=ok")
        expected_keys = {
            (path.stem, repetition, scenario)
            for path in workload_paths
            for repetition in range(args.repeats)
            for scenario in SCENARIOS
        }
        observed_keys = {
            (item.get("workload"), item.get("repetition"), item.get("scenario"))
            for item in records
        }
        if observed_keys != expected_keys:
            parser.error("batch manifest has missing or duplicate scenario runs")
        run_records = records
    else:
        batch_root = RESULTS_DIR / "baseline" / _batch_id()
    raw_dir = batch_root / "raw"
    profile_dir = batch_root / "profiles"
    if not finalize_batch:
        raw_dir.mkdir(parents=True)
        profile_dir.mkdir()
    if not finalize_batch:
        workload_paths = requested_workloads
    if not finalize_batch:
        run_records = []
        manifest = {
            "manifest_schema_version": 1,
            "trace_schema_version": 2,
            "complete": False,
            "batch_id": batch_root.name,
            "experiment_seed": args.experiment_seed,
            "command_line": [sys.executable, *sys.argv],
            "repetitions": args.repeats,
            "workloads": [path.stem for path in workload_paths],
            "scenarios": SCENARIOS,
            "runs": run_records,
            "metadata": {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "host": socket.gethostname(),
                "platform": platform.platform(),
                "python": platform.python_version(),
                "torch": __import__("torch").__version__,
                "backend": "gloo",
                "world_size": 2,
                "completion_poll_interval_s": args.completion_poll_interval_s,
                "source_sha256": {
                    str(path.relative_to(ROOT)): _sha256(path)
                    for path in (
                        Path(__file__),
                        ROOT / "examples/jobpacer/runtime/replay_worker.py",
                        ROOT / "examples/jobpacer/runtime/plan_builder.py",
                        ROOT / "examples/jobpacer/analysis/visualize.py",
                        ROOT / "src/runtime_comm_scheduler/scheduler.py",
                    )
                },
                **_git_metadata(),
            },
        }

    def write_manifest() -> None:
        temporary = batch_root / "manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        temporary.replace(batch_root / "manifest.json")

    if not finalize_batch:
        write_manifest()
    active_run_record = None
    try:
        for workload in workload_paths:
            name = workload.stem
            profile_name = "overlap-window-gloo.json" if name.startswith("overlap-window-") else f"{name}-gloo.json"
            profile = profile_dir / profile_name
            if not finalize_batch:
                if not profile.exists():
                    print(f"profiling {name}", flush=True)
                    run(
                        [
                            sys.executable,
                            str(ROOT / "examples/jobpacer/scripts/run_comm_profile.py"),
                            "--workload", str(workload), "--backend", "gloo", "--world-size", "2",
                            "--warmup", "3", "--iterations", "10", "--timeout", "30", "--output", str(profile),
                        ]
                    )
                else:
                    print(f"reusing profile {profile.name} for {name}", flush=True)
            workload_digest = _sha256(workload)
            profile_digest = _canonical_json_digest(profile)
            profile_sha256 = _sha256(profile)
            if finalize_batch:
                matching = [item for item in run_records if item["workload"] == name]
                if any(
                    item.get("workload_digest") != workload_digest
                    or item.get("profile_digest") != profile_digest
                    or item.get("profile_sha256") != profile_sha256
                    or _sha256(batch_root / item["path"]) != item.get("sha256")
                    for item in matching
                ):
                    parser.error(f"workload/profile/trace digest mismatch for {name}")
            for repetition in range(0 if finalize_batch else args.repeats):
                order = deterministic_scenario_order(tuple(SCENARIOS), args.experiment_seed, repetition)
                for order_index, scenario in enumerate(order):
                    definition = SCENARIOS[scenario]
                    scenario_dir = raw_dir / name / scenario
                    scenario_dir.mkdir(parents=True, exist_ok=True)
                    output = scenario_dir / f"run-{repetition:02d}.json"
                    entrypoint = ROOT / (
                        "examples/jobpacer/scripts/run_phase1.py"
                        if definition["mode"] == "bare"
                        else "examples/jobpacer/scripts/run_phase2.py"
                    )
                    command = [
                        sys.executable, str(entrypoint), "--policy", definition["policy"],
                        "--selection", definition["selection"], "--scenario", scenario,
                        "--workload", str(workload), "--backend", "gloo", "--world-size", "2",
                        "--timeout", "30", "--comm-profile", str(profile), "--output", str(output),
                        "--completion-poll-interval-s", str(args.completion_poll_interval_s),
                    ]
                    if definition["mode"] == "scheduler":
                        command.extend(["--mode", "scheduler"])
                    command.extend(["--max-outstanding", str(definition["max_outstanding"])])
                    print(f"{name} repetition={repetition} order={order_index} {scenario}", flush=True)
                    started_at = datetime.now(timezone.utc).isoformat()
                    run_record = {
                        "workload": name,
                        "scenario": scenario,
                        "repetition": repetition,
                        "order_index": order_index,
                        "path": str(output.relative_to(batch_root)),
                        "status": "running",
                        "workload_digest": workload_digest,
                        "profile_path": str(profile.relative_to(batch_root)),
                        "profile_digest": profile_digest,
                        "profile_sha256": profile_sha256,
                        "command": command,
                        "started_at_utc": started_at,
                    }
                    run_records.append(run_record)
                    active_run_record = run_record
                    write_manifest()
                    run(command)
                    ended_at = datetime.now(timezone.utc).isoformat()
                    trace = json.loads(output.read_text())
                    if trace.get("validation", {}).get("status") != "ok":
                        raise RuntimeError(f"validation failed: {output}")
                    run_record.update(
                        status="ok",
                        sha256=_sha256(output),
                        ended_at_utc=ended_at,
                    )
                    active_run_record = None
                    write_manifest()

        plans = {}
        for workload_path in workload_paths:
            name = workload_path.stem
            profile_name = "overlap-window-gloo.json" if name.startswith("overlap-window-") else f"{name}-gloo.json"
            workload = apply_profile(
                load_workload(workload_path),
                load_profile(profile_dir / profile_name),
                {"backend": "gloo", "device_type": "cpu", "world_size": 2},
            )
            fifo = build_plan(workload, "fifo")
            ltf = build_plan(workload, "ltf")
            plans[name] = {
                "task_count": len(fifo.keys),
                "job_count": len(workload.jobs),
                "rank_tensor_bytes": sum(
                    communication.num_bytes
                    for job in workload.jobs
                    for communication in job.communications
                ),
                "fifo_keys": [key.as_list() for key in fifo.keys],
                "ltf_keys": [key.as_list() for key in ltf.keys],
                "same_plan": fifo.keys == ltf.keys,
                "ltf_diagnostics": policy_diagnostics(workload, "ltf"),
            }
        (batch_root / "plans.json").write_text(
            json.dumps(plans, indent=2, sort_keys=True) + "\n"
        )
        manifest["plans_path"] = "plans.json"

        summaries: dict[str, object] = {}
        for workload in workload_paths:
            name = workload.stem
            scenario_summaries = {
                scenario: summarize(
                    [
                        batch_root / record["path"]
                        for record in run_records
                        if record["workload"] == name and record["scenario"] == scenario
                    ]
                )
                for scenario in SCENARIOS
            }
            baseline = scenario_summaries["phase1_bare"]
            pair_definitions = (
                ("ready_first_unbounded", "phase1_bare"),
                ("ready_first_serial", "ready_first_unbounded"),
                ("fifo_serial", "ready_first_serial"),
                ("ltf_serial", "fifo_serial"),
                ("fifo_serial", "fifo_unbounded"),
            )
            paired_comparisons = {}
            paired_job_comparisons = {}
            for current, baseline_scenario in pair_definitions:
                current_traces = {
                    record["repetition"]: json.loads((batch_root / record["path"]).read_text())
                    for record in run_records
                    if record["workload"] == name and record["scenario"] == current
                }
                baseline_traces = {
                    record["repetition"]: json.loads((batch_root / record["path"]).read_text())
                    for record in run_records
                    if record["workload"] == name and record["scenario"] == baseline_scenario
                }
                repetitions = sorted(set(current_traces) & set(baseline_traces))
                differences = [
                    current_traces[index]["performance"]["workload_makespan_us"]
                    - baseline_traces[index]["performance"]["workload_makespan_us"]
                    for index in repetitions
                ]
                ratios = [
                    current_traces[index]["performance"]["workload_makespan_us"]
                    / baseline_traces[index]["performance"]["workload_makespan_us"]
                    for index in repetitions
                ]
                paired_comparisons[f"{current}_minus_{baseline_scenario}"] = {
                    "repetitions": repetitions,
                    "difference_us": differences,
                    "difference_distribution": stats(differences),
                    "ratio": ratios,
                    "ratio_distribution": stats(ratios),
                }
                paired_job_comparisons[f"{current}_minus_{baseline_scenario}"] = {
                    job_id: {
                        "difference_us": [
                            next(
                                item["makespan_us"]
                                for item in current_traces[index]["performance"]["job_makespans"]
                                if item["job_id"] == job_id
                            )
                            - next(
                                item["makespan_us"]
                                for item in baseline_traces[index]["performance"]["job_makespans"]
                                if item["job_id"] == job_id
                            )
                            for index in repetitions
                        ],
                        "difference_distribution": stats([
                            next(
                                item["makespan_us"]
                                for item in current_traces[index]["performance"]["job_makespans"]
                                if item["job_id"] == job_id
                            )
                            - next(
                                item["makespan_us"]
                                for item in baseline_traces[index]["performance"]["job_makespans"]
                                if item["job_id"] == job_id
                            )
                            for index in repetitions
                        ]),
                        "repetitions": repetitions,
                    }
                    for job_id in baseline["job_makespan_us"]
                }
            summaries[name] = {
                "scenarios": scenario_summaries,
                "paired_comparisons": paired_comparisons,
                "paired_job_makespan_comparisons": paired_job_comparisons,
                "comparisons_to_phase1": {
                    scenario: {
                        "workload_makespan_change_percent": percent_change(
                            scenario_summaries[scenario]["workload_makespan_us"]["median"],
                            baseline["workload_makespan_us"]["median"],
                        ),
                        "job_makespan_change_percent": {
                            job_id: percent_change(
                                scenario_summaries[scenario]["job_makespan_us"][job_id]["median"],
                                baseline["job_makespan_us"][job_id]["median"],
                            )
                            for job_id in baseline["job_makespan_us"]
                        },
                    }
                    for scenario in SCENARIOS
                    if scenario != "phase1_bare"
                },
            }
        summary = {"metadata": manifest["metadata"], "batch_id": batch_root.name, "workloads": summaries}
        (batch_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        manifest["summary_path"] = "summary.json"
        analysis = [
            f"# Phase 1.2 exploratory batch {batch_root.name}",
            "",
            f"Backend: Gloo; repetitions per scenario: {args.repeats}; seed: {args.experiment_seed}.",
            "Samples come only from paths listed in this batch manifest. P10–P90 are observed run-distribution percentiles, not confidence intervals. Paired differences are system-level comparisons, not strict causal decomposition; ready-first includes control-plane coordination. These CPU/Gloo results do not predict GPU/NCCL performance.",
            "",
        ]
        for name, result in summaries.items():
            analysis.extend([
                f"## {name}", "",
                "| Scenario | workload median (ms) | P10–P90 (ms) | n |",
                "|---|---:|---:|---:|",
            ])
            for scenario, values in result["scenarios"].items():
                metric = values["workload_makespan_us"]
                analysis.append(
                    f"| {scenario} | {metric['median'] / 1000:.3f} | "
                    f"{metric['p10'] / 1000:.3f}–{metric['p90'] / 1000:.3f} | {values['runs']} |"
                )
            analysis.extend(["", "Paired workload differences (current − baseline):", ""])
            for pair, values in result["paired_comparisons"].items():
                analysis.append(
                    f"- `{pair}`: median {values['difference_distribution']['median'] / 1000:.3f} ms; "
                    f"ratio median {values['ratio_distribution']['median']:.3f}; "
                    f"paired n={len(values['repetitions'])}."
                )
            analysis.extend(["", "Per-job application makespans (median ms; change vs `phase1_bare`):", ""])
            analysis.append("| Scenario | " + " | ".join(
                f"{job_id} ms / Δ%" for job_id in result["scenarios"]["phase1_bare"]["job_makespan_us"]
            ) + " |")
            analysis.append("|---|" + "---:|" * len(result["scenarios"]["phase1_bare"]["job_makespan_us"]))
            for scenario, values in result["scenarios"].items():
                changes = result["comparisons_to_phase1"].get(scenario, {}).get("job_makespan_change_percent", {})
                cells = []
                for job_id, metric in values["job_makespan_us"].items():
                    change = changes.get(job_id, 0.0)
                    cells.append(f"{metric['median'] / 1000:.3f} / {change:+.1f}%")
                analysis.append(f"| {scenario} | " + " | ".join(cells) + " |")
            analysis.extend(["", "Execution evidence:", ""])
            bare = result["scenarios"]["phase1_bare"]
            analysis.append(
                f"- Phase 1 bare rank-0 launch sequence: {bare['rank0_distinct_launch_sequence_count']} distinct sequences across {bare['runs']} runs; cross-ProcessGroup communication intervals overlapped in {bare['rank0_cross_group_overlap_runs']}/{bare['runs']} runs."
            )
            for scenario, values in result["scenarios"].items():
                if scenario == "phase1_bare":
                    continue
                evidence = []
                if values["static_plan_sequence_match_runs"] is not None:
                    evidence.append(
                        f"static Plan order {values['static_plan_sequence_match_runs']}/{values['runs']}"
                    )
                if values["group_sequence_plan_match_runs"] is not None:
                    evidence.append(
                        f"per-ProcessGroup order {values['group_sequence_plan_match_runs']}/{values['runs']}"
                    )
                if values["rank_local_serial_admission_passes"] is not None:
                    evidence.append(
                        f"rank-local serial admission {values['rank_local_serial_admission_passes']}/{values['runs']}"
                    )
                if values["global_completion_barrier_passes"] is not None:
                    evidence.append(
                        f"global completion barrier {values['global_completion_barrier_passes']}/{values['runs']}"
                    )
                analysis.append(f"- `{scenario}`: " + "; ".join(evidence) + ".")
            if name == "delayed-mixed-small":
                analysis.extend(["", "Delayed job-1 and scheduler idle attribution (rank 0 median ms):", ""])
                for scenario, values in result["scenarios"].items():
                    ready_wait = values["rank0_job1_first_ready_to_admit_us"]
                    states = values["rank0_scheduler_state_us"]
                    if ready_wait is not None:
                        state_text = (
                            "; ".join(
                                f"{field.replace('_us', '').replace('_', ' ')} "
                                f"{states[field]['median'] / 1000:.3f} ms"
                                for field in (
                                    "head_of_line_idle_us",
                                    "work_conserving_idle_us",
                                    "capacity_wait_us",
                                    "scheduler_delay_us",
                                    "coordination_wait_us",
                                )
                            )
                            if states
                            else "scheduler-state attribution unavailable for bare execution"
                        )
                        analysis.append(
                            f"- `{scenario}`: job-1 ordinal-0 ready→admit {ready_wait['median'] / 1000:.3f} ms; {state_text}. Ready→admit starts after producer readiness, so it excludes the workload's pre-ready delay; HOL is separately attributed from scheduler events."
                        )
            ltf = result["paired_comparisons"]["ltf_serial_minus_fifo_serial"]
            analysis.extend([
                "",
                "Corrected LTF vs FIFO (same rank-local serial limit; paired by repetition):",
                "",
                f"- Workload difference median: {ltf['difference_distribution']['median'] / 1000:+.3f} ms (current LTF − FIFO; paired n={len(ltf['repetitions'])}).",
            ])
            for job_id, comparison in result["paired_job_makespan_comparisons"]["ltf_serial_minus_fifo_serial"].items():
                analysis.append(
                    f"- {job_id} difference median: {comparison['difference_distribution']['median'] / 1000:+.3f} ms."
                )
            analysis.extend([
                "",
                "Interpretation: these are this batch's paired medians, not evidence that LTF is generally better or worse. For Gloo, control and data traffic share the host and can interfere; no result here isolates a pure scheduler or policy causal effect.",
                "Source: `summary.json` fields `scenarios.*`, `paired_comparisons.*`, and `paired_job_makespan_comparisons.*`; representative timeline traces are listed in `manifest.json`.",
                "",
            ])
        (batch_root / "analysis.md").write_text("\n".join(analysis), encoding="utf-8")
        manifest["analysis_path"] = "analysis.md"
        manifest["complete"] = True
        write_manifest()
        try:
            from examples.jobpacer.analysis.visualize import (
                render_summary,
                render_timeline,
                representative_runs,
            )

            figures_dir = batch_root / "figures"
            figures_dir.mkdir(exist_ok=True)
            representatives = {}
            figure_paths = []
            for workload in workload_paths:
                name = workload.stem
                selected = representative_runs(batch_root, name)
                representatives[name] = {
                    scenario: str(path.relative_to(batch_root))
                    for scenario, path in selected.items()
                }
                output = figures_dir / f"{name}-timeline-rank0.svg"
                render_timeline(batch_root, name, 0, output)
                figure_paths.append(output)
            summary_figure = figures_dir / "makespan-summary.svg"
            render_summary(batch_root, summary_figure)
            figure_paths.append(summary_figure)
            manifest["representative_traces"] = representatives
            manifest["figures"] = [
                {
                    "path": str(path.relative_to(batch_root)),
                    "sha256": _sha256(path),
                }
                for path in figure_paths
            ]
        except RuntimeError as exc:
            if "visualization requires Matplotlib" not in str(exc):
                raise
            manifest["visualization_unavailable"] = str(exc)
            manifest["figures"] = []
        write_manifest()
    except BaseException:
        manifest["complete"] = False
        if active_run_record is not None:
            active_run_record.update(
                status="failed", ended_at_utc=datetime.now(timezone.utc).isoformat()
            )
        write_manifest()
        raise
    print(batch_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
