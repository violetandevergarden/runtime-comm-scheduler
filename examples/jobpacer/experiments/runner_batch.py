"""Manifest-driven capacity-scan and SRJF experiment batches."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import shutil
import socket
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
BENCHMARK = ROOT / "benchmark/phase1.2"
WORKLOAD_DIR = BENCHMARK / "experiments/shared/workloads"
RESULTS_DIR = BENCHMARK / "results"
WORKLOADS = (
    "comm-heavy",
    "mixed-message-sizes-large-first",
    "mixed-message-sizes-large-last",
    "long-short-chains",
    "staggered-compute",
    "overlap-window-short",
    "overlap-window-medium",
    "overlap-window-long",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_digest(path: Path) -> str:
    value = json.loads(path.read_text(encoding="utf-8"))
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_manifest(root: Path, manifest: dict[str, Any]) -> None:
    _write_json(root / "manifest.json", manifest)


def _batch_id() -> str:
    import secrets

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + secrets.token_hex(3)


def _run(command: list[str], *, timeout: int = 180) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(ROOT), env.get("PYTHONPATH", "")))
    completed = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, timeout=timeout)
    if completed.returncode:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stdout}\n{completed.stderr}"
        )


def _capacity_name(value: int) -> str:
    return "unbounded" if value == 0 else f"k{value}"


def scenario_matrix(experiment: str, capacities: list[int] | None = None) -> dict[str, dict[str, Any]]:
    bare = {
        "mode": "bare",
        "policy": None,
        "selection": "runtime_arrival",
        "max_outstanding": 0,
        "capacity_label": "unbounded",
    }
    scenes: dict[str, dict[str, Any]] = {"phase1_bare": bare}
    if experiment == "capacity-scan":
        capacities = [1, 2, 3, 0]
        for policy in ("fifo", "ltf"):
            for capacity in capacities:
                name = _capacity_name(capacity)
                scenes[f"{policy}_{name}"] = {
                    "mode": "scheduler",
                    "policy": policy,
                    "selection": "runtime_arrival",
                    "max_outstanding": capacity,
                    "capacity_label": name,
                }
    elif experiment == "srjf":
        if not capacities or 1 not in capacities or 0 not in capacities:
            raise ValueError("SRJF capacities must include k1 and unbounded (0)")
        if len(set(capacities)) != len(capacities) or any(item not in (0, 1, 2, 3) for item in capacities):
            raise ValueError("SRJF capacities must be unique values from 0, 1, 2, 3")
        for capacity in capacities:
            name = _capacity_name(capacity)
            for policy in ("fifo", "ltf", "srjf"):
                scenes[f"{policy}_{name}"] = {
                    "mode": "scheduler",
                    "policy": policy,
                    "selection": "runtime_arrival",
                    "max_outstanding": capacity,
                    "capacity_label": name,
                }
    else:
        raise ValueError(f"unknown experiment: {experiment}")
    return scenes


def _selected_workloads(names: list[str] | None) -> list[Path]:
    selected = list(dict.fromkeys(names or WORKLOADS))
    unknown = sorted(set(selected) - set(WORKLOADS))
    if unknown:
        raise ValueError(f"unknown workload(s): {', '.join(unknown)}; choices={', '.join(WORKLOADS)}")
    return [WORKLOAD_DIR / f"{name}.json" for name in selected]


def _order(scenarios: list[str], seed: int, workload: str, repetition: int) -> list[str]:
    result = list(scenarios)
    random.Random(f"{seed}:{workload}:{repetition}").shuffle(result)
    return result


def _validate_capacity_selection(path: Path) -> tuple[list[int], dict[str, Any], Path]:
    path = path.resolve()
    source = json.loads(path.read_text(encoding="utf-8"))
    source_root = path.parent.resolve()
    manifest_path = source_root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"capacity-selection source batch has no manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("complete") or manifest.get("experiment") != "capacity-scan":
        raise ValueError("capacity selection must reference a complete capacity-scan batch")
    if source.get("source_batch_id") != manifest.get("batch_id"):
        raise ValueError("capacity-selection source_batch_id does not match its directory")
    summary = source_root / "summary.json"
    if not summary.is_file() or _sha256(summary) != source.get("source_summary_sha256"):
        raise ValueError("capacity-selection source summary hash mismatch")
    required = source.get("required_capacities")
    optional = source.get("selected_optional_capacities")
    capacities = source.get("second_stage_capacities")
    if required != [1, 0] or not isinstance(optional, list) or not isinstance(capacities, list):
        raise ValueError("capacity selection must retain required [1, 0] and list optional capacities")
    if any(item not in (2, 3) for item in optional) or len(optional) != len(set(optional)):
        raise ValueError("optional capacities may contain unique values 2 and/or 3 only")
    expected_order = [1, *sorted(optional), 0]
    if capacities != expected_order:
        raise ValueError("second_stage_capacities must be [1, selected optional capacities, 0]")
    return capacities, source, source_root


def _source_metadata() -> dict[str, Any]:
    sources = (
        Path(__file__),
        ROOT / "examples/jobpacer/scripts/run_phase1_2_experiments.py",
        ROOT / "examples/jobpacer/runtime/replay_worker.py",
        ROOT / "examples/jobpacer/scripts/run_phase2.py",
        ROOT / "examples/jobpacer/runtime/plan_builder.py",
        ROOT / "examples/jobpacer/analysis/measurement.py",
        ROOT / "examples/jobpacer/analysis/visualize.py",
        ROOT / "src/runtime_comm_scheduler/scheduler.py",
    )
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    import torch

    return {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "backend": "gloo",
        "world_size": 2,
        "source_sha256": {str(path.relative_to(ROOT)): _sha256(path) for path in sources},
        "git_commit": commit,
        "git_worktree_dirty": dirty,
    }


def _prepare_profile(workload: Path, profile_dir: Path) -> Path:
    name = workload.stem
    profile = profile_dir / ("overlap-window-gloo.json" if name.startswith("overlap-window-") else f"{name}-gloo.json")
    if profile.exists():
        return profile
    if name.startswith("overlap-window-"):
        short = WORKLOAD_DIR / "overlap-window-short.json"
        if not short.is_file():
            raise FileNotFoundError(short)
        workload = short
    _run([
        sys.executable,
        str(ROOT / "examples/jobpacer/scripts/run_comm_profile.py"),
        "--workload", str(workload), "--backend", "gloo", "--world-size", "2",
        "--warmup", "3", "--iterations", "10", "--timeout", "30", "--output", str(profile),
    ], timeout=180)
    return profile


def _plan_set(workload_path: Path, profile_path: Path, policies: list[str]) -> tuple[Any, dict[str, Any]]:
    from examples.jobpacer.gloo.comm_profile import apply_profile
    from examples.jobpacer.runtime.comm_profile import load_profile
    from examples.jobpacer.runtime.plan_builder import build_plan, policy_diagnostics
    from examples.jobpacer.gloo.workloads import load_workload

    workload = apply_profile(
        load_workload(workload_path),
        load_profile(profile_path),
        {"backend": "gloo", "device_type": "cpu", "world_size": 2},
    )
    plans = {}
    for policy in policies:
        plan = build_plan(workload, policy)
        plans[policy] = {
            "digest": plan.digest(),
            "keys": [key.as_list() for key in plan.keys],
            "key_labels": [f"{key.process_group_id}:{key.ordinal}" for key in plan.keys],
            "score_definition": policy_diagnostics(workload, policy)["score_definition"],
            "diagnostics": policy_diagnostics(workload, policy),
        }
    details = {
        "job_count": len(workload.jobs),
        "task_count": sum(len(job.communications) for job in workload.jobs),
        "rank_tensor_bytes": sum(comm.num_bytes for job in workload.jobs for comm in job.communications),
        "policies": plans,
        "identical_plan": {
            f"{left}_equals_{right}": plans[left]["digest"] == plans[right]["digest"]
            for left, right in (("fifo", "ltf"), ("fifo", "srjf"), ("ltf", "srjf"))
            if left in plans and right in plans
        },
    }
    return workload, details


def _distribution(values: list[float]) -> dict[str, Any]:
    from examples.jobpacer.analysis.visualize import _percentile

    if not values:
        raise ValueError("cannot summarize an empty sample")
    return {
        "n": len(values),
        "samples": values,
        "median_us": _percentile(values, 0.5),
        "p10_us": _percentile(values, 0.1),
        "p90_us": _percentile(values, 0.9),
        "min_us": min(values),
        "max_us": max(values),
    }


def _count_distribution(values: list[float]) -> dict[str, Any]:
    distribution = _distribution(values)
    return {
        key.removesuffix("_us"): value
        for key, value in distribution.items()
    }


def _pair_definitions(manifest: dict[str, Any]) -> list[tuple[str, str]]:
    scenes = manifest["scenarios"]
    if manifest["experiment"] == "poll-sensitivity":
        pairs = [tuple(pair) for pair in manifest["pair_definitions"]]
    elif manifest["experiment"] == "capacity-scan":
        pairs = []
        for policy in ("fifo", "ltf"):
            pairs.extend((f"{policy}_{cap}", f"{policy}_k1") for cap in ("k2", "k3", "unbounded"))
            pairs.extend((f"{policy}_{cap}", f"{policy}_unbounded") for cap in ("k1", "k2", "k3"))
        pairs.extend((f"ltf_{cap}", f"fifo_{cap}") for cap in ("k1", "k2", "k3", "unbounded"))
    else:
        capacities = list(dict.fromkeys(
            scene["capacity_label"] for scene in scenes.values() if scene["mode"] == "scheduler"
        ))
        pairs = []
        for cap in capacities:
            pairs.extend(((f"srjf_{cap}", f"fifo_{cap}"), (f"srjf_{cap}", f"ltf_{cap}"), (f"ltf_{cap}", f"fifo_{cap}")))
    pairs.extend((scenario, "phase1_bare") for scenario in scenes if scenario != "phase1_bare")
    return [(current, baseline) for current, baseline in pairs if current in scenes and baseline in scenes]


def _read_trace(root: Path, record: dict[str, Any]) -> dict[str, Any]:
    path = (root / record["path"]).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"manifest trace path is missing or escapes batch: {record.get('path')}")
    if _sha256(path) != record.get("sha256"):
        raise ValueError(f"manifest trace hash mismatch: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _scenario_summary(traces: list[dict[str, Any]], plan_digest: str | None) -> dict[str, Any]:
    from examples.jobpacer.analysis.measurement import occupancy_metrics
    from examples.jobpacer.analysis.visualize import scheduler_state_summary

    if not traces:
        raise ValueError("scenario has no successful runs")
    jobs = [item["job_id"] for item in traces[0]["performance"]["job_makespans"]]
    result: dict[str, Any] = {
        "runs": len(traces),
        "plan_digest": plan_digest,
        "workload_makespan_us": _distribution([trace["performance"]["workload_makespan_us"] for trace in traces]),
        "mean_job_completion_us": _distribution([trace["performance"]["mean_job_completion_us"] for trace in traces]),
        "job_completion_us": {
            job: _distribution([
                next(item["makespan_us"] for item in trace["performance"]["job_makespans"] if item["job_id"] == job)
                for trace in traces
            ])
            for job in jobs
        },
        "longest_ready_to_admit_us": _distribution([
            max(
                task["admit_ts"] - task["ready_record_ts"]
                for rank in trace["ranks"] for job in rank["jobs"] for task in job["tasks"]
            )
            for trace in traces
        ]),
    }
    observations = [
        item for trace in traces for item in trace["performance"]["capacity_observations"]
    ]
    result["capacity_observations"] = {
        field: _count_distribution([item[field] for item in observations])
        for field in ("peak_admission_occupancy", "peak_launched_inflight")
    }
    result["capacity_observations"].update({
        field: _distribution([item[field] for item in observations])
        for field in (
            "admission_occupancy_time_us", "launched_inflight_time_us",
            "pending_launch_time_us",
        )
    })
    result["capacity_observations"]["configured_max_outstanding"] = observations[0]["configured_max_outstanding"]
    for field in (
        "admission_occupancy_by_count_us", "launched_inflight_by_count_us", "pending_launch_by_count_us"
    ):
        count_samples: dict[str, list[float]] = defaultdict(list)
        for item in observations:
            for count, value in item[field].items():
                count_samples[str(count)].append(value)
        result["capacity_observations"][field] = {
            count: _distribution(values) for count, values in sorted(count_samples.items(), key=lambda item: int(item[0]))
        }
    state_names = (
        "work_conserving_idle_us", "head_of_line_idle_us", "capacity_wait_us",
        "capacity_occupied_us", "scheduler_delay_us", "coordination_wait_us",
        "pending_launch_us", "inflight_us", "no_ready_work_us", "unattributed_wait_us",
    )
    state_values: dict[str, list[float]] = {name: [] for name in state_names}
    for trace in traces:
        rank_states = [scheduler_state_summary(rank) for rank in trace["ranks"] if rank.get("mode") == "scheduler"]
        for name in state_names:
            if rank_states:
                state_values[name].append(max(item[name] for item in rank_states))
    result["scheduler_state_us"] = {
        name: _distribution(values) for name, values in state_values.items() if values
    }
    result["run_boundaries_us"] = {
        field: _distribution([trace["performance"][field] for trace in traces])
        for field in ("communication_drain_makespan_us", "validation_total_us", "harness_total_us")
        if field in traces[0]["performance"]
    }
    result["finite_capacity_verified"] = all(
        trace["validation"].get("finite_capacity_verified", False)
        for trace in traces
    )
    return result


def _paired_summary(current: dict[int, dict[str, Any]], baseline: dict[int, dict[str, Any]]) -> dict[str, Any]:
    from examples.jobpacer.analysis.visualize import metric_value_us

    if set(current) != set(baseline):
        raise ValueError("paired comparison has a missing repetition")
    repetitions = sorted(current)
    jobs = [item["job_id"] for item in current[repetitions[0]]["performance"]["job_makespans"]]
    names = ("workload", "mean_job_completion_us", *jobs)
    metrics = {}
    for metric in names:
        differences = [metric_value_us(current[index], metric) - metric_value_us(baseline[index], metric) for index in repetitions]
        distribution = _distribution(differences)
        metrics["workload_makespan_us" if metric == "workload" else metric] = {
            "repetitions": repetitions,
            "difference_us": differences,
            "sample_count": len(differences),
            "median_us": distribution["median_us"],
            "p10_us": distribution["p10_us"],
            "p90_us": distribution["p90_us"],
        }
    return {"metrics": metrics}


def _summarize_batch(root: Path, manifest: dict[str, Any], plans: dict[str, Any]) -> dict[str, Any]:
    records_by_key = {
        (record["workload"], record["scenario"], int(record["repetition"])): record
        for record in manifest["runs"]
    }
    summaries: dict[str, Any] = {}
    for workload in manifest["workloads"]:
        traces: dict[str, dict[int, dict[str, Any]]] = {}
        scenarios: dict[str, Any] = {}
        for scenario, definition in manifest["scenarios"].items():
            trace_map = {
                repetition: _read_trace(root, records_by_key[(workload, scenario, repetition)])
                for repetition in range(manifest["repetitions"])
            }
            traces[scenario] = trace_map
            scenarios[scenario] = _scenario_summary(
                list(trace_map.values()),
                definition.get("plan_digests", {}).get(workload),
            )
        pairs = {
            f"{current}_minus_{baseline}": _paired_summary(traces[current], traces[baseline])
            for current, baseline in _pair_definitions(manifest)
        }
        summaries[workload] = {
            "scenarios": scenarios,
            "paired_comparisons": pairs,
            "identical_plan": plans[workload]["identical_plan"],
        }
    return {
        "batch_id": manifest["batch_id"],
        "experiment": manifest["experiment"],
        "metadata": manifest["metadata"],
        "scenario_order": manifest["scenario_order"],
        "workloads": summaries,
    }


def _analysis(summary: dict[str, Any], manifest: dict[str, Any], plans: dict[str, Any]) -> str:
    from datetime import datetime

    experiment = manifest["experiment"]
    start = datetime.fromisoformat(manifest["created_at_utc"])
    elapsed_s = (datetime.now(timezone.utc) - start).total_seconds()
    successful = [item for item in manifest["runs"] if item.get("status") == "ok"]
    failed = [item for item in manifest["runs"] if item.get("status") == "failed"]
    by_workload = defaultdict(list)
    for record in successful:
        by_workload[record["workload"]].append(record)
    lines = [
        f"# JobPacer {experiment} batch {manifest['batch_id']}", "",
        f"Backend: CPU/Gloo, world size 2, poll interval {manifest['metadata']['completion_poll_interval_s'] * 1000:g} ms; {manifest['repetitions']} repetitions.",
        f"Completed successfully: {len(successful)}/{manifest['expected_run_count']} runs; failures={len(failed)}; wall-clock from batch creation through analysis: {elapsed_s:.1f} s.",
        "P10–P90 describes the observed run distribution, not a confidence interval. Completion timestamps are probe observations. All comparisons are paired by workload and repetition; they are system-level observations, not strict causal decompositions.",
        "Bare is a separate execution path, not an unbounded scheduler capacity point. These finite CPU/Gloo sleep-workload results do not establish GPU/NCCL or real-training performance.", "",
    ]
    if experiment == "poll-sensitivity":
        lines.insert(4, f"Sensitivity source: capacity batch `{manifest['source_batch_id']}`; all compared scenarios use {manifest['metadata']['completion_poll_interval_s'] * 1000:g} ms polling. These samples are separate and are not pooled with the 1 ms main batch.")
    for workload, result in summary["workloads"].items():
        run_records = by_workload[workload]
        workload_elapsed = (
            max(datetime.fromisoformat(item["ended_at_utc"]) for item in run_records)
            - min(datetime.fromisoformat(item["started_at_utc"]) for item in run_records)
        ).total_seconds()
        lines.extend([f"## {workload}", ""])
        lines.append(f"Workload replay window: {workload_elapsed:.1f} s, {len(run_records)} successful runs.")
        lines.append("| Scenario | makespan median (P10–P90) ms | run-mean job completion ms | peak admission / launched | n |")
        lines.append("|---|---:|---:|---:|---:|")
        for scenario, values in result["scenarios"].items():
            make = values["workload_makespan_us"]
            mean = values["mean_job_completion_us"]["median_us"] / 1000
            capacity = values["capacity_observations"]
            adm = capacity["peak_admission_occupancy"]["median"]
            inflight = capacity["peak_launched_inflight"]["median"]
            lines.append(
                f"| {scenario} | {make['median_us']/1000:.3f} ({make['p10_us']/1000:.3f}–{make['p90_us']/1000:.3f}) | {mean:.3f} | {adm:g} / {inflight:g} | {values['runs']} |"
            )
        job_ids = list(next(iter(result["scenarios"].values()))["job_completion_us"])
        lines.extend(["", "Per-job completion medians (ms from common release):", ""])
        lines.append("| Scenario | " + " | ".join(job_ids) + " |")
        lines.append("|---|" + "---:|" * len(job_ids))
        for scenario, values in result["scenarios"].items():
            cells = [f"{values['job_completion_us'][job]['median_us']/1000:.3f}" for job in job_ids]
            lines.append(f"| {scenario} | " + " | ".join(cells) + " |")
        lines.extend(["", f"Plan equality diagnostics: `{json.dumps(result['identical_plan'], sort_keys=True)}`.", ""])
        lines.append("Paired differences (current − baseline; milliseconds):")
        lines.append("| Comparison | workload makespan median (P10–P90) | run-mean job completion median | n |")
        lines.append("|---|---:|---:|---:|")
        for pair, comparison in result["paired_comparisons"].items():
            makespan = comparison["metrics"]["workload_makespan_us"]
            mean = comparison["metrics"]["mean_job_completion_us"]
            lines.append(
                f"| `{pair}` | {makespan['median_us']/1000:+.3f} ({makespan['p10_us']/1000:+.3f}–{makespan['p90_us']/1000:+.3f}) | {mean['median_us']/1000:+.3f} | {makespan['sample_count']} |"
            )
        lines.append("\nPaired per-job completion differences, including raw samples and P10–P90, are in `summary.json`.")
        if experiment == "capacity-scan":
            lines.extend(["", "Capacity interpretation:"])
            for policy in ("fifo", "ltf"):
                base = result["scenarios"][f"{policy}_k1"]["workload_makespan_us"]["median_us"]
                for capacity in ("k2", "k3", "unbounded"):
                    current = result["scenarios"][f"{policy}_{capacity}"]["workload_makespan_us"]["median_us"]
                    diff = current - base
                    peak = result["scenarios"][f"{policy}_{capacity}"]["capacity_observations"]["peak_admission_occupancy"]["median"]
                    lines.append(
                        f"- {policy.upper()} {capacity} vs k1: paired makespan median difference {diff/1000:+.3f} ms; observed median rank peak admission={peak:g}. This reports the full curve, not a selected optimum."
                    )
        elif experiment == "srjf":
            lines.extend(["", "Static SRJF interpretation:"])
            for scenario in manifest["scenario_order"]:
                if not scenario.startswith("srjf_"):
                    continue
                fifo = f"fifo_{scenario.removeprefix('srjf_')}"
                ltf = f"ltf_{scenario.removeprefix('srjf_')}"
                srjf_makespan = result["scenarios"][scenario]["workload_makespan_us"]["median_us"]
                srjf_mean = result["scenarios"][scenario]["mean_job_completion_us"]["median_us"]
                fifo_mean = result["scenarios"][fifo]["mean_job_completion_us"]["median_us"]
                ltf_mean = result["scenarios"][ltf]["mean_job_completion_us"]["median_us"]
                lines.append(
                    f"- {scenario}: SRJF median makespan {srjf_makespan/1000:.3f} ms; run-mean job completion {srjf_mean/1000:.3f} ms vs FIFO {fifo_mean/1000:.3f} and LTF {ltf_mean/1000:.3f}. Per-job values and paired distributions are in `summary.json`."
                )
        lines.append("")
    lines.extend([
        "## Method and limits", "",
        "Admission occupancy counts admitted tasks until completion is observed, including admitted-but-not-launched tasks. Launched inflight begins at collective-call start. Both use half-open intervals over application release through communication drain; finite capacity is checked independently per rank.",
        "Static SRJF is non-preemptive. Its deterministic remaining-path score assumes zero admission delay and candidate readiness; it may wait for a fixed plan head and is not online work-conserving SJF.",
        "Source run paths, SHA-256 values, execution order, source hashes, profiles, plans, and representative timeline paths are recorded in `manifest.json`.", "",
    ])
    return "\n".join(lines)


def _verify_batch(root: Path, manifest: dict[str, Any], *, repeats: int | None, seed: int | None) -> dict[str, Any]:
    from examples.jobpacer.gloo.comm_profile import apply_profile
    from examples.jobpacer.runtime.comm_profile import load_profile
    from examples.jobpacer.runtime.plan_builder import build_plan
    from examples.jobpacer.analysis.visualize import validate_trace
    from examples.jobpacer.gloo.workloads import load_workload

    if repeats is not None and repeats != manifest["repetitions"]:
        raise ValueError("--repeats does not match the manifest")
    if seed is not None and seed != manifest["experiment_seed"]:
        raise ValueError("--experiment-seed does not match the manifest")
    if (
        len(manifest.get("scenario_order", [])) != len(manifest["scenarios"])
        or set(manifest["scenario_order"]) != set(manifest["scenarios"])
    ):
        raise ValueError("scenario_order must list every manifest scenario exactly once")
    expected = {
        (workload, repetition, scenario)
        for workload in manifest["workloads"]
        for repetition in range(manifest["repetitions"])
        for scenario in manifest["scenario_order"]
    }
    keys = [(item.get("workload"), item.get("repetition"), item.get("scenario")) for item in manifest.get("runs", [])]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("manifest run keys are not the expected workload × repetition × scenario product")
    if any(item.get("status") != "ok" for item in manifest["runs"]):
        raise ValueError("batch has failed or unfinished runs")
    for workload in manifest["workloads"]:
        path = WORKLOAD_DIR / f"{workload}.json"
        if _sha256(path) != manifest["metadata"]["workload_sha256"][workload]:
            raise ValueError(f"workload digest mismatch: {workload}")
    profiles = manifest.get("profiles", {})
    for profile in profiles.values():
        path = (root / profile["path"]).resolve()
        if not path.is_relative_to(root.resolve()) or _sha256(path) != profile["sha256"] or _json_digest(path) != profile["digest"]:
            raise ValueError(f"profile hash mismatch: {profile.get('path')}")

    plan_data = json.loads((root / manifest["plans_path"]).read_text(encoding="utf-8"))
    if manifest.get("plans_sha256") and _sha256(root / manifest["plans_path"]) != manifest["plans_sha256"]:
        raise ValueError("plans.json hash mismatch")
    for workload in manifest["workloads"]:
        workload_path = WORKLOAD_DIR / f"{workload}.json"
        profile_name = manifest["workload_profiles"][workload]
        profile_path = root / profiles[profile_name]["path"]
        configured = apply_profile(
            load_workload(workload_path), load_profile(profile_path),
            {"backend": "gloo", "device_type": "cpu", "world_size": 2},
        )
        policies = list(dict.fromkeys(
            scene["policy"] for scene in manifest["scenarios"].values() if scene["policy"]
        ))
        for policy in policies:
            digest = build_plan(configured, policy).digest()
            if digest != plan_data[workload]["policies"][policy]["digest"]:
                raise ValueError(f"Plan digest mismatch for {workload}/{policy}")
            for scene in manifest["scenarios"].values():
                if scene["policy"] == policy and scene["plan_digests"].get(workload) != digest:
                    raise ValueError(f"manifest Plan digest mismatch for {workload}/{policy}")
    for record in manifest["runs"]:
        trace = _read_trace(root, record)
        validate_trace(trace, root / record["path"])
        scenario = manifest["scenarios"][record["scenario"]]
        profile = profiles[manifest["workload_profiles"][record["workload"]]]
        if (
            record.get("workload_sha256") != manifest["metadata"]["workload_sha256"][record["workload"]]
            or record.get("profile_path") != profile["path"]
            or record.get("profile_digest") != profile["digest"]
            or record.get("profile_sha256") != profile["sha256"]
        ):
            raise ValueError(f"run input digest mismatch: {record['path']}")
        if trace.get("validation", {}).get("status") != "ok":
            raise ValueError(f"trace validation failed: {record['path']}")
        if trace["config"]["mode"] != scenario["mode"] or trace["config"]["policy"] != (scenario["policy"] or "fifo"):
            raise ValueError(f"scenario config mismatch: {record['path']}")
        if scenario["mode"] == "scheduler" and record.get("plan_digest") != scenario["plan_digests"][record["workload"]]:
            raise ValueError(f"run Plan digest mismatch: {record['path']}")
        if scenario["mode"] == "scheduler" and not trace["validation"].get("finite_capacity_verified"):
            raise ValueError(f"rank-local finite capacity not verified: {record['path']}")
        if scenario["mode"] == "scheduler" and record["max_outstanding"] != scenario["max_outstanding"]:
            raise ValueError(f"configured capacity mismatch: {record['path']}")
        if trace["config"].get("max_outstanding") != scenario["max_outstanding"]:
            raise ValueError(f"trace capacity config mismatch: {record['path']}")
        round_records = [
            item for item in manifest["runs"]
            if item["workload"] == record["workload"] and item["repetition"] == record["repetition"]
        ]
        observed_order = [item["scenario"] for item in sorted(round_records, key=lambda item: item["order_index"])]
        if observed_order != _order(manifest["scenario_order"], manifest["experiment_seed"], record["workload"], record["repetition"]):
            raise ValueError("recorded per-workload scenario order does not match deterministic seed")
    return plan_data


def _finalize(root: Path, manifest: dict[str, Any]) -> None:
    plans = _verify_batch(root, manifest, repeats=None, seed=None)
    summary = _summarize_batch(root, manifest, plans)
    _write_json(root / "summary.json", summary)
    (root / "analysis.md").write_text(_analysis(summary, manifest, plans), encoding="utf-8")
    finalization_sources = (
        Path(__file__),
        ROOT / "examples/jobpacer/analysis/visualize.py",
        ROOT / "examples/jobpacer/scripts/run_phase2.py",
        ROOT / "examples/jobpacer/analysis/measurement.py",
        ROOT / "examples/jobpacer/runtime/plan_builder.py",
    )
    manifest["finalization_source_sha256"] = {
        str(path.relative_to(ROOT)): _sha256(path) for path in finalization_sources
    }
    manifest["artifacts"] = {
        "plans.json": _sha256(root / "plans.json"),
        "summary.json": _sha256(root / "summary.json"),
        "analysis.md": _sha256(root / "analysis.md"),
    }
    if manifest.get("capacity_selection"):
        manifest["artifacts"][manifest["capacity_selection"]["path"]] = manifest["capacity_selection"]["sha256"]
    manifest["complete"] = False
    manifest["finalizing"] = True
    _write_manifest(root, manifest)
    try:
        from examples.jobpacer.analysis.visualize import (
            render_capacity_job_completion,
            render_capacity_makespan,
            render_capacity_observed_concurrency,
            render_capacity_occupancy_time,
            render_policy_makespan,
            render_policy_mean_job_completion,
            render_policy_paired_differences,
            render_policy_per_job_completion,
            render_policy_ready_wait,
            render_policy_scheduler_state,
            render_summary,
            render_timeline,
            representative_runs,
        )

        figures = root / "figures"
        figures.mkdir(exist_ok=True)
        outputs: list[Path] = []
        if manifest["experiment"] == "capacity-scan":
            for name, renderer in (
                ("capacity-makespan.svg", render_capacity_makespan),
                ("capacity-job-completion.svg", render_capacity_job_completion),
                ("capacity-observed-concurrency.svg", render_capacity_observed_concurrency),
                ("capacity-occupancy-time.svg", render_capacity_occupancy_time),
            ):
                path = figures / name
                renderer(root, path)
                outputs.append(path)
        elif manifest["experiment"] == "srjf":
            for name, renderer in (
                ("policy-makespan-by-capacity.svg", render_policy_makespan),
                ("policy-mean-job-completion.svg", render_policy_mean_job_completion),
                ("policy-per-job-completion.svg", render_policy_per_job_completion),
                ("policy-paired-differences.svg", render_policy_paired_differences),
                ("policy-ready-wait.svg", render_policy_ready_wait),
                ("policy-scheduler-state.svg", render_policy_scheduler_state),
            ):
                path = figures / name
                renderer(root, path)
                outputs.append(path)
        summary_figure = figures / "summary.svg"
        render_summary(root, summary_figure)
        outputs.append(summary_figure)
        representatives = {}
        for workload in manifest["workloads"]:
            representatives[workload] = {
                scenario: str(path.relative_to(root))
                for scenario, path in representative_runs(root, workload).items()
            }
            path = figures / f"{workload}-timeline-rank0.svg"
            render_timeline(root, workload, 0, path)
            outputs.append(path)
        manifest["representative_traces"] = representatives
        manifest["artifacts"].update({
            str(path.relative_to(root)): _sha256(path) for path in outputs
        })
        manifest["figures"] = [
            {"path": str(path.relative_to(root)), "sha256": _sha256(path)}
            for path in outputs
        ]
        manifest["completed_at_utc"] = _now()
        manifest["complete"] = True
        manifest.pop("finalizing", None)
        _write_manifest(root, manifest)
    except BaseException as exc:
        manifest["complete"] = False
        manifest.pop("finalizing", None)
        manifest["finalization_error"] = f"{type(exc).__name__}: {exc}"
        _write_manifest(root, manifest)
        raise


def _new_manifest(experiment: str, root: Path, workloads: list[Path], scenes: dict[str, Any], repeats: int, seed: int, poll: float) -> dict[str, Any]:
    import torch

    return {
        "manifest_schema_version": 2,
        "trace_schema_version": 2,
        "experiment": experiment,
        "batch_id": root.name,
        "complete": False,
        "created_at_utc": _now(),
        "experiment_seed": seed,
        "repetitions": repeats,
        "expected_run_count": len(workloads) * repeats * len(scenes),
        "workloads": [path.stem for path in workloads],
        "scenario_order": list(scenes),
        "scenarios": scenes,
        "round_orders": [],
        "profiles": {},
        "workload_profiles": {},
        "runs": [],
        "metadata": {
            **_source_metadata(),
            "generated_at_utc": _now(),
            "command_line": [sys.executable, *sys.argv],
            "completion_poll_interval_s": poll,
            "workload_sha256": {path.stem: _sha256(path) for path in workloads},
            "torch": torch.__version__,
        },
    }


def run_poll_sensitivity_batch(
    *,
    source_batch: Path,
    workload_name: str,
    pairs: list[tuple[str, str]],
    repeats: int | None = None,
    seed: int | None = None,
) -> Path:
    source_root = source_batch.resolve()
    if not source_root.is_relative_to(RESULTS_DIR.resolve()):
        raise ValueError("sensitivity source must be a batch under benchmark/phase1.2/results")
    source_manifest = json.loads((source_root / "manifest.json").read_text(encoding="utf-8"))
    if not source_manifest.get("complete") or source_manifest.get("experiment") != "capacity-scan":
        raise ValueError("poll sensitivity requires a complete capacity-scan source batch")
    if workload_name not in source_manifest["workloads"]:
        raise ValueError(f"workload is absent from source batch: {workload_name}")
    source_summary = source_root / "summary.json"
    source_summary_sha = _sha256(source_summary)
    profiles = source_manifest["profiles"]
    profile_name = source_manifest["workload_profiles"][workload_name]
    source_profile = source_root / profiles[profile_name]["path"]
    scenes_in_order = source_manifest["scenario_order"]
    for current, baseline in pairs:
        if current not in scenes_in_order or baseline not in scenes_in_order:
            raise ValueError(f"unknown source comparison: {current} − {baseline}")
    if not pairs:
        raise ValueError("at least one paired comparison is required")
    selected_scenarios = [
        scenario for scenario in scenes_in_order
        if any(scenario in pair for pair in pairs)
    ]
    scenes = {
        scenario: json.loads(json.dumps(source_manifest["scenarios"][scenario]))
        for scenario in selected_scenarios
    }
    repeats = repeats if repeats is not None else source_manifest["repetitions"]
    seed = seed if seed is not None else source_manifest["experiment_seed"] + 1
    if repeats < 2 or seed == source_manifest["experiment_seed"]:
        raise ValueError("sensitivity needs at least 2 repeats and a new experiment seed")
    workload_path = WORKLOAD_DIR / f"{workload_name}.json"
    root = RESULTS_DIR / "polling" / (_batch_id() + "-poll-sensitivity")
    (root / "raw").mkdir(parents=True)
    (root / "profiles").mkdir()
    manifest = _new_manifest(
        "poll-sensitivity", root, [workload_path], scenes, repeats, seed, 0.0001
    )
    manifest["source_batch_id"] = source_manifest["batch_id"]
    manifest["source_summary_sha256"] = source_summary_sha
    manifest["pair_definitions"] = [list(pair) for pair in pairs]
    copied_profile = root / "profiles" / profile_name
    shutil.copyfile(source_profile, copied_profile)
    manifest["profiles"][profile_name] = {
        **profiles[profile_name],
        "path": str(copied_profile.relative_to(root)),
    }
    manifest["workload_profiles"][workload_name] = profile_name
    policies = list(dict.fromkeys(
        scene["policy"] for scene in scenes.values() if scene["policy"]
    ))
    _, plan_details = _plan_set(workload_path, copied_profile, policies)
    for scenario, definition in scenes.items():
        policy = definition["policy"]
        definition["plan_digests"] = {
            workload_name: plan_details["policies"][policy]["digest"] if policy else None
        }
    manifest["plans"] = {workload_name: plan_details}
    _write_json(root / "plans.json", manifest["plans"])
    manifest["plans_path"] = "plans.json"
    manifest["plans_sha256"] = _sha256(root / "plans.json")
    _write_manifest(root, manifest)
    active_record = None
    try:
        for repetition in range(repeats):
            order = _order(manifest["scenario_order"], seed, workload_name, repetition)
            manifest["round_orders"].append({"workload": workload_name, "repetition": repetition, "order": order})
            _write_manifest(root, manifest)
            for order_index, scenario in enumerate(order):
                definition = scenes[scenario]
                output = root / "raw" / workload_name / scenario / f"run-{repetition:02d}.json"
                output.parent.mkdir(parents=True, exist_ok=True)
                entrypoint = ROOT / ("examples/jobpacer/scripts/run_phase1.py" if definition["mode"] == "bare" else "examples/jobpacer/scripts/run_phase2.py")
                command = [
                    sys.executable, str(entrypoint), "--policy", definition["policy"] or "fifo",
                    "--selection", definition["selection"], "--scenario", scenario,
                    "--workload", str(workload_path), "--backend", "gloo", "--world-size", "2",
                    "--timeout", "60", "--comm-profile", str(copied_profile), "--output", str(output),
                    "--completion-poll-interval-s", "0.0001", "--max-outstanding", str(definition["max_outstanding"]),
                ]
                if definition["mode"] == "scheduler":
                    command.extend(("--mode", "scheduler"))
                record = {
                    "workload": workload_name, "scenario": scenario,
                    "policy": definition["policy"], "max_outstanding": definition["max_outstanding"],
                    "repetition": repetition, "order_index": order_index, "command": command,
                    "path": str(output.relative_to(root)),
                    "workload_sha256": manifest["metadata"]["workload_sha256"][workload_name],
                    "profile_path": str(copied_profile.relative_to(root)),
                    "profile_digest": manifest["profiles"][profile_name]["digest"],
                    "profile_sha256": manifest["profiles"][profile_name]["sha256"],
                    "plan_digest": definition["plan_digests"][workload_name],
                    "status": "running", "started_at_utc": _now(),
                }
                manifest["runs"].append(record)
                active_record = record
                _write_manifest(root, manifest)
                print(f"sensitivity {workload_name} repetition={repetition} order={order_index} {scenario}", flush=True)
                try:
                    _run(command, timeout=180)
                    trace = json.loads(output.read_text(encoding="utf-8"))
                    if trace.get("validation", {}).get("status") != "ok":
                        raise RuntimeError(f"sensitivity trace validation failed: {output}")
                    record.update(status="ok", sha256=_sha256(output), ended_at_utc=_now())
                    active_record = None
                except BaseException as exc:
                    record.update(status="failed", error=f"{type(exc).__name__}: {exc}", ended_at_utc=_now())
                    active_record = None
                    _write_manifest(root, manifest)
                    raise
                _write_manifest(root, manifest)
        _verify_batch(root, manifest, repeats=repeats, seed=seed)
        _finalize(root, manifest)
    except BaseException as exc:
        manifest["complete"] = False
        if active_record is not None:
            active_record.update(status="failed", error=f"{type(exc).__name__}: {exc}", ended_at_utc=_now())
        manifest["failure"] = f"{type(exc).__name__}: {exc}"
        _write_manifest(root, manifest)
        raise
    return root


def run_experiment(
    *,
    experiment: str,
    repeats: int | None,
    seed: int | None,
    poll_interval_s: float,
    workload_names: list[str] | None = None,
    capacity_selection: Path | None = None,
    finalize_batch: Path | None = None,
) -> Path:
    if finalize_batch is not None:
        root = finalize_batch.resolve()
        if not root.is_relative_to(RESULTS_DIR.resolve()):
            raise ValueError("--finalize-batch must point inside benchmark/phase1.2/results")
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("experiment") not in {"capacity-scan", "srjf"}:
            raise ValueError("not a capacity-scan/SRJF batch")
        if experiment and experiment != manifest["experiment"]:
            raise ValueError("--experiment does not match batch manifest")
        _verify_batch(root, manifest, repeats=repeats, seed=seed)
        _finalize(root, manifest)
        return root

    repeats = repeats if repeats is not None else 10
    if repeats < 2 or poll_interval_s != 0.001:
        raise ValueError("main capacity/SRJF batches require repeats >= 2 and a 1 ms poll interval")
    if experiment == "capacity-scan":
        if capacity_selection is not None:
            raise ValueError("--capacity-selection is only valid with --experiment srjf")
        scenes = scenario_matrix(experiment)
        seed = seed if seed is not None else 120914
        selection_record = None
    else:
        if capacity_selection is None:
            raise ValueError("--experiment srjf requires the frozen --capacity-selection file")
        capacities, selection_record, source_root = _validate_capacity_selection(capacity_selection)
        scenes = scenario_matrix(experiment, capacities)
        seed = seed if seed is not None else 120915
        capacity_selection = capacity_selection.resolve()
        # Freeze the post-analysis choice in its source batch and bind it to the summary hash.
        source_manifest_path = source_root / "manifest.json"
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        source_manifest["capacity_selection"] = {
            "path": "capacity-selection.json",
            "sha256": _sha256(capacity_selection),
        }
        source_manifest.setdefault("artifacts", {})["capacity-selection.json"] = _sha256(capacity_selection)
        _write_manifest(source_root, source_manifest)
    workloads = _selected_workloads(workload_names)
    category = "capacity" if experiment == "capacity-scan" else "priority"
    root = RESULTS_DIR / category / _batch_id()
    (root / "raw").mkdir(parents=True)
    (root / "profiles").mkdir()
    manifest = _new_manifest(experiment, root, workloads, scenes, repeats, seed, poll_interval_s)
    if selection_record is not None:
        shutil.copyfile(capacity_selection, root / "capacity-selection.json")
        manifest["capacity_selection"] = {
            "path": "capacity-selection.json",
            "sha256": _sha256(root / "capacity-selection.json"),
            "source_batch_id": selection_record["source_batch_id"],
        }
    _write_manifest(root, manifest)
    active_record = None
    try:
        for workload_path in workloads:
            profile = _prepare_profile(workload_path, root / "profiles")
            profile_name = profile.name
            if profile_name not in manifest["profiles"]:
                manifest["profiles"][profile_name] = {
                    "path": str(profile.relative_to(root)),
                    "digest": _json_digest(profile),
                    "sha256": _sha256(profile),
                    "source_workload": "overlap-window-short" if profile_name == "overlap-window-gloo.json" else workload_path.stem,
                }
            manifest["workload_profiles"][workload_path.stem] = profile_name
            policies = list(dict.fromkeys(scene["policy"] for scene in scenes.values() if scene["policy"]))
            _, plan_details = _plan_set(workload_path, profile, policies)
            manifest.setdefault("plans", {})[workload_path.stem] = plan_details
            for scenario, definition in scenes.items():
                policy = definition["policy"]
                definition.setdefault("plan_digests", {})[workload_path.stem] = (
                    plan_details["policies"][policy]["digest"] if policy else None
                )
            _write_json(root / "plans.json", manifest["plans"])
            manifest["plans_path"] = "plans.json"
            manifest["plans_sha256"] = _sha256(root / "plans.json")
            _write_manifest(root, manifest)

        for workload_path in workloads:
            workload = workload_path.stem
            profile = root / manifest["profiles"][manifest["workload_profiles"][workload]]["path"]
            for repetition in range(repeats):
                order = _order(manifest["scenario_order"], seed, workload, repetition)
                manifest["round_orders"].append({"workload": workload, "repetition": repetition, "order": order})
                _write_manifest(root, manifest)
                for order_index, scenario in enumerate(order):
                    definition = scenes[scenario]
                    output = root / "raw" / workload / scenario / f"run-{repetition:02d}.json"
                    output.parent.mkdir(parents=True, exist_ok=True)
                    entrypoint = ROOT / ("examples/jobpacer/scripts/run_phase1.py" if definition["mode"] == "bare" else "examples/jobpacer/scripts/run_phase2.py")
                    command = [
                        sys.executable, str(entrypoint),
                        "--policy", definition["policy"] or "fifo",
                        "--selection", definition["selection"], "--scenario", scenario,
                        "--workload", str(workload_path), "--backend", "gloo", "--world-size", "2",
                        "--timeout", "60", "--comm-profile", str(profile), "--output", str(output),
                        "--completion-poll-interval-s", str(poll_interval_s),
                        "--max-outstanding", str(definition["max_outstanding"]),
                    ]
                    if definition["mode"] == "scheduler":
                        command.extend(("--mode", "scheduler"))
                    record = {
                        "workload": workload,
                        "scenario": scenario,
                        "policy": definition["policy"],
                        "max_outstanding": definition["max_outstanding"],
                        "repetition": repetition,
                        "order_index": order_index,
                        "command": command,
                        "path": str(output.relative_to(root)),
                        "workload_sha256": manifest["metadata"]["workload_sha256"][workload],
                        "profile_path": str(profile.relative_to(root)),
                        "profile_digest": manifest["profiles"][profile.name]["digest"],
                        "profile_sha256": manifest["profiles"][profile.name]["sha256"],
                        "plan_digest": definition["plan_digests"][workload],
                        "status": "running",
                        "started_at_utc": _now(),
                    }
                    manifest["runs"].append(record)
                    active_record = record
                    _write_manifest(root, manifest)
                    print(f"{workload} repetition={repetition} order={order_index} {scenario}", flush=True)
                    try:
                        _run(command, timeout=180)
                        trace = json.loads(output.read_text(encoding="utf-8"))
                        if trace.get("validation", {}).get("status") != "ok":
                            raise RuntimeError(f"run validation failed: {output}")
                        if trace.get("config", {}).get("profile_digest") != record["profile_digest"]:
                            raise RuntimeError(f"trace profile digest mismatch: {output}")
                        if definition["mode"] == "scheduler" and trace["validation"].get("plan_digest") != record["plan_digest"]:
                            raise RuntimeError(f"trace Plan digest mismatch: {output}")
                        record.update(status="ok", sha256=_sha256(output), ended_at_utc=_now())
                        active_record = None
                    except BaseException as exc:
                        record.update(status="failed", error=f"{type(exc).__name__}: {exc}", ended_at_utc=_now())
                        active_record = None
                        _write_manifest(root, manifest)
                        raise
                    _write_manifest(root, manifest)
        _write_json(root / "plans.json", manifest["plans"])
        manifest["plans_sha256"] = _sha256(root / "plans.json")
        _verify_batch(root, manifest, repeats=repeats, seed=seed)
        _finalize(root, manifest)
    except BaseException as exc:
        manifest["complete"] = False
        if active_record is not None:
            active_record.update(status="failed", error=f"{type(exc).__name__}: {exc}", ended_at_utc=_now())
        manifest["failure"] = f"{type(exc).__name__}: {exc}"
        _write_manifest(root, manifest)
        raise
    return root
