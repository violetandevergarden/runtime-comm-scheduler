"""Render JobPacer Phase 1/2 traces as SVG figures.

The parsing and interval-building helpers intentionally do not import
Matplotlib.  Install the optional dependency before using the renderers:

    python -m pip install -e '.[visualization]'
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable


SCENARIOS = (
    "phase1_bare",
    "ready_first_unbounded",
    "ready_first_serial",
    "fifo_unbounded",
    "fifo_serial",
    "ltf_serial",
)
TRACE_SCHEMA_VERSION = 2
SCENARIO_LABELS = {
    "phase1_bare": "Phase 1 bare",
    "ready_first_unbounded": "Ready-first unbounded",
    "ready_first_serial": "Ready-first serial",
    "fifo_unbounded": "FIFO unbounded",
    "fifo_serial": "FIFO serial",
    "ltf_serial": "LTF serial",
}
METRIC_LABELS = {
    "workload": "workload",
    "job-0": "job-0",
    "job-1": "job-1",
}
PAIRED_SCENARIO_PAIRS = (
    ("ready_first_unbounded", "phase1_bare"),
    ("ready_first_serial", "ready_first_unbounded"),
    ("fifo_serial", "ready_first_serial"),
    ("ltf_serial", "fifo_serial"),
    ("fifo_serial", "fifo_unbounded"),
)


def load_trace(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_manifest(result_dir: str | Path) -> dict[str, Any]:
    path = Path(result_dir) / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not manifest.get("complete", False) and not manifest.get("finalizing", False):
        raise ValueError(f"batch manifest is incomplete: {path}")
    return manifest


def scenario_order(result_dir: str | Path) -> list[str]:
    manifest = load_manifest(result_dir)
    return list(manifest.get("scenario_order") or manifest["scenarios"])


def scenario_label(name: str) -> str:
    if name in SCENARIO_LABELS:
        return SCENARIO_LABELS[name]
    if name == "phase1_bare":
        return "Bare"
    if name.endswith("_unbounded"):
        return f"{name[:-10].upper()} unbounded"
    strategy, _, capacity = name.partition("_k")
    if strategy in {"fifo", "ltf", "srjf"}:
        return f"{strategy.upper()} {capacity or 'unbounded'}"
    return name.replace("_", " ")


def _manifest_paths(result_dir: str | Path, workload: str, scenario: str) -> list[Path] | None:
    manifest = load_manifest(result_dir)
    paths = []
    for run in manifest.get("runs", []):
        if run.get("workload") != workload or run.get("scenario") != scenario:
            continue
        relative = run.get("path")
        if not relative:
            raise ValueError("manifest run has no path")
        path = Path(result_dir) / relative
        if not path.resolve().is_relative_to(Path(result_dir).resolve()):
            raise ValueError(f"manifest path escapes batch directory: {relative}")
        if not path.is_file():
            raise FileNotFoundError(f"manifest-listed trace is missing: {path}")
        expected_hash = run.get("sha256")
        if expected_hash:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != expected_hash:
                raise ValueError(f"manifest hash mismatch: {path}")
        paths.append(path)
    if not paths:
        raise FileNotFoundError(
            f"manifest has no runs for workload={workload!r}, scenario={scenario!r}"
        )
    return sorted(paths)


def trace_paths(result_dir: str | Path, workload: str, scenario: str) -> list[Path]:
    manifest_paths = _manifest_paths(result_dir, workload, scenario)
    assert manifest_paths is not None
    return manifest_paths


def _workload_makespan_us(trace: dict[str, Any]) -> float:
    performance = trace.get("performance", {})
    if "workload_makespan_us" in performance:
        return float(performance["workload_makespan_us"])
    ranks = trace.get("ranks", [])
    return max(float(rank["replay_makespan_us"]) for rank in ranks)


def _percentile(values: Iterable[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot calculate a percentile of an empty sample")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def metric_value_us(trace: dict[str, Any], metric: str) -> float:
    if metric == "workload":
        return _workload_makespan_us(trace)
    if metric == "mean_job_completion_us":
        performance = trace.get("performance", {})
        if "mean_job_completion_us" in performance:
            return float(performance["mean_job_completion_us"])
        values = [item["makespan_us"] for item in performance.get("job_makespans", [])]
        return sum(values) / len(values)
    for item in trace.get("performance", {}).get("job_makespans", []):
        if item["job_id"] == metric:
            return float(item["makespan_us"])
    raise KeyError(f"trace has no metric {metric!r}")


def choose_representative_run(paths: Iterable[str | Path]) -> Path:
    candidates = [(Path(path), load_trace(path)) for path in paths]
    if not candidates:
        raise ValueError("at least one trace is required")
    for path, trace in candidates:
        validate_trace(trace, path)
    median_us = _percentile(
        (_workload_makespan_us(trace) for _, trace in candidates), 0.5
    )
    return min(
        candidates,
        key=lambda item: (
            abs(_workload_makespan_us(item[1]) - median_us),
            item[0].name,
        ),
    )[0]


def representative_runs(
    result_dir: str | Path, workload: str
) -> dict[str, Path]:
    return {
        scenario: choose_representative_run(
            trace_paths(result_dir, workload, scenario)
        )
        for scenario in scenario_order(result_dir)
    }


def _rank_trace(trace: dict[str, Any], rank: int) -> dict[str, Any]:
    for item in trace.get("ranks", []):
        if int(item["rank"]) == rank:
            return item
    raise ValueError(f"trace has no rank {rank}")


_TASK_TIMESTAMP_FIELDS = (
    "producer_compute_start_ts",
    "ready_record_ts",
    "submit_api_start_ts",
    "submit_api_return_ts",
    "admit_ts",
    "collective_call_start_ts",
    "collective_call_return_ts",
    "completion_observed_ts",
    "consumer_compute_start_ts",
    "consumer_compute_end_ts",
    "application_wait_start_ts",
    "underlying_wait_start_ts",
    "wait_return_ts",
    "application_task_end_ts",
    "validation_start_ts",
    "validation_end_ts",
)


def validate_trace(trace: dict[str, Any], source: str | Path = "<trace>") -> None:
    """Reject incomplete/ambiguous traces before building any plot."""
    version = trace.get("trace_schema_version")
    if version is None:
        versions = {
            rank_trace.get("trace_schema_version")
            for rank_trace in trace.get("ranks", [])
        }
        version = versions.pop() if len(versions) == 1 else None
    if version != TRACE_SCHEMA_VERSION:
        raise ValueError(
            f"{source}: expected trace_schema_version={TRACE_SCHEMA_VERSION}, "
            f"got {trace.get('trace_schema_version')!r}"
        )
    for rank_trace in trace.get("ranks", []):
        rank = rank_trace.get("rank")
        required = (
            "application_release_ts",
            "application_end_ts",
            "communication_drain_end_ts",
            "validation_end_ts",
            "harness_start_ts",
            "harness_end_ts",
        )
        _require_fields(rank_trace, required, source, rank=rank)
        release = rank_trace["application_release_ts"]
        application_end = rank_trace["application_end_ts"]
        drain = rank_trace["communication_drain_end_ts"]
        validation_end = rank_trace["validation_end_ts"]
        harness_end = rank_trace["harness_end_ts"]
        _require_order(
            (release, application_end, drain, validation_end, harness_end),
            source,
            f"rank {rank} run boundaries",
        )
        _require_order(
            (validation_end, rank_trace["harness_start_ts"], harness_end),
            source,
            f"rank {rank} validation/harness boundaries",
        )
        for job in rank_trace.get("jobs", []):
            for task in job.get("tasks", []):
                context = (rank, job.get("job_id"), task.get("ordinal"))
                _require_fields(
                    task,
                    _TASK_TIMESTAMP_FIELDS,
                    source,
                    rank=rank,
                    job=job.get("job_id"),
                    ordinal=task.get("ordinal"),
                )
                values = [task[field] for field in _TASK_TIMESTAMP_FIELDS]
                _require_order(values[:4], source, "task producer/API interval", *context)
                _require_order(
                    (task["submit_api_start_ts"], task["submit_api_return_ts"]),
                    source,
                    "task API interval",
                    *context,
                )
                if rank_trace.get("mode") == "bare":
                    _require_order(
                        (task["submit_api_start_ts"], task["collective_call_start_ts"], task["collective_call_return_ts"], task["submit_api_return_ts"]),
                        source,
                        "bare API/collective boundaries",
                        *context,
                    )
                _require_order(
                    (task["admit_ts"], task["collective_call_start_ts"], task["collective_call_return_ts"], task["completion_observed_ts"]),
                    source,
                    "task admission/collective interval",
                    *context,
                )
                _require_order(
                    (task["consumer_compute_start_ts"], task["consumer_compute_end_ts"]),
                    source,
                    "task consumer interval",
                    *context,
                )
                _require_order(
                    (task["application_wait_start_ts"], task["underlying_wait_start_ts"], task["wait_return_ts"], task["application_task_end_ts"]),
                    source,
                    "task wait interval",
                    *context,
                )
                if task["validation_start_ts"] < drain:
                    raise ValueError(
                        f"{source}: validation starts before communication drain "
                        f"(rank={rank}, job={job.get('job_id')}, ordinal={task.get('ordinal')})"
                    )
                if task["completion_observed_ts"] > drain:
                    raise ValueError(
                        f"{source}: completion observation is after communication drain "
                        f"(rank={rank}, job={job.get('job_id')}, ordinal={task.get('ordinal')})"
                    )
                if task["application_task_end_ts"] > application_end:
                    raise ValueError(
                        f"{source}: application task end is after application end "
                        f"(rank={rank}, job={job.get('job_id')}, ordinal={task.get('ordinal')})"
                    )
                _require_order(
                    (task["consumer_compute_end_ts"], task["application_wait_start_ts"]),
                    source,
                    "consumer-to-application-wait boundary",
                    *context,
                )
                _require_order(
                    (task["validation_start_ts"], task["validation_end_ts"]),
                    source,
                    "task validation interval",
                    *context,
                )


def _require_fields(record: dict[str, Any], fields, source, **context) -> None:
    missing = [field for field in fields if record.get(field) is None]
    if missing:
        details = ", ".join(f"{key}={value!r}" for key, value in context.items())
        raise ValueError(f"{source}: missing {missing} ({details})")


def _require_order(values, source, label, *context) -> None:
    if any(end < start for start, end in zip(values, values[1:])):
        raise ValueError(f"{source}: reversed {label}: {values!r}; context={context!r}")


def scheduler_state_intervals(rank_trace: dict[str, Any]) -> list[dict[str, Any]]:
    """Reconstruct rank-local scheduler idle causes from recorded events."""
    tasks = [
        task
        for job in rank_trace.get("jobs", [])
        for task in job.get("tasks", [])
    ]
    events = {rank_trace["application_release_ts"], rank_trace["application_end_ts"]}
    for task in tasks:
        events.update(
            task[name]
            for name in (
                "ready_record_ts",
                "admit_ts",
                "collective_call_start_ts",
                "collective_call_return_ts",
                "completion_observed_ts",
            )
        )
    coordination = rank_trace.get("control", {}).get("coordination_intervals", [])
    for interval in coordination:
        events.update((interval["start_ts"], interval["end_ts"]))
    ordered = sorted(events)
    plan = [tuple(key) for key in rank_trace.get("plan", {}).get("keys", [])]
    local_keys = {tuple(task["key"]) for task in tasks}
    plan = [key for key in plan if key in local_keys]
    max_outstanding = rank_trace.get("max_outstanding")
    intervals = []
    for start, end in zip(ordered, ordered[1:]):
        if end <= start:
            continue
        ready = {tuple(task["key"]) for task in tasks if task["ready_record_ts"] <= start < task["admit_ts"]}
        admitted = {
            tuple(task["key"])
            for task in tasks
            if task["admit_ts"] <= start < task["completion_observed_ts"]
        }
        pending_launch = {
            tuple(task["key"])
            for task in tasks
            if task["admit_ts"] <= start < task["collective_call_start_ts"]
        }
        inflight = {
            tuple(task["key"])
            for task in tasks
            if task["collective_call_start_ts"] <= start < task["completion_observed_ts"]
        }
        coordinating = any(
            interval["start_ts"] <= start < interval["end_ts"]
            for interval in coordination
        )
        if pending_launch:
            state = "admission_pending_launch"
        elif max_outstanding and len(admitted) >= max_outstanding:
            state = "capacity_busy"
        elif inflight:
            state = "inflight"
        elif coordinating:
            state = "coordination_wait"
        elif not ready:
            state = "no_ready_work"
        elif rank_trace.get("selection") == "ready_first":
            # Local readiness does not prove that all member ranks are ready.
            state = "unattributed_wait"
        else:
            remaining = [key for key in plan if not any(tuple(task["key"]) == key and task["admit_ts"] <= start for task in tasks)]
            head = remaining[0] if remaining else None
            if head not in ready and ready:
                state = "head_not_ready_with_later_ready"
            elif head in ready:
                state = "ready_head_scheduler_delay"
            elif ready:
                state = "work_conserving_idle"
            else:
                state = "inflight"
        intervals.append({"state": state, "start_ts": start, "end_ts": end, "duration_us": end - start})
    return intervals


def scheduler_state_summary(rank_trace: dict[str, Any]) -> dict[str, float]:
    summary = {
        "work_conserving_idle_us": 0.0,
        "head_of_line_idle_us": 0.0,
        "capacity_wait_us": 0.0,
        "capacity_occupied_us": 0.0,
        "scheduler_delay_us": 0.0,
        "coordination_wait_us": 0.0,
        "pending_launch_us": 0.0,
        "inflight_us": 0.0,
        "no_ready_work_us": 0.0,
        "unattributed_wait_us": 0.0,
    }
    for interval in scheduler_state_intervals(rank_trace):
        duration = float(interval["duration_us"])
        state = interval["state"]
        if state in {"ready_head_scheduler_delay", "head_not_ready_with_later_ready"}:
            summary["work_conserving_idle_us"] += duration
        if state == "head_not_ready_with_later_ready":
            summary["head_of_line_idle_us"] += duration
        elif state == "capacity_busy":
            summary["capacity_wait_us"] += duration
            summary["capacity_occupied_us"] += duration
        elif state == "ready_head_scheduler_delay":
            summary["scheduler_delay_us"] += duration
        elif state == "admission_pending_launch":
            summary["pending_launch_us"] += duration
        elif state == "inflight":
            summary["inflight_us"] += duration
        elif state == "no_ready_work":
            summary["no_ready_work_us"] += duration
        elif state == "unattributed_wait":
            summary["unattributed_wait_us"] += duration
    summary["coordination_wait_us"] = float(
        rank_trace.get("control", {}).get("coordination_wait_us", 0.0)
    )
    return summary


def timeline_data(trace: dict[str, Any], rank: int) -> dict[str, Any]:
    """Return relative-ms intervals for one rank without importing Matplotlib."""

    validate_trace(trace)
    rank_trace = _rank_trace(trace, rank)
    origin = int(rank_trace["application_release_ts"])

    def relative(ts: int | None) -> float | None:
        return None if ts is None else (int(ts) - origin) / 1000.0

    jobs = []
    for job in rank_trace.get("jobs", []):
        tasks = []
        for task in job.get("tasks", []):
            intervals = []
            for name, start_key, end_key, color, lane in (
                (
                    "producer_compute",
                    "producer_compute_start_ts",
                    "ready_record_ts",
                    "producer",
                    "compute",
                ),
                (
                    "consumer_compute",
                    "consumer_compute_start_ts",
                    "consumer_compute_end_ts",
                    "consumer",
                    "compute",
                ),
                (
                    "admission_wait",
                    "ready_record_ts",
                    "admit_ts",
                    "admission",
                    "admission",
                ),
                (
                    "communication",
                    "collective_call_start_ts",
                    "completion_observed_ts",
                    "communication",
                    "communication",
                ),
                (
                    "application_binding_wait",
                    "application_wait_start_ts",
                    "underlying_wait_start_ts",
                    "application_wait",
                    "application_wait",
                ),
                (
                    "backend_wait",
                    "underlying_wait_start_ts",
                    "wait_return_ts",
                    "wait",
                    "backend_wait",
                ),
                (
                    "validation",
                    "validation_start_ts",
                    "validation_end_ts",
                    "validation",
                    "validation",
                ),
            ):
                start = relative(task.get(start_key))
                end = relative(task.get(end_key))
                if start is None or end is None:
                    raise ValueError(f"timeline interval missing after validation: {name}")
                intervals.append(
                    {
                        "kind": name,
                        "lane": lane,
                        "color": color,
                        "start_ms": start,
                        "end_ms": end,
                        "ordinal": task.get("ordinal"),
                    }
                )
            tasks.append(
                {
                    "ordinal": task.get("ordinal"),
                    "intervals": intervals,
                    "markers": {
                        name: relative(task.get(key))
                        for name, key in (
                            ("ready", "ready_record_ts"),
                            ("admit", "admit_ts"),
                            ("submit", "collective_call_start_ts"),
                            ("complete", "completion_observed_ts"),
                        )
                    },
                }
            )
        jobs.append({"job_id": job["job_id"], "tasks": tasks})
    return {
        "rank": rank,
        "origin_ts": origin,
        "makespan_ms": float(rank_trace["application_makespan_us"]) / 1000.0,
        "communication_drain_ms": float(
            rank_trace["communication_drain_makespan_us"]
        )
        / 1000.0,
        "coordination_intervals": [
            {
                "operation": item["operation"],
                "start_ms": relative(item["start_ts"]),
                "end_ms": relative(item["end_ts"]),
            }
            for item in rank_trace.get("control", {}).get("coordination_intervals", [])
        ],
        "scheduler_state_intervals": [
            {
                **item,
                "start_ms": relative(item["start_ts"]),
                "end_ms": relative(item["end_ts"]),
            }
            for item in scheduler_state_intervals(rank_trace)
        ],
        "jobs": jobs,
    }


def summary_data(
    result_dir: str | Path, workloads: Iterable[str] | None = None
) -> dict[str, Any]:
    manifest = load_manifest(result_dir)
    root = Path(result_dir)
    names = list(workloads) if workloads is not None else list(manifest["workloads"])
    scenarios_in_order = scenario_order(result_dir)
    pairs = _pair_definitions(manifest)
    result: dict[str, Any] = {"workloads": {}, "paired_comparisons": {}}
    for workload in names:
        scenarios: dict[str, Any] = {}
        trace_by_scenario: dict[str, dict[int, dict[str, Any]]] = {}
        for scenario in scenarios_in_order:
            records = {
                int(record["repetition"]): (root / record["path"])
                for record in manifest["runs"]
                if record.get("workload") == workload
                and record.get("scenario") == scenario
                and record.get("status") == "ok"
            }
            paths = [records[index] for index in sorted(records)]
            traces = [load_trace(path) for path in paths]
            if not traces:
                raise ValueError(f"manifest has no successful runs for {workload}/{scenario}")
            for path, trace in zip(paths, traces):
                validate_trace(trace, path)
            trace_by_scenario[scenario] = {
                repetition: load_trace(path) for repetition, path in records.items()
            }
            metrics = ["workload", "mean_job_completion_us"] + [
                item["job_id"]
                for item in traces[0].get("performance", {}).get("job_makespans", [])
            ]
            scenarios[scenario] = {
                metric: {
                    "samples": [metric_value_us(trace, metric) for trace in traces],
                    "count": len(traces),
                    "p10_us": _percentile(
                        (metric_value_us(trace, metric) for trace in traces), 0.1
                    ),
                    "median_us": _percentile(
                        (metric_value_us(trace, metric) for trace in traces), 0.5
                    ),
                    "p90_us": _percentile(
                        (metric_value_us(trace, metric) for trace in traces), 0.9
                    ),
                }
                for metric in metrics
            }
        result["workloads"][workload] = scenarios
        result["paired_comparisons"][workload] = {
            f"{current}_minus_{baseline}": _paired_metrics(
                trace_by_scenario[current], trace_by_scenario[baseline]
            )
            for current, baseline in pairs
        }
    return result


def _pair_definitions(manifest: dict[str, Any]) -> list[tuple[str, str]]:
    scenes = manifest["scenarios"]
    if manifest.get("experiment") == "poll-sensitivity":
        pairs = [tuple(pair) for pair in manifest["pair_definitions"]]
    elif manifest.get("experiment") is None:
        pairs = list(PAIRED_SCENARIO_PAIRS)
    elif manifest.get("experiment") == "capacity-scan":
        pairs = []
        for policy in ("fifo", "ltf"):
            pairs.extend(
                (f"{policy}_{capacity}", f"{policy}_k1")
                for capacity in ("k2", "k3", "unbounded")
            )
            pairs.extend(
                (f"{policy}_{capacity}", f"{policy}_unbounded")
                for capacity in ("k1", "k2", "k3")
            )
        pairs.extend(
            (f"ltf_{capacity}", f"fifo_{capacity}")
            for capacity in ("k1", "k2", "k3", "unbounded")
        )
    else:
        capacities = list(dict.fromkeys(
            scene.get("capacity_label", "unbounded")
            for scene in scenes.values()
            if scene.get("mode") == "scheduler"
        ))
        pairs = []
        for capacity in capacities:
            fifo = f"fifo_{capacity}"
            ltf = f"ltf_{capacity}"
            srjf = f"srjf_{capacity}"
            pairs.extend(((srjf, fifo), (srjf, ltf), (ltf, fifo)))
    pairs.extend(
        (scenario, "phase1_bare")
        for scenario in scenes
        if scenario != "phase1_bare"
    )
    return [(current, baseline) for current, baseline in pairs if current in scenes and baseline in scenes]


def _paired_metrics(
    current: dict[int, dict[str, Any]], baseline: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    repetitions = sorted(set(current) & set(baseline))
    if not repetitions or set(current) != set(baseline):
        raise ValueError("paired scenarios have missing repetitions")
    job_ids = [
        item["job_id"]
        for item in current[repetitions[0]]["performance"]["job_makespans"]
    ]
    names = ["workload", "mean_job_completion_us", *job_ids]
    metrics = {}
    for name in names:
        differences = [
            metric_value_us(current[index], name)
            - metric_value_us(baseline[index], name)
            for index in repetitions
        ]
        metrics[name] = {
            "repetitions": repetitions,
            "difference_us": differences,
            "sample_count": len(differences),
            "median_us": _percentile(differences, 0.5),
            "p10_us": _percentile(differences, 0.1),
            "p90_us": _percentile(differences, 0.9),
        }
    return {"metrics": metrics}


def paired_differences(
    result_dir: str | Path, workload: str, baseline: str, scenario: str
) -> dict[str, Any]:
    """Match runs by manifest repetition (or run index for legacy results)."""
    root = Path(result_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("complete", False):
        raise ValueError(f"batch manifest is incomplete: {root / 'manifest.json'}")

    def values(scenario_name):
        selected = {}
        for index, record in enumerate(manifest.get("runs", [])):
            if record.get("workload") != workload or record.get("scenario") != scenario_name:
                continue
            path = root / record["path"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != record.get("sha256"):
                raise ValueError(f"manifest hash mismatch: {path}")
            repetition = record.get("repetition", index)
            selected[int(repetition)] = metric_value_us(load_trace(path), "workload")
        return selected

    baseline_by_repetition = values(baseline)
    scenario_by_repetition = values(scenario)
    common = sorted(set(baseline_by_repetition) & set(scenario_by_repetition))
    differences = [scenario_by_repetition[index] - baseline_by_repetition[index] for index in common]
    ratios = [scenario_by_repetition[index] / baseline_by_repetition[index] for index in common]
    return {"repetitions": common, "difference_us": differences, "ratio": ratios}


def _matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "visualization requires Matplotlib; install with "
            "python -m pip install -e '.[visualization]'"
        ) from exc
    return plt, Line2D, Patch


def render_timeline(
    result_dir: str | Path,
    workload: str,
    rank: int,
    output: str | Path,
    *,
    x_min_ms: float | None = None,
    x_max_ms: float | None = None,
) -> None:
    plt, Line2D, Patch = _matplotlib()
    scenarios = scenario_order(result_dir)
    selected = representative_runs(result_dir, workload)
    data_by_scenario = {
        scenario: timeline_data(load_trace(path), rank)
        for scenario, path in selected.items()
    }
    job_count = max(len(data["jobs"]) for data in data_by_scenario.values())
    colors = {
        "producer": "#8c8c8c",
        "consumer": "#55a868",
        "admission": "#e17c05",
        "communication": "#4c78a8",
        "wait": "#c44e52",
        "application_wait": "#d65f5f",
        "validation": "#8172b2",
    }
    lane_order = {
        "compute": 0,
        "admission": 1,
        "communication": 2,
        "application_wait": 3,
        "backend_wait": 4,
        "validation": 5,
    }
    state_colors = {
        "no_ready_work": "#d9d9d9",
        "capacity_busy": "#f0ad4e",
        "head_not_ready_with_later_ready": "#e17c05",
        "ready_head_scheduler_delay": "#8172b2",
        "inflight": "#4c78a8",
        "work_conserving_idle": "#c44e52",
        "unattributed_wait": "#777777",
        "admission_pending_launch": "#bcbd22",
        "coordination_wait": "#9467bd",
    }
    job_stride = len(lane_order) + 1
    columns = 1
    rows = len(scenarios)
    panel_height = max(5.4, 0.25 * (job_count * job_stride + 2) + 1.2)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(16, panel_height * rows + 2.2),
        sharey=False,
        squeeze=False,
    )
    for axis, scenario in zip(axes.flat, scenarios):
        data = data_by_scenario[scenario]
        labels = []
        tick_positions = []
        for job_index, job in enumerate(data["jobs"]):
            base_y = job_index * job_stride
            for lane_index, lane in enumerate(
                ("compute", "admission", "communication", "application", "backend", "validation")
            ):
                tick_positions.append(base_y + lane_index)
                labels.append(f"{job['job_id']} {lane}")
            for task in job["tasks"]:
                for interval in task["intervals"]:
                    y = base_y + lane_order[interval["lane"]]
                    axis.barh(
                        y,
                        interval["end_ms"] - interval["start_ms"],
                        left=interval["start_ms"],
                        height=0.48,
                        color=colors[interval["color"]],
                        alpha=0.85,
                    )
                markers = task["markers"]
                if markers["ready"] is not None:
                    axis.plot(markers["ready"], base_y + 0.28, "^", color="black", ms=5)
                for marker_name, marker_y in (
                    ("admit", base_y + lane_order["admission"]),
                    ("submit", base_y + lane_order["communication"]),
                    ("complete", base_y + lane_order["communication"]),
                ):
                    if markers[marker_name] is not None:
                        axis.vlines(
                            markers[marker_name],
                            marker_y - 0.28,
                            marker_y + 0.28,
                            color="black",
                            linewidth=0.8,
                        )
        coordination_y = len(data["jobs"]) * job_stride
        state_y = coordination_y + 1
        tick_positions.extend((coordination_y, state_y))
        labels.extend(("control coordination", "scheduler state"))
        for interval in data["coordination_intervals"]:
            axis.barh(
                coordination_y,
                interval["end_ms"] - interval["start_ms"],
                left=interval["start_ms"],
                height=0.48,
                color="#8172b2",
                alpha=0.85,
            )
        for interval in data["scheduler_state_intervals"]:
            axis.barh(
                state_y,
                interval["end_ms"] - interval["start_ms"],
                left=interval["start_ms"],
                height=0.48,
                color=state_colors.get(interval["state"], "#777777"),
                alpha=0.85,
            )
        axis.set_title(
            f"{scenario_label(scenario)} · rank {rank} · "
            f"{data['makespan_ms']:.3f} ms",
            fontsize=10,
        )
        axis.set_xlabel("relative time (ms)")
        axis.set_yticks(tick_positions)
        axis.set_yticklabels(labels)
        axis.tick_params(axis="y", labelsize=9, pad=10)
        axis.grid(axis="x", alpha=0.25)
        if x_min_ms is not None or x_max_ms is not None:
            axis.set_xlim(x_min_ms, x_max_ms)
    for axis in list(axes.flat)[len(scenarios):]:
        axis.set_axis_off()
    interval_legend = [
        Patch(color=colors["producer"], label="producer compute"),
        Patch(color=colors["consumer"], label="consumer overlap compute"),
        Patch(color=colors["admission"], label="ready → admit"),
        Patch(color=colors["communication"], label="submit → complete"),
        Patch(color=colors["application_wait"], label="application wait → binding"),
        Patch(color=colors["wait"], label="underlying wait → return"),
        Patch(color=colors["validation"], label="harness validation"),
        Patch(color="#8172b2", label="control coordination"),
        Line2D([], [], marker="^", color="black", linestyle="None", label="ready"),
        Line2D([], [], marker="|", color="black", linestyle="None", label="admit / submit / complete"),
    ]
    state_labels = {
        "no_ready_work": "no ready work",
        "capacity_busy": "capacity busy",
        "head_not_ready_with_later_ready": "scheduler HOL",
        "ready_head_scheduler_delay": "scheduler delay",
        "inflight": "data collective in flight",
        "work_conserving_idle": "work-conserving idle",
        "unattributed_wait": "unattributed wait",
        "admission_pending_launch": "admission pending launch",
        "coordination_wait": "coordination wait",
    }
    state_legend = [
        Patch(color=state_colors[state], label=label)
        for state, label in state_labels.items()
    ]
    figure.legend(
        handles=interval_legend,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.095),
        ncol=5,
        fontsize=8,
        frameon=False,
    )
    figure.legend(
        handles=state_legend,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=5,
        fontsize=8,
        frameon=False,
        title="scheduler-state lane",
        title_fontsize=8,
    )
    manifest = load_manifest(result_dir)
    figure.suptitle(
        f"JobPacer timeline · {workload} · n={manifest['repetitions']} per scenario · "
        f"poll={manifest['metadata']['completion_poll_interval_s'] * 1000:g} ms",
        y=0.99,
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0.16, 1, 0.96), h_pad=2.4, w_pad=1.0)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, format="svg")
    plt.close(figure)


def render_summary(
    result_dir: str | Path, output: str | Path, workloads: Iterable[str] | None = None
) -> None:
    plt, Line2D, _Patch = _matplotlib()
    manifest = load_manifest(result_dir)
    data = summary_data(result_dir, workloads)
    scenarios_in_order = scenario_order(result_dir)
    workload_names = list(data["workloads"])
    if not workload_names:
        raise ValueError("no workloads found")
    metrics_by_workload = {
        workload: list(next(iter(data["workloads"][workload].values())))
        for workload in workload_names
    }
    metric_rows = max(map(len, metrics_by_workload.values()))
    workload_columns = min(4, len(workload_names))
    workload_rows = (len(workload_names) + workload_columns - 1) // workload_columns
    block_rows = metric_rows + 1
    height_ratios = [
        ratio
        for _ in range(workload_rows)
        for ratio in ([2.0] * metric_rows + [3.0])
    ]
    figure, axes = plt.subplots(
        workload_rows * block_rows,
        workload_columns,
        figsize=(max(10, 4.8 * workload_columns), sum(height_ratios) + 1.5),
        squeeze=False,
        height_ratios=height_ratios,
    )
    scenario_positions = range(len(scenarios_in_order))
    scenario_labels = [scenario_label(scenario) for scenario in scenarios_in_order]
    for workload_index, workload in enumerate(workload_names):
        scenarios = data["workloads"][workload]
        block_start = (workload_index // workload_columns) * block_rows
        column = workload_index % workload_columns
        for metric_index, metric in enumerate(metrics_by_workload[workload]):
            axis = axes[block_start + metric_index][column]
            for index, scenario in enumerate(scenarios_in_order):
                record = scenarios[scenario].get(metric)
                if record is None:
                    continue
                values_ms = [value / 1000.0 for value in record["samples"]]
                jitter = [
                    (sample_index - (len(values_ms) - 1) / 2)
                    * 0.16
                    / max(1, len(values_ms) - 1)
                    for sample_index in range(len(values_ms))
                ]
                axis.scatter(
                    [index + delta for delta in jitter],
                    values_ms,
                    color="#4c78a8",
                    alpha=0.55,
                    s=17,
                )
                axis.vlines(
                    index,
                    record["p10_us"] / 1000.0,
                    record["p90_us"] / 1000.0,
                    color="#4c78a8",
                    linewidth=2,
                )
                axis.plot(
                    index,
                    record["median_us"] / 1000.0,
                    "_",
                    color="#c44e52",
                    ms=13,
                    mew=2,
                )
            metric_label = METRIC_LABELS.get(metric, metric)
            title = f"{workload} · {metric_label}" if metric_index == 0 else metric_label
            axis.set_title(title, loc="left", fontsize=9)
            axis.set_xticks(scenario_positions)
            if metric_index == len(metrics_by_workload[workload]) - 1:
                axis.set_xticklabels(scenario_labels, rotation=25, ha="right", fontsize=7)
                axis.set_xlabel("scenario")
            else:
                axis.tick_params(axis="x", labelbottom=False)
            axis.set_xlim(-0.5, len(scenarios_in_order) - 0.5)
            axis.set_ylabel("ms")
            axis.grid(axis="y", alpha=0.25)

        for metric_index in range(len(metrics_by_workload[workload]), metric_rows):
            axes[block_start + metric_index][column].set_axis_off()
        paired_axis = axes[block_start + metric_rows][column]
        paired = data["paired_comparisons"][workload]
        if manifest.get("experiment") == "capacity-scan":
            paired = {
                name: record for name, record in paired.items()
                if name.partition("_minus_")[2].endswith("_k1")
                or name.partition("_minus_")[0].startswith("ltf_")
                and name.partition("_minus_")[2].startswith("fifo_")
            }
        else:
            paired = {
                name: record for name, record in paired.items()
                if name.partition("_minus_")[2] != "phase1_bare"
            }
        pair_labels = []
        for y, (name, record) in enumerate(paired.items()):
            difference = record["metrics"]["workload"]
            values_ms = [value / 1000.0 for value in difference["difference_us"]]
            jitter = [
                (sample_index - (len(values_ms) - 1) / 2)
                * 0.32
                / max(1, len(values_ms) - 1)
                for sample_index in range(len(values_ms))
            ]
            paired_axis.scatter(
                values_ms,
                [y + delta for delta in jitter],
                color="#4c78a8",
                alpha=0.6,
                s=17,
            )
            if values_ms:
                low = _percentile(values_ms, 0.1)
                high = _percentile(values_ms, 0.9)
                center = _percentile(values_ms, 0.5)
                paired_axis.hlines(y, low, high, color="#4c78a8", linewidth=2)
                paired_axis.plot(center, y, "|", color="#c44e52", ms=12, mew=2)
            current, _, baseline = name.partition("_minus_")
            pair_labels.append(f"{scenario_label(current)} − {scenario_label(baseline)}")
        paired_axis.axvline(0, color="black", linewidth=0.8, alpha=0.7)
        paired_axis.set_yticks(range(len(paired)))
        paired_axis.set_yticklabels(pair_labels, fontsize=8)
        paired_axis.set_title("paired workload differences", loc="left", fontsize=9)
        paired_axis.set_xlabel("difference (ms; positive = slower)")
        paired_axis.grid(axis="x", alpha=0.25)
    for workload_index in range(len(workload_names), workload_rows * workload_columns):
        block_start = (workload_index // workload_columns) * block_rows
        column = workload_index % workload_columns
        for row in range(block_rows):
            axes[block_start + row][column].set_axis_off()
    figure.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="None", color="#4c78a8", label="individual runs"),
            Line2D([], [], color="#4c78a8", linewidth=2, label="P10–P90"),
            Line2D([], [], marker="_", linestyle="None", color="#c44e52", ms=13, mew=2, label="median"),
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
    )
    figure.suptitle(
        f"JobPacer run distributions and paired differences · n={manifest['repetitions']} · "
        f"poll={manifest['metadata']['completion_poll_interval_s'] * 1000:g} ms · "
        "P10–P90 is not a confidence interval",
        y=0.99,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.96), h_pad=0.8, w_pad=1.0)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, format="svg")
    plt.close(figure)


def _workload_panels(plt, count: int, *, width: float = 16, height: float = 3.2):
    columns = min(2, count)
    rows = (count + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(width, height * rows), squeeze=False)
    flat = list(axes.flat)
    for axis in flat[count:]:
        axis.set_axis_off()
    return figure, flat[:count]


def _scenario_runs(root: Path, manifest: dict[str, Any], workload: str, scenario: str):
    records = sorted(
        (
            record for record in manifest["runs"]
            if record.get("workload") == workload
            and record.get("scenario") == scenario
            and record.get("status") == "ok"
        ),
        key=lambda record: int(record["repetition"]),
    )
    return [(record, load_trace(root / record["path"])) for record in records]


def _draw_distribution(axis, x: float, values: list[float], color: str, *, scale: float = 1000):
    scaled = [value / scale for value in values]
    jitter = [
        (index - (len(scaled) - 1) / 2) * 0.09 / max(1, len(scaled) - 1)
        for index in range(len(scaled))
    ]
    axis.scatter([x + item for item in jitter], scaled, color=color, alpha=0.5, s=12)
    axis.errorbar(
        x,
        _percentile(scaled, 0.5),
        yerr=[
            [_percentile(scaled, 0.5) - _percentile(scaled, 0.1)],
            [_percentile(scaled, 0.9) - _percentile(scaled, 0.5)],
        ],
        fmt="s",
        color=color,
        capsize=2,
        markersize=4,
        linewidth=1,
    )


def render_capacity_makespan(result_dir: str | Path, output: str | Path) -> None:
    plt, Line2D, _Patch = _matplotlib()
    root = Path(result_dir)
    manifest = load_manifest(root)
    workloads = manifest["workloads"]
    figure, axes = _workload_panels(plt, len(workloads), height=3.0)
    capacities = ("k1", "k2", "k3", "unbounded")
    colors = {"fifo": "#4c78a8", "ltf": "#e17c05"}
    for axis, workload in zip(axes, workloads):
        for policy, offset in (("fifo", -0.12), ("ltf", 0.12)):
            for index, capacity in enumerate(capacities):
                scenario = f"{policy}_{capacity}"
                if scenario not in manifest["scenarios"]:
                    continue
                runs = _scenario_runs(root, manifest, workload, scenario)
                values = [trace["performance"]["workload_makespan_us"] for _, trace in runs]
                _draw_distribution(axis, index + offset, values, colors[policy])
        bare_runs = _scenario_runs(root, manifest, workload, "phase1_bare")
        bare = [trace["performance"]["workload_makespan_us"] / 1000 for _, trace in bare_runs]
        _draw_distribution(axis, 4, [value * 1000 for value in bare], "#555555")
        axis.set_title(f"{workload} · n={manifest['repetitions']}", loc="left", fontsize=9)
        axis.set_xticks(range(5), (*capacities, "bare ref"), rotation=18, ha="right")
        axis.set_xlim(-0.5, 4.5)
        axis.set_ylabel("workload makespan (ms)")
        axis.grid(axis="y", alpha=0.25)
    figure.legend(
        handles=[
            Line2D([], [], marker="s", linestyle="None", color=colors["fifo"], label="FIFO median / P10–P90"),
            Line2D([], [], marker="s", linestyle="None", color=colors["ltf"], label="LTF median / P10–P90"),
            Line2D([], [], marker="s", linestyle="None", color="#555555", label="bare (not a capacity point)"),
            Line2D([], [], marker="o", linestyle="None", color="#777777", label="individual run"),
        ],
        loc="lower center",
        ncol=4,
        frameon=False,
    )
    poll_ms = manifest["metadata"]["completion_poll_interval_s"] * 1000
    figure.suptitle(f"Capacity scan · makespan · poll={poll_ms:g} ms · P10–P90 is not a confidence interval")
    figure.tight_layout(rect=(0, 0.055, 1, 0.96), h_pad=1.0)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, format="svg")
    plt.close(figure)


def render_capacity_job_completion(result_dir: str | Path, output: str | Path) -> None:
    plt, Line2D, _Patch = _matplotlib()
    root = Path(result_dir)
    manifest = load_manifest(root)
    scenarios = scenario_order(root)
    all_jobs = sorted({
        item["job_id"]
        for workload in manifest["workloads"]
        for item in _scenario_runs(root, manifest, workload, scenarios[0])[0][1]["performance"]["job_makespans"]
    })
    job_colors = {job: plt.get_cmap("tab10")(i) for i, job in enumerate(all_jobs)}
    figure, axes = _workload_panels(plt, len(manifest["workloads"]), height=4.2, width=19)
    for axis, workload in zip(axes, manifest["workloads"]):
        sample = _scenario_runs(root, manifest, workload, scenarios[0])[0][1]
        jobs = [item["job_id"] for item in sample["performance"]["job_makespans"]]
        for x, scenario in enumerate(scenarios):
            runs = _scenario_runs(root, manifest, workload, scenario)
            for job_index, job_id in enumerate(jobs):
                values = [
                    next(item["makespan_us"] for item in trace["performance"]["job_makespans"] if item["job_id"] == job_id)
                    for _, trace in runs
                ]
                _draw_distribution(axis, x + (job_index - (len(jobs) - 1) / 2) * 0.12, values, job_colors[job_id])
            means = [trace["performance"]["mean_job_completion_us"] for _, trace in runs]
            _draw_distribution(axis, x + 0.34, means, "#222222")
        axis.set_title(workload, loc="left", fontsize=9)
        axis.set_xticks(range(len(scenarios)), [scenario_label(item) for item in scenarios], rotation=45, ha="right", fontsize=7)
        axis.set_ylabel("job completion from release (ms)")
        axis.grid(axis="y", alpha=0.25)
    legend = [Line2D([], [], marker="s", linestyle="None", color=job_colors[job], label=job) for job in all_jobs]
    legend.append(Line2D([], [], marker="s", linestyle="None", color="#222222", label="run mean"))
    figure.legend(handles=legend, loc="lower center", ncol=min(6, len(legend)), frameon=False)
    figure.suptitle(
        f"Per-job and run-mean completion · n={manifest['repetitions']} · "
        f"poll={manifest['metadata']['completion_poll_interval_s'] * 1000:g} ms · "
        "raw runs, median, P10–P90 (not a confidence interval)"
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.96), h_pad=1.4)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, format="svg")
    plt.close(figure)


def render_capacity_observed_concurrency(result_dir: str | Path, output: str | Path) -> None:
    plt, Line2D, _Patch = _matplotlib()
    root = Path(result_dir)
    manifest = load_manifest(root)
    scenarios = scenario_order(root)
    figure, axes = _workload_panels(plt, len(manifest["workloads"]), height=3.4)
    colors = {"admission": "#e17c05", "launched": "#4c78a8"}
    for axis, workload in zip(axes, manifest["workloads"]):
        for x, scenario in enumerate(scenarios):
            definition = manifest["scenarios"][scenario]
            runs = _scenario_runs(root, manifest, workload, scenario)
            for name, field, offset in (
                ("admission", "peak_admission_occupancy", -0.12),
                ("launched", "peak_launched_inflight", 0.12),
            ):
                values = [
                    item[field]
                    for _, trace in runs
                    for item in trace["performance"]["capacity_observations"]
                ]
                jitter = [(i % 7 - 3) * 0.025 for i in range(len(values))]
                axis.scatter([x + offset + item for item in jitter], values, color=colors[name], alpha=0.55, s=12)
            configured = definition.get("max_outstanding", 0)
            if configured:
                axis.hlines(configured, x - 0.28, x + 0.28, color="#333333", linewidth=1)
        axis.set_title(workload, loc="left", fontsize=9)
        axis.set_xticks(range(len(scenarios)), [scenario_label(item) for item in scenarios], rotation=45, ha="right", fontsize=7)
        axis.set_ylabel("peak tasks per rank")
        axis.grid(axis="y", alpha=0.25)
    figure.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="None", color=colors["admission"], label="peak admission occupancy"),
            Line2D([], [], marker="o", linestyle="None", color=colors["launched"], label="peak launched inflight"),
            Line2D([], [], color="#333333", label="configured finite capacity"),
        ], loc="lower center", ncol=3, frameon=False,
    )
    figure.suptitle(
        f"Observed per-rank concurrency · count · n={manifest['repetitions']} · "
        f"poll={manifest['metadata']['completion_poll_interval_s'] * 1000:g} ms · "
        "bare is a separate execution path"
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.96), h_pad=1.0)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, format="svg")
    plt.close(figure)


def render_capacity_occupancy_time(result_dir: str | Path, output: str | Path) -> None:
    plt, _Line2D, Patch = _matplotlib()
    root = Path(result_dir)
    manifest = load_manifest(root)
    scenarios = scenario_order(root)
    figure, axes = _workload_panels(plt, len(manifest["workloads"]), height=4.0)
    max_count = 0
    for axis, workload in zip(axes, manifest["workloads"]):
        counts = set()
        tables = {}
        for scenario in scenarios:
            observations = [
                item
                for _, trace in _scenario_runs(root, manifest, workload, scenario)
                for item in trace["performance"]["capacity_observations"]
            ]
            tables[scenario] = observations
            for item in observations:
                counts.update(map(int, item["admission_occupancy_by_count_us"]))
                counts.update(map(int, item["launched_inflight_by_count_us"]))
                counts.update(map(int, item["pending_launch_by_count_us"]))
        counts = sorted(counts)
        max_count = max(max_count, max(counts, default=0))
        count_colors = {count: plt.get_cmap("viridis")(index / max(1, len(counts) - 1)) for index, count in enumerate(counts)}
        for x, scenario in enumerate(scenarios):
            observations = tables[scenario]
            for offset, field in (
                (-0.24, "admission_occupancy_by_count_us"),
                (0.0, "launched_inflight_by_count_us"),
                (0.24, "pending_launch_by_count_us"),
            ):
                bottoms = 0.0
                for count in counts:
                    vals = [int(item[field].get(str(count), 0)) / 1000 for item in observations]
                    height = sum(vals) / len(vals) if vals else 0
                    axis.bar(x + offset, height, bottom=bottoms, width=0.22, color=count_colors[count], edgecolor="white", linewidth=0.25)
                    bottoms += height
        axis.set_title(workload, loc="left", fontsize=9)
        axis.set_xticks(range(len(scenarios)), [scenario_label(item) for item in scenarios], rotation=45, ha="right", fontsize=7)
        axis.set_ylabel("mean rank-time by count (ms)")
        axis.grid(axis="y", alpha=0.2)
    figure.legend(handles=[Patch(color=plt.get_cmap("viridis")(count / max(1, max_count)), label=f"occupancy={count}") for count in range(max_count + 1)], loc="lower center", ncol=min(8, max_count + 1), frameon=False, title="stack color = time by count; each scenario groups admission, launched, and pending-launch bars")
    figure.suptitle(
        f"Time-weighted occupancy distribution · ms per rank · n={manifest['repetitions']} · "
        f"poll={manifest['metadata']['completion_poll_interval_s'] * 1000:g} ms · "
        "grouped admission / launched inflight"
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.96), h_pad=1.2)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, format="svg")
    plt.close(figure)


def _render_policy_capacity_metric(result_dir: str | Path, output: str | Path, metric: str) -> None:
    plt, Line2D, _Patch = _matplotlib()
    root = Path(result_dir)
    manifest = load_manifest(root)
    capacities = list(dict.fromkeys(
        scene["capacity_label"]
        for scene in manifest["scenarios"].values()
        if scene.get("mode") == "scheduler"
    ))
    strategies = ("fifo", "ltf", "srjf")
    colors = {"fifo": "#4c78a8", "ltf": "#e17c05", "srjf": "#55a868"}
    figure, axes = _workload_panels(plt, len(manifest["workloads"]), height=3.0)
    field = "workload_makespan_us" if metric == "workload" else "mean_job_completion_us"
    ylabel = "workload makespan (ms)" if metric == "workload" else "mean job completion (ms)"
    for axis, workload in zip(axes, manifest["workloads"]):
        for capacity_index, capacity in enumerate(capacities):
            for policy_index, policy in enumerate(strategies):
                scenario = f"{policy}_{capacity}"
                if scenario not in manifest["scenarios"]:
                    continue
                values = [trace["performance"][field] for _, trace in _scenario_runs(root, manifest, workload, scenario)]
                x = capacity_index + (policy_index - 1) * 0.18
                _draw_distribution(axis, x, values, colors[policy])
        bare = [trace["performance"][field] for _, trace in _scenario_runs(root, manifest, workload, "phase1_bare")]
        _draw_distribution(axis, len(capacities), bare, "#555555")
        axis.set_title(workload, loc="left", fontsize=9)
        axis.set_xticks(range(len(capacities) + 1), [*capacities, "bare ref"])
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
    figure.legend(handles=[Line2D([], [], marker="s", linestyle="None", color=colors[policy], label=policy.upper()) for policy in strategies] + [Line2D([], [], marker="s", linestyle="None", color="#555555", label="bare (separate path)")], loc="lower center", ncol=4, frameon=False)
    figure.suptitle(
        f"Static policy comparison by capacity · {ylabel} · n={manifest['repetitions']} · "
        f"poll={manifest['metadata']['completion_poll_interval_s'] * 1000:g} ms · "
        "P10–P90 is not a confidence interval"
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.96), h_pad=1.0)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, format="svg")
    plt.close(figure)


def render_policy_makespan(result_dir: str | Path, output: str | Path) -> None:
    _render_policy_capacity_metric(result_dir, output, "workload")


def render_policy_mean_job_completion(result_dir: str | Path, output: str | Path) -> None:
    _render_policy_capacity_metric(result_dir, output, "mean_job_completion_us")


def render_policy_per_job_completion(result_dir: str | Path, output: str | Path) -> None:
    plt, Line2D, _Patch = _matplotlib()
    root = Path(result_dir)
    manifest = load_manifest(root)
    scenarios = scenario_order(root)
    figure, axes = _workload_panels(plt, len(manifest["workloads"]), height=4.2, width=19)
    all_jobs = sorted({
        item["job_id"]
        for workload in manifest["workloads"]
        for item in _scenario_runs(root, manifest, workload, scenarios[0])[0][1]["performance"]["job_makespans"]
    })
    job_colors = {job: plt.get_cmap("tab10")(index) for index, job in enumerate(all_jobs)}
    for axis, workload in zip(axes, manifest["workloads"]):
        sample = _scenario_runs(root, manifest, workload, scenarios[0])[0][1]
        jobs = [item["job_id"] for item in sample["performance"]["job_makespans"]]
        for x, scenario in enumerate(scenarios):
            runs = _scenario_runs(root, manifest, workload, scenario)
            for job_index, job_id in enumerate(jobs):
                values = [
                    next(item["makespan_us"] for item in trace["performance"]["job_makespans"] if item["job_id"] == job_id)
                    for _, trace in runs
                ]
                _draw_distribution(axis, x + (job_index - (len(jobs) - 1) / 2) * 0.14, values, job_colors[job_id])
        axis.set_title(workload, loc="left", fontsize=9)
        axis.set_xticks(range(len(scenarios)), [scenario_label(item) for item in scenarios], rotation=45, ha="right", fontsize=7)
        axis.set_ylabel("job completion (ms)")
        axis.grid(axis="y", alpha=0.25)
    figure.legend(handles=[Line2D([], [], marker="s", linestyle="None", color=job_colors[job], label=job) for job in all_jobs], loc="lower center", ncol=min(6, len(all_jobs)), frameon=False)
    figure.suptitle(
        f"Per-job completion by static policy and capacity · ms · n={manifest['repetitions']} · "
        f"poll={manifest['metadata']['completion_poll_interval_s'] * 1000:g} ms · "
        "P10–P90 is not a confidence interval"
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.96), h_pad=1.4)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, format="svg")
    plt.close(figure)


def render_policy_paired_differences(result_dir: str | Path, output: str | Path) -> None:
    plt, Line2D, _Patch = _matplotlib()
    root = Path(result_dir)
    manifest = load_manifest(root)
    data = summary_data(root)
    figure, axes = _workload_panels(plt, len(manifest["workloads"]), height=4.5)
    for axis, workload in zip(axes, manifest["workloads"]):
        pairs = {
            name: record["metrics"]["workload"]
            for name, record in data["paired_comparisons"][workload].items()
            if not name.partition("_minus_")[2] == "phase1_bare"
        }
        for y, (name, record) in enumerate(pairs.items()):
            values = [value / 1000 for value in record["difference_us"]]
            jitter = [(index - (len(values) - 1) / 2) * 0.24 / max(1, len(values) - 1) for index in range(len(values))]
            axis.scatter(values, [y + offset for offset in jitter], color="#4c78a8", alpha=0.6, s=13)
            axis.hlines(y, record["p10_us"] / 1000, record["p90_us"] / 1000, color="#4c78a8", linewidth=2)
            axis.plot(record["median_us"] / 1000, y, "|", color="#c44e52", ms=11, mew=2)
        axis.axvline(0, color="#333333", linewidth=0.8)
        axis.set_yticks(range(len(pairs)), [name.replace("_minus_", " − ") for name in pairs], fontsize=7)
        axis.set_title(workload, loc="left", fontsize=9)
        axis.set_xlabel("current − baseline (ms; positive = slower)")
        axis.grid(axis="x", alpha=0.25)
    figure.suptitle(
        f"Paired same-capacity policy differences · ms · n={manifest['repetitions']} matched · "
        f"poll={manifest['metadata']['completion_poll_interval_s'] * 1000:g} ms · "
        "P10–P90 is not a confidence interval"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96), h_pad=1.4)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, format="svg")
    plt.close(figure)


def render_policy_ready_wait(result_dir: str | Path, output: str | Path) -> None:
    plt, _Line2D, _Patch = _matplotlib()
    root = Path(result_dir)
    manifest = load_manifest(root)
    scenarios = scenario_order(root)
    figure, axes = _workload_panels(plt, len(manifest["workloads"]), height=3.5)
    for axis, workload in zip(axes, manifest["workloads"]):
        for x, scenario in enumerate(scenarios):
            runs = _scenario_runs(root, manifest, workload, scenario)
            values = [
                max(
                    (task["admit_ts"] - task["ready_record_ts"])
                    for rank in trace["ranks"]
                    for job in rank["jobs"] for task in job["tasks"]
                )
                for _, trace in runs
            ]
            _draw_distribution(axis, x, values, "#8172b2")
        axis.set_title(workload, loc="left", fontsize=9)
        axis.set_xticks(range(len(scenarios)), [scenario_label(item) for item in scenarios], rotation=45, ha="right", fontsize=7)
        axis.set_ylabel("longest ready→admit (ms)")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle(
        f"Longest ready-to-admit interval per run · ms · n={manifest['repetitions']} · "
        f"poll={manifest['metadata']['completion_poll_interval_s'] * 1000:g} ms · "
        "P10–P90 is not a confidence interval"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96), h_pad=1.1)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, format="svg")
    plt.close(figure)


def render_policy_scheduler_state(result_dir: str | Path, output: str | Path) -> None:
    plt, _Line2D, Patch = _matplotlib()
    root = Path(result_dir)
    manifest = load_manifest(root)
    scenarios = scenario_order(root)
    states = (
        ("head_of_line_idle_us", "HOL"),
        ("capacity_wait_us", "capacity busy"),
        ("scheduler_delay_us", "scheduler delay"),
        ("pending_launch_us", "pending launch"),
        ("inflight_us", "inflight"),
        ("no_ready_work_us", "no ready work"),
        ("unattributed_wait_us", "unattributed"),
    )
    colors = ["#e17c05", "#f0ad4e", "#8172b2", "#bcbd22", "#4c78a8", "#d9d9d9", "#777777"]
    figure, axes = _workload_panels(plt, len(manifest["workloads"]), height=4.0)
    for axis, workload in zip(axes, manifest["workloads"]):
        for x, scenario in enumerate(scenarios):
            runs = _scenario_runs(root, manifest, workload, scenario)
            summaries = [
                scheduler_state_summary(rank)
                for _, trace in runs for rank in trace["ranks"]
                if rank.get("mode") == "scheduler"
            ]
            bottom = 0.0
            for (field, _label), color in zip(states, colors):
                values = [item[field] / 1000 for item in summaries]
                height = sum(values) / len(values) if values else 0.0
                axis.bar(x, height, bottom=bottom, color=color, width=0.72)
                bottom += height
        axis.set_title(workload, loc="left", fontsize=9)
        axis.set_xticks(range(len(scenarios)), [scenario_label(item) for item in scenarios], rotation=45, ha="right", fontsize=7)
        axis.set_ylabel("mean rank time by state (ms)")
        axis.grid(axis="y", alpha=0.2)
    figure.legend(handles=[Patch(color=color, label=label) for (_, label), color in zip(states, colors)], loc="lower center", ncol=4, frameon=False)
    figure.suptitle(
        f"Scheduler-state attribution · mean rank ms · n={manifest['repetitions']} · "
        f"poll={manifest['metadata']['completion_poll_interval_s'] * 1000:g} ms · "
        "coordinator time remains separate and can overlap"
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.96), h_pad=1.2)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, format="svg")
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    timeline = subparsers.add_parser("timeline")
    timeline.add_argument("--result-dir", type=Path, required=True)
    timeline.add_argument("--workload", required=True)
    timeline.add_argument("--rank", type=int, default=0)
    timeline.add_argument("--output", type=Path, required=True)
    timeline.add_argument("--x-min-ms", type=float)
    timeline.add_argument("--x-max-ms", type=float)
    summary = subparsers.add_parser("summary")
    summary.add_argument("--result-dir", type=Path, required=True)
    summary.add_argument("--output", type=Path, required=True)
    summary.add_argument("--workload", action="append")
    args = parser.parse_args(argv)
    try:
        if args.command == "timeline":
            render_timeline(
                args.result_dir,
                args.workload,
                args.rank,
                args.output,
                x_min_ms=args.x_min_ms,
                x_max_ms=args.x_max_ms,
            )
        else:
            render_summary(args.result_dir, args.output, args.workload)
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"visualization failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
