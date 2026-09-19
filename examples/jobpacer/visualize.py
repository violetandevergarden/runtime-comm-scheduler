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


def _manifest_paths(result_dir: str | Path, workload: str, scenario: str) -> list[Path] | None:
    manifest_path = Path(result_dir) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"batch manifest is required: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("complete", False):
        raise ValueError(f"batch manifest is incomplete: {manifest_path}")
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
        for scenario in SCENARIOS
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
            for name in ("ready_record_ts", "admit_ts", "completion_observed_ts")
        )
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
        inflight = {
            tuple(task["key"])
            for task in tasks
            if task["collective_call_start_ts"] <= start < task["completion_observed_ts"]
        }
        if not ready and not inflight:
            state = "no_ready_work"
        elif max_outstanding and len(inflight) >= max_outstanding:
            state = "capacity_busy"
        elif inflight:
            state = "inflight"
        elif rank_trace.get("selection") == "ready_first":
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
        "scheduler_delay_us": 0.0,
        "coordination_wait_us": 0.0,
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
        elif state == "ready_head_scheduler_delay":
            summary["scheduler_delay_us"] += duration
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
                    "communication",
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
    root = Path(result_dir) / "raw"
    names = sorted(workloads) if workloads is not None else sorted(
        path.name for path in root.iterdir() if path.is_dir()
    )
    result: dict[str, Any] = {"workloads": {}, "paired_comparisons": {}}
    for workload in names:
        scenarios: dict[str, Any] = {}
        for scenario in SCENARIOS:
            paths = trace_paths(root.parent, workload, scenario)
            traces = [load_trace(path) for path in paths]
            for path, trace in zip(paths, traces):
                validate_trace(trace, path)
            metrics = ["workload"]
            metrics.extend(
                item["job_id"]
                for item in traces[0].get("performance", {}).get("job_makespans", [])
            )
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
            f"{current}_minus_{baseline}": paired_differences(
                result_dir, workload, baseline, current
            )
            for current, baseline in PAIRED_SCENARIO_PAIRS
        }
    return result


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
    selected = representative_runs(result_dir, workload)
    data_by_scenario = {
        scenario: timeline_data(load_trace(path), rank)
        for scenario, path in selected.items()
    }
    job_count = max(len(data["jobs"]) for data in data_by_scenario.values())
    height = max(4.0, 1.4 + 0.8 * job_count)
    figure, axes = plt.subplots(1, len(SCENARIOS), figsize=(18, height), sharey=True)
    axes = [axes] if len(SCENARIOS) == 1 else list(axes)
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
    }
    for axis, scenario in zip(axes, SCENARIOS):
        data = data_by_scenario[scenario]
        labels = []
        for job_index, job in enumerate(data["jobs"]):
            base_y = job_index * len(lane_order)
            labels.extend(
                f"{job['job_id']} {lane}"
                for lane in ("compute", "admission", "communication", "application", "backend", "validation")
            )
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
        coordination_y = len(data["jobs"]) * len(lane_order)
        state_y = coordination_y + 1
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
            f"{SCENARIO_LABELS[scenario]}\n"
            f"{selected[scenario].name}, rank {rank}, "
            f"makespan {data['makespan_ms']:.3f} ms"
        )
        axis.set_xlabel("relative time (ms)")
        axis.set_yticks(range(len(labels)))
        axis.set_yticklabels(labels)
        axis.grid(axis="x", alpha=0.25)
        if x_min_ms is not None or x_max_ms is not None:
            axis.set_xlim(x_min_ms, x_max_ms)
    legend = [
        Patch(color=colors["producer"], label="producer compute"),
        Patch(color=colors["consumer"], label="consumer overlap compute"),
        Patch(color=colors["admission"], label="ready → admit"),
        Patch(color=colors["communication"], label="submit → complete"),
        Patch(color=colors["application_wait"], label="application wait → binding"),
        Patch(color=colors["wait"], label="underlying wait → return"),
        Patch(color=colors["validation"], label="harness validation"),
        Patch(color="#8172b2", label="control coordination"),
        Patch(color=state_colors["head_not_ready_with_later_ready"], label="scheduler HOL idle"),
        Line2D([], [], marker="^", color="black", linestyle="None", label="ready"),
        Line2D([], [], marker="|", color="black", linestyle="None", label="admit / submit / complete"),
    ]
    figure.legend(handles=legend, loc="lower center", ncol=4)
    figure.suptitle(f"JobPacer timeline: {workload}", y=1.01)
    figure.tight_layout(rect=(0, 0.08, 1, 0.98))
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, format="svg")
    plt.close(figure)


def render_summary(
    result_dir: str | Path, output: str | Path, workloads: Iterable[str] | None = None
) -> None:
    plt, Line2D, _Patch = _matplotlib()
    data = summary_data(result_dir, workloads)
    workload_names = list(data["workloads"])
    if not workload_names:
        raise ValueError("no workloads found")
    figure, axes = plt.subplots(
        2,
        len(workload_names),
        figsize=(6 * len(workload_names), 8),
        squeeze=False,
        height_ratios=(2, 1),
    )
    metric_colors = {"workload": "#4c78a8", "job-0": "#55a868", "job-1": "#c44e52"}
    offsets = {"workload": -0.18, "job-0": 0.0, "job-1": 0.18}
    for workload_index, (axis, workload) in enumerate(zip(axes[0], workload_names)):
        scenarios = data["workloads"][workload]
        metrics = list(next(iter(scenarios.values())))
        for metric in metrics:
            color = metric_colors.get(metric, "#8172b2")
            offset = offsets.get(metric, 0.0)
            for index, scenario in enumerate(SCENARIOS):
                record = scenarios[scenario].get(metric)
                if record is None:
                    continue
                x = index + offset
                values_ms = [value / 1000.0 for value in record["samples"]]
                axis.scatter([x] * len(values_ms), values_ms, color=color, alpha=0.55, s=18)
                axis.vlines(
                    x,
                    record["p10_us"] / 1000.0,
                    record["p90_us"] / 1000.0,
                    color=color,
                    linewidth=2,
                )
                axis.plot(x, record["median_us"] / 1000.0, "_", color=color, ms=16, mew=2)
        sample_count = next(iter(scenarios[SCENARIOS[0]].values()))["count"]
        axis.set_title(f"{workload}\nn={sample_count} per scenario")
        axis.set_xticks(range(len(SCENARIOS)))
        axis.set_xticklabels(
            [SCENARIO_LABELS[scenario] for scenario in SCENARIOS],
            rotation=35,
            ha="right",
        )
        axis.set_ylabel("makespan (ms)")
        axis.grid(axis="y", alpha=0.25)
        paired_axis = axes[1][workload_index]
        paired = data["paired_comparisons"][workload]
        for y, (name, record) in enumerate(paired.items()):
            values_ms = [value / 1000.0 for value in record["difference_us"]]
            paired_axis.scatter(values_ms, [y] * len(values_ms), alpha=0.65, s=20)
            if values_ms:
                low = _percentile(values_ms, 0.1)
                high = _percentile(values_ms, 0.9)
                center = _percentile(values_ms, 0.5)
                paired_axis.hlines(y, low, high, color="#4c78a8", linewidth=2)
                paired_axis.plot(center, y, "|", color="#c44e52", ms=12, mew=2)
        paired_axis.axvline(0, color="black", linewidth=0.8, alpha=0.7)
        paired_axis.set_yticks(range(len(paired)))
        paired_axis.set_yticklabels(list(paired))
        paired_axis.set_xlabel("paired workload difference (ms; positive = slower)")
        paired_axis.grid(axis="x", alpha=0.25)
    handles = [
        Line2D([], [], marker="o", linestyle="None", color=color, label=METRIC_LABELS.get(metric, metric))
        for metric, color in metric_colors.items()
    ]
    figure.legend(handles=handles, loc="upper center", ncol=3)
    figure.suptitle("JobPacer run distributions and paired differences", y=1.01)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
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
