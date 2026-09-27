import csv
import json
from pathlib import Path

from examples.jobpacer.analysis.visualize_phase3 import (
    collect_comparable_batches,
    collect_main_batches,
    phase3_timeline_data,
    select_representative_runs,
    select_paired_runs,
    write_overhead_diagnostics,
    write_job_paired_summary,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_trace(batch: Path, run_id: str) -> None:
    path = batch / "raw" / f"{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"validation": {"status": "ok"}}), encoding="utf-8")


def test_collect_main_batches_keeps_successful_run_level_makespans(tmp_path):
    root = tmp_path / "results"
    batch = root / "baseline/L0-balanced/20260923-compact-main"
    _write_csv(batch / "summary.csv", [
        {"run_id": "ok", "group": "new", "policy": "fifo", "status": "ok", "makespan_s": "0.03"},
        {"run_id": "failed", "group": "new", "policy": "fifo", "status": "failed", "makespan_s": "0.01"},
    ])

    batches = collect_main_batches(root)

    assert len(batches) == 1
    assert batches[0]["label"] == "L0"
    assert len(batches[0]["rows"]) == 1
    assert batches[0]["total"] == 2


def test_collect_comparable_batches_skips_single_arm_result_sets(tmp_path):
    root = tmp_path / "results"
    paired = root / "readiness/L1/batch"
    _write_csv(paired / "summary.csv", [
        {"group": "new", "policy": "fifo", "status": "ok", "makespan_s": "0.03"},
        {"group": "new", "policy": "ltf", "status": "ok", "makespan_s": "0.02"},
    ])
    single = root / "bridge/G0/batch"
    _write_csv(single / "summary.csv", [
        {"group": "new", "policy": "static_ltf", "status": "ok", "makespan_s": "0.03"},
    ])

    batches = collect_comparable_batches(root)

    assert [batch["path"] for batch in batches] == [paired.resolve()]
    assert batches[0]["label"] == "readiness/L1/batch"


def test_collect_comparable_batches_joins_bridge_and_noise_treatments(tmp_path):
    root = tmp_path / "results"
    bridge = root / "bridge/G0-bridge/batch"
    for mode in ("linear", "dag"):
        _write_csv(bridge / mode / "summary.csv", [
            {"run_id": mode, "group": "new", "policy": "static_ltf", "status": "ok", "makespan_s": "0.02"},
        ])

    noise = root / "noise/L0-balanced/batch"
    for seed in (1, 2):
        for condition in ("A", "B"):
            _write_csv(noise / f"seed-{seed}" / condition / "summary.csv", [
                {"run_id": f"{condition}-{seed}", "group": "new", "policy": "fifo", "status": "ok", "makespan_s": "0.02"},
            ])

    batches = collect_comparable_batches(root)
    by_label = {batch["label"]: batch for batch in batches}

    assert set(by_label) == {"bridge/G0-bridge/batch", "noise/L0-balanced/batch"}
    assert {_row["_plot_arm"] for _row in by_label["bridge/G0-bridge/batch"]["rows"]} == {"linear", "dag"}
    assert {_row["_plot_arm"] for _row in by_label["noise/L0-balanced/batch"]["rows"]} == {"A", "B"}


def test_representative_run_is_successful_trace_nearest_arm_median(tmp_path):
    batch = tmp_path / "batch-main"
    rows = [
        {"run_id": "r0", "group": "new", "policy": "fifo", "status": "ok", "makespan_s": "0.010"},
        {"run_id": "r1", "group": "new", "policy": "fifo", "status": "ok", "makespan_s": "0.011"},
        {"run_id": "r2", "group": "new", "policy": "fifo", "status": "ok", "makespan_s": "0.030"},
        {"run_id": "bad", "group": "new", "policy": "fifo", "status": "failed", "makespan_s": "0.011"},
    ]
    _write_csv(batch / "summary.csv", rows)
    for run_id in ("r0", "r1", "r2"):
        _write_trace(batch, run_id)

    selected = select_representative_runs(batch)

    assert len(selected) == 1
    assert selected[0]["run_id"] == "r1"
    assert selected[0]["arm"] == "new-fifo"


def test_paired_timeline_uses_first_common_seed_repeat_not_median_run(tmp_path):
    batch = tmp_path / "batch"
    rows = []
    for epoch, makespan in ((8, 0.001), (2, 100.0)):
        for arm in ("old-ltf", "new-ltf-precreate"):
            run_id = f"{arm}-e{epoch}"
            rows.append({"run_id": run_id, "arm": arm, "epoch": epoch, "repeat": 0,
                         "status": "ok", "makespan_s": str(makespan)})
            _write_trace(batch, run_id)

    block, selected = select_paired_runs(batch, rows)

    assert block == (2, 0)
    assert {item["arm"] for item in selected} == {"old-ltf", "new-ltf-precreate"}
    assert {item["run_id"] for item in selected} == {"old-ltf-e2", "new-ltf-precreate-e2"}


def test_overhead_diagnostics_exports_rank_local_polling_costs(tmp_path):
    batch = tmp_path / "batch"
    run_id = "new-ltf-poll-0.2ms-e5300-r0"
    _write_csv(batch / "summary.csv", [{
        "run_id": run_id, "arm": "new-ltf-poll-0.2ms", "epoch": "5300", "repeat": "0",
        "poll_interval_s": "0.0002", "status": "ok", "makespan_s": "0.021",
    }])
    trace = {
        "metrics": {
            "rank_task_timings": {"0": {"job-0/comm-0": {
                "application_wait_s": 0.004, "completion_probe_count": 8,
                "completion_observed_to_application_continue_s": 0.0002,
                "call_return_to_completion_observation_s": 0.002,
            }}},
            "coordinator_task_timings": {"job-0/comm-0": {
                "grant_to_all_submitted_s": 0.001,
                "all_submitted_to_all_completed_s": 0.003,
                "all_completed_to_next_grant_s": 0.0005,
                "eligible_to_grant_s": 0.0,
                "legal_candidate_present_during_gap": True,
                "next_task_id": "job-0/comm-1",
            }},
        },
        "ranks": [{
            "rank": 0, "application_makespan_us": 20000,
            "communication_drain_makespan_us": 21000, "preparation_total_us": 3000,
            "binding_creation_total_us": 1000, "process_cpu_time_s": 0.012,
            "voluntary_context_switches": 24, "involuntary_context_switches": 1,
        }],
    }
    raw = batch / "raw" / f"{run_id}.json"
    raw.parent.mkdir(parents=True)
    raw.write_text(json.dumps(trace), encoding="utf-8")

    outputs = write_overhead_diagnostics(batch)

    assert (batch / "analysis.md").is_file()
    assert len(outputs) == 5
    with (batch / "process-diagnostics.csv").open(encoding="utf-8", newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["rank"] == "0"
    assert float(row["median_application_wait_ms"]) == 4.0
    assert float(row["completion_probe_count"]) == 8.0
    assert float(row["process_cpu_ms"]) == 12.0
    with (batch / "coordinator-diagnostics.csv").open(encoding="utf-8", newline="") as stream:
        coordinator = next(csv.DictReader(stream))
    assert coordinator["legal_candidate_present_during_gap"] == "True"
    assert float(coordinator["all_completed_to_next_grant_ms"]) == 0.5


def test_job_paired_summary_uses_common_repeats_then_seed_medians(tmp_path):
    batch = tmp_path / "batch"
    rows = []
    for seed in (10, 20):
        for repeat, values in ((0, (0.010, 0.008)), (1, (0.012, 0.009))):
            for arm, jct in zip(("new-static_fifo", "new-fifo"), values):
                rows.append({"run_id": f"{arm}-{seed}-{repeat}", "arm": arm,
                             "epoch": seed, "repeat": repeat, "job_id": "job-0",
                             "jct_s": jct, "status": "ok"})
    _write_csv(batch / "jobs.csv", rows)
    (batch / "analysis.json").write_text(json.dumps({
        "bootstrap_samples": 100, "tie_threshold": 0.0,
    }), encoding="utf-8")

    outputs = write_job_paired_summary(batch)

    assert len(outputs) == 2
    with (batch / "job-paired-summary.csv").open(encoding="utf-8", newline="") as stream:
        result = next(csv.DictReader(stream))
    assert result["baseline"] == "new-static_fifo"
    assert result["candidate"] == "new-fifo"
    assert result["paired_seeds"] == "2"
    assert float(result["baseline_median_jct_ms"]) == 11.0
    assert float(result["candidate_median_jct_ms"]) == 8.5
    assert float(result["median_speedup"]) > 1.0


def test_linear_runtime_timeline_uses_rank_local_release_and_runtime_events():
    trace = {
        "input_mode": "linear",
        "ranks": [{
            "rank": 0,
            "application_release_ts": 1000,
            "application_makespan_us": 700,
            "validation_start_ts": 1600,
            "validation_end_ts": 1700,
            "jobs": [{"job_id": "job-0", "tasks": [{
                "task_id": "job-0/comm-0",
                "ordinal": 0,
                "producer_start_ts": 1100,
                "ready_ts": 1200,
                "submit_call_ts": 1250,
                "submit_return_ts": 1310,
                "consumer_start_ts": 1320,
                "first_wait_ts": 1400,
                "consumer_end_ts": 1500,
            }]}],
            "runtime_events": [
                {"task_id": "job-0/comm-0", "kind": "declare_call_start", "time_us": 1210},
                {"task_id": "job-0/comm-0", "kind": "declare_call_end", "time_us": 1220},
                {"task_id": "job-0/comm-0", "kind": "grant_received", "time_us": 1300},
                {"task_id": "job-0/comm-0", "kind": "launch_start", "time_us": 1350},
                {"task_id": "job-0/comm-0", "kind": "completion_observed", "time_us": 1450},
            ],
        }],
    }

    data = phase3_timeline_data(trace, 0)
    intervals = data["jobs"][0]["intervals"]

    assert [(item["kind"], item["start_ms"], item["end_ms"]) for item in intervals] == [
        ("DECLARE call", 0.21, 0.22),
        ("producer", 0.1, 0.2),
        ("producer ready → submit call", 0.2, 0.25),
        ("submit API call", 0.25, 0.31),
        ("submit call → grant/admit", 0.25, 0.3),
        ("grant/admit → collective start", 0.3, 0.35),
        ("collective start → completion observation", 0.35, 0.45),
        ("consumer", 0.32, 0.4),
        ("application/backend wait", 0.4, 0.5),
    ]
    marker = data["jobs"][0]["markers"][0]
    assert marker["declare_call_start_ms"] == 0.21
    assert marker["declare_call_end_ms"] == 0.22
    assert data["validation"] == {"start_ms": 0.6, "end_ms": 0.7}
    assert data["makespan_ms"] == 0.7


def test_legacy_timeline_uses_phase12_fields_and_separate_rank_release():
    trace = {
        "input_mode": "linear",
        "ranks": [{
            "rank": 1,
            "application_release_ts": 5000,
            "jobs": [{"job_id": "job-1", "tasks": [{
                "ordinal": 0,
                "producer_compute_start_ts": 5100,
                "ready_record_ts": 5200,
                "submit_api_start_ts": 5250,
                "submit_api_return_ts": 5260,
                "admit_ts": 5300,
                "collective_call_start_ts": 5310,
                "completion_observed_ts": 5400,
                "consumer_compute_start_ts": 5350,
                "consumer_compute_end_ts": 5450,
                "application_wait_start_ts": 5460,
                "wait_return_ts": 5600,
            }]}],
            "runtime_events": [],
        }],
    }

    intervals = phase3_timeline_data(trace, 1)["jobs"][0]["intervals"]

    assert [(item["kind"], item["start_ms"], item["end_ms"]) for item in intervals] == [
        ("producer", 0.1, 0.2),
        ("producer ready → submit call", 0.2, 0.25),
        ("submit API call", 0.25, 0.26),
        ("submit call → grant/admit", 0.25, 0.3),
        ("grant/admit → collective start", 0.3, 0.31),
        ("collective start → completion observation", 0.31, 0.4),
        ("consumer", 0.35, 0.45),
        ("application/backend wait", 0.46, 0.6),
    ]


def test_dag_timeline_reads_node_events_not_placeholder_task_records():
    trace = {
        "input_mode": "dag",
        "ranks": [{
            "rank": 0,
            "application_release_ts": 100,
            "jobs": [{"job_id": "job-0", "tasks": [{"task_id": "job-0/comm-0"}]}],
            "runtime_events": [
                {"task_id": "job-0/comm-0", "kind": "grant_received", "time_us": 135},
                {"task_id": "job-0/comm-0", "kind": "launch_start", "time_us": 137},
                {"task_id": "job-0/comm-0", "kind": "completion_observed", "time_us": 150},
            ],
            "dag_events": [
                {"job_id": "job-0", "node_id": "calc", "kind": "compute_started", "time_us": 110},
                {"job_id": "job-0", "node_id": "calc", "kind": "compute_completed", "time_us": 120},
                {"job_id": "job-0", "node_id": "comm-0", "kind": "node_ready", "time_us": 130},
                {"job_id": "job-0", "node_id": "comm-0", "kind": "comm_submit_return", "time_us": 145},
                {"job_id": "job-0", "node_id": "comm-0", "kind": "comm_completed_observed", "time_us": 160},
            ],
        }],
    }

    data = phase3_timeline_data(trace, 0)
    intervals = data["jobs"][0]["intervals"]

    assert {(item["lane"], item["kind"], item["start_ms"], item["end_ms"]) for item in intervals} == {
        ("compute", "compute", 0.01, 0.02),
        ("admission", "ready → grant/admit", 0.03, 0.035),
        ("communication", "collective → completion observed", 0.037, 0.05),
        ("application_wait", "application/backend wait", 0.045, 0.06),
    }
