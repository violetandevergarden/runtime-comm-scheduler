"""Run and audit the independent NCCL G1 qualification for the seventh arm."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from examples.jobpacer.experiments.phase3.batch import (
    source_snapshot, _reserve_run_attempt, _run_child, _verify_frozen_inputs,
)
from examples.jobpacer.experiments.phase3.suite import BARE_ARM, SCENARIOS


ROOT = Path(__file__).resolve().parents[4]
CONTRACT_VERSION = "bare-ordered-v2-layered-round-robin"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def _uuid(value: str | None) -> str | None:
    if value is None:
        return None
    return value.removeprefix("GPU-")


def _normal_run_checks(raw: dict[str, Any], scenario: str,
                       sample: dict[str, Any], expected_uuids: set[str]) -> dict[str, bool]:
    validation = raw.get("validation", {})
    ranks = raw.get("ranks", [])
    check: dict[str, bool] = {
        "successful_exit_validation": validation.get("status") == "ok" and not validation.get("errors"),
        "two_rank_results": len(ranks) == 2 and {row.get("rank") for row in ranks} == {0, 1},
        "all_collectives_correct": validation.get("all_collectives_correct") is True,
    }
    if not check["two_rank_results"]:
        return check
    rank_orders = [tuple(row.get("bare_launch_sequence", ())) for row in ranks]
    check["common_launch_order"] = rank_orders[0] == rank_orders[1]
    check["nonempty_exact_task_coverage"] = all(
        len(row.get("expected_task_ids", [])) == len(row.get("bare_launch_sequence", []))
        and set(row.get("expected_task_ids", [])) == set(row.get("bare_launch_sequence", []))
        for row in ranks
    )
    group_ranks = {
        group["group_id"]: tuple(group["ranks"])
        for group in ranks[0].get("groups", [])
    }
    task_groups = {
        f"{job['job_id']}/{node['node_id']}": (node["group_id"], node["group_seq"])
        for job in ranks[0].get("canonical_dag", {}).get("jobs", [])
        for node in job.get("nodes", []) if node.get("kind") == "comm"
    }
    expected_order = tuple(sample.get("profiled_fifo_order", sample.get("fifo_sequence", ())))
    for rank in ranks:
        local_tasks = set(rank.get("expected_task_ids", []))
        if tuple(task_id for task_id in expected_order if task_id in local_tasks) != tuple(
                rank.get("bare_launch_sequence", ())):
            check["common_fifo_projection"] = False
            break
    else:
        check["common_fifo_projection"] = True
    check["group_sequence_projection"] = all(
        tuple(task_id for task_id in row.get("bare_launch_sequence", ())
              if task_groups.get(task_id, (None, None))[0] == group_id)
        == tuple(task_id for _seq, task_id in sorted(
            (seq, task_id) for task_id, (candidate_group, seq) in task_groups.items()
            if candidate_group == group_id
        ))
        for row in ranks for group_id, members in group_ranks.items()
        if row.get("rank") in members
    )
    check["bare_execution_contract"] = all(
        row.get("comm_engine") == "bare" and row.get("max_inflight") is None
        and row.get("bare_contract") == CONTRACT_VERSION
        and row.get("nccl_launch_order_implicit") == "1" for row in ranks)
    check["physical_drain_after_application_end"] = all(
        isinstance(row.get("application_end_ts"), int)
        and isinstance(row.get("communication_drain_end_ts"), int)
        and row["application_end_ts"] <= row["communication_drain_end_ts"]
        for row in ranks
    )
    check["gpu_buffer_validation"] = all(
        all(value is True for checks in (row.get("gpu_dag_validation") or {}).values()
            for value in checks.values())
        for row in ranks
    )
    check["terminal_nodes_completed"] = all(
        set(row.get("expected_node_ids", []))
        == {node_id for job in row.get("jobs", []) for node_id in job.get("completed_node_ids", [])}
        for row in ranks
    )
    check["device_uuids_match_profile"] = {
        _uuid(row.get("device_uuid")) for row in ranks
    } == expected_uuids
    if scenario == "L1-skew-tail":
        progress_during_head_wait = []
        for row in ranks:
            blocked = [event for event in row.get("runtime_events", [])
                       if event.get("kind") == "bare_order_head_wait"]
            released = [event for event in row.get("runtime_events", [])
                        if event.get("kind") == "bare_order_head_released"]
            compute_done = [event for event in row.get("dag_events", [])
                            if event.get("kind") == "compute_completed"]
            progress_during_head_wait.append(any(
                begin.get("task_id") == end.get("task_id")
                and begin.get("time_us") <= done.get("time_us") <= end.get("time_us")
                for begin in blocked for end in released for done in compute_done
            ))
        check["independent_compute_progress_during_static_head_wait"] = all(progress_during_head_wait)
    else:
        check["independent_compute_progress_during_static_head_wait"] = True
    check["multi_group_input"] = len(group_ranks) >= 2
    return check


def _mechanism_checks(raw: dict[str, Any]) -> dict[str, bool]:
    ranks = raw.get("ranks", [])
    return {
        "contract": raw.get("contract") == CONTRACT_VERSION,
        "two_ranks": len(ranks) == 2 and {row.get("rank") for row in ranks} == {0, 1},
        "five_repeats": raw.get("repeats_per_mode_per_rank", 0) >= 5,
        "backend": bool(ranks) and all(row.get("nccl_launch_order_implicit") == "1"
            and tuple(row.get("nccl_version", [])) >= (2, 26, 0) for row in ranks),
        "numeric": bool(ranks) and all(row.get("samples") and all(
            sample.get("numeric_checks") and all(sample["numeric_checks"].values())
            for sample in row["samples"]) for row in ranks),
        "launch_before_completion_observation": bool(ranks) and all(any(
            sample.get("mode") == "multi-inflight" and len(sample.get("api_calls", [])) == 2
            and [item["collective"] for item in sample["api_calls"]] == ["A", "B"]
            and "A" in sample["api_calls"][1].get("prior_pending_at_launch_probe", [])
            and sample["api_calls"][1]["api_start_ns"] < sample["api_calls"][0]["physical_complete_observed_ns"]
            for sample in row.get("samples", [])) for row in ranks),
    }


def _fault_run_checks(output_dir: Path, record: dict[str, Any],
                      expected_message: str) -> dict[str, bool]:
    raw_path = output_dir / record["raw_path"]
    raw = json.loads(raw_path.read_text()) if raw_path.is_file() else {}
    validation = raw.get("validation", {})
    errors = "\n".join(validation.get("errors", []))
    log = "\n".join(((output_dir / record["stdout_path"]).read_text(),
                     (output_dir / record["stderr_path"]).read_text()))
    config = raw.get("config", {})
    return {
        "returned_failure": record.get("returncode") not in {0, None},
        "process_group_exited": record.get("child_process_group_exited") is True,
        "bounded_by_replay_deadline": record.get("duration_s", float("inf")) <= record["timeout_s"],
        "fault_requested_and_recorded": config.get("fault") == record.get("fault"),
        "fault_observed": expected_message in errors or expected_message in log,
        "failed_validation_recorded": validation.get("status") == "failed",
        "fail_stop_recorded": "fail-stop after collective error" in log,
    }


def qualify_g1(suite_manifest_path: Path, output_dir: Path, *, timeout_s: float = 60.0,
               setup_timeout_s: float = 30.0, warmup: int = 5,
               mechanism_evidence: Path | None = None) -> dict[str, Any]:
    suite_manifest_path = suite_manifest_path.resolve(strict=True)
    suite_root = suite_manifest_path.parent
    suite = json.loads(suite_manifest_path.read_text())
    if suite.get("scope") != "pilot":
        raise ValueError("G1 qualification requires the separately seeded pilot suite")
    if suite.get("status") not in {"frozen-inputs-pilot-awaiting-batch", "profiled-awaiting-E2-pilot"}:
        raise ValueError("G1 qualification requires frozen, profiled pilot inputs")
    _verify_frozen_inputs(suite_root, suite)
    if mechanism_evidence is None:
        raise ValueError("bare qualification requires independent multi-inflight mechanism evidence")
    mechanism = json.loads(mechanism_evidence.read_text())
    mechanism_checks = _mechanism_checks(mechanism)
    samples = {}
    for sample in suite.get("samples", []):
        samples.setdefault(sample["scenario"], sample)
    required = SCENARIOS
    if any(scenario not in samples for scenario in required):
        raise ValueError("pilot suite lacks one of the required L1/D1/D3 G1 inputs")
    profile_paths = {
        key: suite_root / suite["profiles"][key]["path"]
        for key in ("communication", "compute")
    }
    for path in profile_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    compute_profile = json.loads(profile_paths["compute"].read_text())
    expected_uuids = {_uuid(row.get("device_uuid")) for row in compute_profile.get("devices", [])}
    if len(expected_uuids) != 2 or None in expected_uuids:
        raise ValueError("compute profile must identify the two target GPU UUIDs")

    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    (output_dir / "raw").mkdir(parents=True)
    (output_dir / "logs").mkdir()
    command_runs: dict[str, dict[str, Any]] = {}

    def execute(run_id: str, scenario: str, fault: str = "none") -> dict[str, Any]:
        sample = samples[scenario]
        reservation = _reserve_run_attempt({"run_id": run_id, "block_id": run_id}, output_dir, 1)
        raw_path = output_dir / reservation["raw_path"]
        stdout_path = output_dir / reservation["stdout_path"]
        stderr_path = output_dir / reservation["stderr_path"]
        command = [
            sys.executable, "-m", "examples.jobpacer.runtime.replay_launcher",
            "--policy", "bare", "--comm-engine", "bare", "--startup-attempts", "1",
            "--static-order", str(suite_root / sample["fifo_order_path"]),
            "--dag", str(suite_root / sample["path"]), "--backend", "nccl",
            "--world-size", "2", "--timeout", str(timeout_s),
            "--setup-timeout", str(setup_timeout_s), "--warmup-iterations", str(warmup),
            "--matmul-precision", "highest", "--comm-profile",
            str(profile_paths["communication"]), "--compute-profile", str(profile_paths["compute"]),
            "--output", str(raw_path), "--fault", fault,
        ]
        environment = dict(os.environ)
        python_path = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = os.pathsep.join(
            item for item in ("src", ".", python_path) if item
        )
        started = time.monotonic()
        returncode, stdout, stderr, timed_out, group_exited = _run_child(
            command, cwd=ROOT, env=environment, timeout_s=setup_timeout_s + timeout_s + 5.0,
            batch_dir=output_dir, reservation=reservation)
        duration_s = time.monotonic() - started
        stdout_path.write_text(stdout)
        stderr_path.write_text(stderr)
        record = {
            "run_id": run_id, "scenario": scenario, "fault": fault,
            "command": command, "returncode": returncode, "duration_s": duration_s,
            "timeout_s": setup_timeout_s + timeout_s + 5.0,
            "timed_out": timed_out, "child_process_group_exited": group_exited,
            "raw_path": raw_path.relative_to(output_dir).as_posix(),
            "stdout_path": stdout_path.relative_to(output_dir).as_posix(),
            "stderr_path": stderr_path.relative_to(output_dir).as_posix(),
            "raw_sha256": _sha(raw_path.read_bytes()) if raw_path.is_file() else None,
            "stdout_sha256": _sha(stdout.encode()), "stderr_sha256": _sha(stderr.encode()),
        }
        command_runs[run_id] = record
        return record

    normal_checks: dict[str, dict[str, bool]] = {}
    for scenario in required:
        record = execute(f"{scenario}-normal", scenario)
        raw_path = output_dir / record["raw_path"]
        if not raw_path.is_file():
            normal_checks[scenario] = {"raw_output_present": False}
            continue
        raw = json.loads(raw_path.read_text())
        normal_checks[scenario] = _normal_run_checks(raw, scenario, samples[scenario], expected_uuids)
        normal_checks[scenario]["runner_exit_zero"] = record["returncode"] == 0

    fault_specs = (
        ("L1-skew-tail-missing-task", "L1-skew-tail", "missing_task", "missing"),
        ("L1-skew-tail-binding-failure", "L1-skew-tail", "binding_failure", "injected DAG binding failure"),
        ("D1-asymmetric-frontiers-launch-failure", "D1-asymmetric-frontiers", "launch_failure", "injected launch failure"),
        ("D3-order-and-sinks-probe-failure", "D3-order-and-sinks", "completion_probe_failure",
         "injected completion probe failure"),
    )
    fault_checks: dict[str, dict[str, bool]] = {}
    for run_id, scenario, fault, expected_message in fault_specs:
        record = execute(run_id, scenario, fault)
        fault_checks[run_id] = _fault_run_checks(output_dir, record, expected_message)

    profile_sha = _sha(json.dumps({key: _sha(path.read_bytes())
                                   for key, path in sorted(profile_paths.items())},
                                  sort_keys=True, separators=(",", ":")).encode())
    snapshot = source_snapshot()
    all_checks = [value for checks in normal_checks.values() for value in checks.values()]
    all_checks.extend(value for checks in fault_checks.values() for value in checks.values())
    all_checks.extend(mechanism_checks.values())
    audit = {
        "schema": "jobpacer-gpu-seven-arm-g1-audit", "schema_version": 1,
        "arm": BARE_ARM, "contract_version": CONTRACT_VERSION,
        "suite_manifest_sha256_before_qualification": _sha(suite_manifest_path.read_bytes()),
        "source_snapshot_sha256": snapshot["digest"], "profile_sha256": profile_sha,
        "device_uuids": sorted(expected_uuids), "normal_checks": normal_checks,
        "mechanism_checks": mechanism_checks,
        "mechanism_evidence": {"path": str(mechanism_evidence.resolve()),
                               "sha256": _sha(mechanism_evidence.read_bytes())},
        "fault_checks": fault_checks, "attempts": command_runs,
        "passed": bool(all_checks) and all(all_checks),
        "scope": "six scenario normal NCCL paths, missing request and binding/launch/probe failures",
        "limitations": ["network disconnect not tested", "repeated epoch not tested"],
    }
    audit_path = output_dir / "qualification-audit.json"
    _write_json(audit_path, audit)
    if audit["passed"]:
        suite["bare_qualification"] = {
            "arm": BARE_ARM, "status": "qualified",
            "evidence": {
                "evidence_path": str(audit_path.resolve()),
                "evidence_sha256": _sha(audit_path.read_bytes()),
                "source_snapshot_sha256": snapshot["digest"],
                "profile_sha256": profile_sha, "device_uuids": sorted(expected_uuids),
                "contract_version": CONTRACT_VERSION,
            },
        }
    else:
        suite["bare_qualification"] = {
            "arm": BARE_ARM, "status": "unqualified", "evidence": None,
        }
    _write_json(suite_manifest_path, suite)
    return audit
