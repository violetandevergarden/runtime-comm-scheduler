"""Freeze, run, resume, and analyze paired Phase 3 GPU seven-arm blocks."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import signal
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from examples.jobpacer.experiments.seven_arm.suite import (
    ARMS,
    FORMAL_WORKLOAD_SEEDS,
    MECHANISM_PILOT_ARMS,
    BARE_ARM,
    CONTRACT_VERSION,
    PILOT_WORKLOAD_SEEDS,
    SCENARIOS,
    ORDER_SEED,
    make_order_table,
    verify_order_table,
)


ROOT = Path(__file__).resolve().parents[4]
RUNNER = ROOT / "examples/jobpacer/scripts/run_phase3.py"
ARM_CONFIG = {
    BARE_ARM: ("bare", "bare", "fifo_order_path"),
    "old-static-fifo": ("old", "static_fifo", "fifo_order_path"),
    "old-static-ltf": ("old", "static_ltf", "ltf_order_path"),
    "new-static-fifo": ("new", "static_fifo", "fifo_order_path"),
    "new-static-ltf": ("new", "static_ltf", "ltf_order_path"),
    "new-dynamic-fifo": ("new", "fifo", None),
    "new-dynamic-ltf": ("new", "ltf", None),
}
ENV_RETRY_MARKERS = ("EADDRINUSE", "Address already in use", "errno 98", "port is already in use")
SOURCE_PATHS = (
    ROOT / "pyproject.toml",
    ROOT / "src/runtime_comm_scheduler",
    ROOT / "examples/jobpacer/runtime",
    ROOT / "examples/jobpacer/gpu",
    ROOT / "examples/jobpacer/diagnostics",
    ROOT / "examples/jobpacer/scripts",
    ROOT / "examples/jobpacer/experiments",
    ROOT / "examples/jobpacer/analysis/runtime_results.py",
    ROOT / "examples/jobpacer/analysis/benchmark_paths.py",
    ROOT / "examples/jobpacer/comm_profile.py",
    ROOT / "examples/jobpacer/workloads.py",
    ROOT / "docs/JobPacer/plan/phase3-gpu-seven-arm-experiments.md",
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                                text=True, check=False)
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def source_snapshot(destination: Path | None = None) -> dict[str, Any]:
    files: list[dict[str, str]] = []
    for source in SOURCE_PATHS:
        candidates = sorted(source.rglob("*.py")) if source.is_dir() else [source]
        for path in candidates:
            if not path.is_file():
                continue
            relative = path.relative_to(ROOT).as_posix()
            data = path.read_bytes()
            files.append({"path": relative, "sha256": _sha(data)})
            if destination is not None:
                archived = destination / relative
                archived.parent.mkdir(parents=True, exist_ok=True)
                archived.write_bytes(data)
    digest_data = "\n".join(f"{row['path']} {row['sha256']}" for row in files).encode()
    return {"digest": _sha(digest_data), "files": files}


def _cuda_environment() -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": sys.version, "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "thread_environment": {key: os.environ.get(key) for key in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "CUDA_VISIBLE_DEVICES", "TORCH_NCCL_BLOCKING_WAIT",
            "NCCL_LAUNCH_ORDER_IMPLICIT", "NCCL_ALGO", "NCCL_PROTO", "NCCL_P2P_DISABLE")},
    }
    try:
        import torch
        result.update({"torch": torch.__version__, "torch_git_version": torch.version.git_version, "cuda": torch.version.cuda,
                       "cuda_available": torch.cuda.is_available(),
                       "visible_device_count": torch.cuda.device_count(),
                       "device_uuids": [str(getattr(torch.cuda.get_device_properties(index), "uuid", ""))
                                        for index in range(torch.cuda.device_count())]
                       if torch.cuda.is_available() else [],
                       "nccl": torch.cuda.nccl.version() if torch.cuda.is_available() else None})
    except Exception as exc:  # Environment inspection must not mask CPU-side preview.
        result["torch_inspection_error"] = f"{type(exc).__name__}: {exc}"
    return result


def _suite_root(manifest_path: Path) -> Path:
    return manifest_path.resolve(strict=True).parent


def _limit_order_table(order: dict[str, Any], max_blocks: int | None) -> dict[str, Any]:
    if max_blocks is None:
        return order
    if (not isinstance(max_blocks, int) or isinstance(max_blocks, bool)
            or max_blocks <= 0 or max_blocks > order["block_count"]):
        raise ValueError("max_blocks must be positive and no greater than the planned block count")
    limited = dict(order)
    limited["blocks"] = list(order["blocks"][:max_blocks])
    limited["block_count"] = len(limited["blocks"])
    limited["run_count"] = sum(len(block["arms"]) for block in limited["blocks"])
    positions = {arm: [0] * len(order["arms"]) for arm in order["arms"]}
    for block in limited["blocks"]:
        for row in block["arms"]:
            positions[row["arm"]][row["arm_position"]] += 1
    limited["arm_position_counts"] = positions
    limited["planned_block_limit"] = max_blocks
    limited["full_block_count_before_limit"] = order["block_count"]
    verify_order_table(limited)
    return limited


def _verify_bare_qualification(suite: dict[str, Any]) -> dict[str, Any]:
    from examples.jobpacer.runtime.dag_comm_adapters import BARE_CONTRACT_VERSION
    qualification = suite.get("bare_qualification", {})
    evidence = qualification.get("evidence") or {}
    if (qualification.get("arm") != BARE_ARM or qualification.get("status") != "qualified"
            or evidence.get("contract_version") != BARE_CONTRACT_VERSION):
        raise ValueError("bare-ordered requires current backend qualification")
    path = Path(evidence.get("evidence_path", ""))
    path = path if path.is_absolute() else ROOT / path
    if not path.is_file() or _sha(path.read_bytes()) != evidence.get("evidence_sha256"):
        raise ValueError("bare qualification evidence missing or changed")
    audit = json.loads(path.read_text())
    if (audit.get("passed") is not True or audit.get("arm") != BARE_ARM
            or audit.get("source_snapshot_sha256") != source_snapshot()["digest"]):
        raise ValueError("bare qualification source or arm differs from current revision")
    expected_profile_sha = _sha(_canonical({key: value["sha256"] for key, value in sorted(suite["profiles"].items())}))
    if audit.get("profile_sha256") != expected_profile_sha:
        raise ValueError("bare qualification profiles differ from suite")
    return qualification


def _verify_readiness(suite: dict[str, Any]) -> None:
    from examples.jobpacer.experiments.seven_arm.gates import verify_readiness
    verify_readiness(suite, source_snapshot()["digest"])


def create_batch(suite_manifest_path: Path, output_dir: Path, *, order_seed: int,
                 repeats: int, arms: tuple[str, ...], timeout_s: float,
                 max_blocks: int | None = None) -> dict[str, Any]:
    suite_root = _suite_root(suite_manifest_path)
    suite_manifest = json.loads(suite_manifest_path.read_text())
    if not str(suite_manifest.get("status", "")).startswith("frozen-inputs"):
        raise ValueError("suite inputs and both calibrated profiles must be frozen before batch planning")
    _verify_frozen_inputs(suite_root, suite_manifest)
    if suite_manifest.get("contract_version") != CONTRACT_VERSION:
        raise ValueError("legacy suite cannot be used for the v2 seven-arm contract")
    qualification = _verify_bare_qualification(suite_manifest) if BARE_ARM in arms else {}
    if suite_manifest.get("scope") == "formal":
        _verify_readiness(suite_manifest)
    order = make_order_table(suite_manifest, order_seed=order_seed, repeats=repeats, arms=arms)
    if max_blocks is not None:
        if (suite_manifest.get("scope") != "pilot"
                or set(arms) not in (set(MECHANISM_PILOT_ARMS), set(ARMS))):
            raise ValueError("--max-blocks is reserved for pilot smoke or seven-arm rehearsal")
        order = _limit_order_table(order, max_blocks)
    calibration = suite_manifest.get("profiles", {}).get("compute", {}).get("calibration", {})
    settings = calibration.get("settings", {})
    comm_settings = suite_manifest.get("profiles", {}).get("communication", {}).get("settings", {})
    if settings.get("warmup", 0) < 5 or settings.get("iterations", 0) < 30:
        raise ValueError("compute calibration requires at least warmup=5 and iterations=30")
    if comm_settings.get("warmup", 0) < 5 or comm_settings.get("iterations", 0) < 30:
        raise ValueError("communication calibration requires at least warmup=5 and iterations=30")
    if timeout_s <= 0:
        raise ValueError("replay timeout must be positive")
    environment = _cuda_environment()
    if environment.get("visible_device_count") != 2:
        raise ValueError("batch planning requires exactly two currently visible GPUs")
    compute_profile = json.loads((suite_root / suite_manifest["profiles"]["compute"]["path"]).read_text())
    profiled_uuids = sorted(row["device_uuid"] for row in compute_profile.get("devices", []))
    if profiled_uuids != sorted(environment.get("device_uuids", [])):
        raise ValueError("batch planning GPU UUIDs differ from the frozen compute profile")
    verify_order_table(order)
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"batch path already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    for directory in ("inputs", "source", "raw", "logs", "tables", "figures"):
        (output_dir / directory).mkdir()
    for relative in ("templates", "inputs", "profiles"):
        source = suite_root / relative
        if source.exists():
            shutil.copytree(source, output_dir / "inputs" / relative)
    for filename in ("suite-inputs.json", "suite-manifest.json"):
        shutil.copyfile(suite_root / filename, output_dir / "inputs" / filename)
    _verify_frozen_inputs(output_dir / "inputs", suite_manifest)
    _write_json(output_dir / "order.json", order)
    snapshot = source_snapshot(output_dir / "source" / "repository")
    manifest = {
        "schema": "jobpacer-gpu-seven-arm-batch", "schema_version": 1,
        "batch_id": output_dir.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": ("formal" if suite_manifest.get("scope") == "formal" and set(arms) == set(ARMS)
                  else "pilot-seven-arm-rehearsal" if suite_manifest.get("scope") == "pilot"
                  and set(arms) == set(ARMS) else
                  "pilot-six-arm" if suite_manifest.get("scope") == "pilot"
                  and set(arms) == set(MECHANISM_PILOT_ARMS)
                  and order["block_count"] == 54 and repeats == 3 else
                  "pilot-block-smoke" if suite_manifest.get("scope") == "pilot"
                  and set(arms) == set(MECHANISM_PILOT_ARMS) else "incomplete-pilot-diagnostic"),
        "scope": suite_manifest.get("scope"),
        "planned_blocks": order["block_count"], "planned_runs": order["run_count"],
        "planned_block_limit": max_blocks,
        "formal_complete_scope": "six scenarios x five workload seeds x five repeats x seven arms",
        "arms": list(arms), "order_seed": order_seed, "repeats_per_sample": repeats,
        "suite_manifest_sha256": _sha(suite_manifest_path.read_bytes()),
        "order_sha256": _sha((output_dir / "order.json").read_bytes()),
        "source_snapshot": snapshot,
        "environment_at_plan": environment,
        "bare_qualification": qualification,
        "contract_version": CONTRACT_VERSION,
        "retries": {"scope": "whole paired block", "max_environment_retries": 1,
                    "retry_markers": list(ENV_RETRY_MARKERS), "max_block_attempts": 2,
                    "inner_startup_retries": 0},
        "execution_contract": suite_manifest.get("execution_contract"),
        "replay_settings": {"timeout_s": timeout_s, "setup_timeout_s": max(20.0, timeout_s),
                            "warmup_iterations": 5, "poll_interval_s": 0.001,
                            "dag_poll_interval_s": 0.001, "wait_budget_s": 0.02,
                            "observation_mode": "minimal" if suite_manifest.get("scope") == "formal" else "full",
                            "matmul_precision": "highest",
                            "profile_strict": True},
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def _read_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    result = []
    seen: set[tuple[str, int]] = set()
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        key = (row.get("run_id"), row.get("attempt"))
        if key in seen:
            raise ValueError(f"duplicate ledger run attempt at line {line_number}: {key}")
        seen.add(key)
        result.append(row)
    return result


def _reserve_run_attempt(run: dict[str, Any], batch_dir: Path, block_attempt: int) -> dict[str, Any]:
    """Atomically claim fresh raw/log paths before starting a child process."""
    reservations_dir = batch_dir / "logs" / "reservations"
    reservations_dir.mkdir(parents=True, exist_ok=True)
    ledger = _read_ledger(batch_dir / "runs.jsonl")
    attempt = max((int(row["attempt"]) for row in ledger if row.get("run_id") == run["run_id"]),
                 default=0) + 1
    while True:
        stem = f"{run['run_id']}.block-{block_attempt}.attempt-{attempt}"
        raw_path = batch_dir / "raw" / f"{stem}.json"
        stdout_path = batch_dir / "logs" / f"{stem}.stdout"
        stderr_path = batch_dir / "logs" / f"{stem}.stderr"
        reservation_path = reservations_dir / f"{run['run_id']}.attempt-{attempt}.json"
        if raw_path.exists() or stdout_path.exists() or stderr_path.exists():
            attempt += 1
            continue
        reservation = {
            "run_id": run["run_id"], "block_id": run["block_id"],
            "block_attempt": block_attempt, "attempt": attempt,
            "reserved_at": datetime.now(timezone.utc).isoformat(),
            "raw_path": raw_path.relative_to(batch_dir).as_posix(),
            "stdout_path": stdout_path.relative_to(batch_dir).as_posix(),
            "stderr_path": stderr_path.relative_to(batch_dir).as_posix(),
            "status": "reserved-before-launch",
            "launch_state": "reserved-before-launch",
            "state_history": [{
                "state": "reserved-before-launch",
                "at": datetime.now(timezone.utc).isoformat(),
            }],
        }
        try:
            descriptor = os.open(reservation_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            attempt += 1
            continue
        data = _canonical(reservation)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        return {
            **reservation, "reservation_path": reservation_path.relative_to(batch_dir).as_posix(),
            "reservation_sha256": _sha(data),
        }


def _persist_reservation_state(batch_dir: Path, reservation: dict[str, Any], state: str,
                               **updates: Any) -> dict[str, Any]:
    """Atomically persist a launch lifecycle transition before relying on it."""
    relative = reservation.get("reservation_path")
    if not relative:
        raise ValueError("reservation path is required before updating launch state")
    path = batch_dir / relative
    current = json.loads(path.read_text())
    current.update(updates)
    current["launch_state"] = state
    current["status"] = state
    current.setdefault("state_history", []).append({
        "state": state, "at": datetime.now(timezone.utc).isoformat(),
        **{key: value for key, value in updates.items() if key != "command"},
    })
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp",
                                                  dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            data = _canonical(current)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()
    reservation.update(current)
    reservation["reservation_sha256"] = _sha(_canonical(current))
    return reservation


def _process_start_ticks(pid: int) -> int | None:
    """Read Linux /proc start-time ticks to distinguish a reused PID."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    closing = stat.rfind(")")
    if closing < 0:
        return None
    fields_after_command = stat[closing + 2:].split()
    # /proc/<pid>/stat field 22 is starttime; the suffix starts at field 3.
    try:
        return int(fields_after_command[19])
    except (IndexError, ValueError):
        return None


def _process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _reservation_has_live_child(reservation: dict[str, Any]) -> bool:
    """Return true for a live/ambiguous child group; only absence is safe to recover."""
    state = reservation.get("launch_state", reservation.get("status"))
    if state == "spawn-intent":
        raise ValueError(
            "resume rejected: child startup is ambiguous (spawn intent has no durable process identity); "
            "inspect the reservation and GPU before resuming"
        )
    if state in {"reserved-before-launch", "launch-failed-known-no-child"}:
        return False
    if state not in {"child-started", "child-exited"}:
        raise ValueError(f"resume rejected: unknown child launch state {state!r}")
    try:
        pid = int(reservation["pid"])
        pgid = int(reservation["pgid"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("resume rejected: child process identity is incomplete") from exc
    expected_start = reservation.get("process_start_ticks")
    observed_start = _process_start_ticks(pid)
    group_exists = _process_group_exists(pgid)
    if not group_exists:
        return False
    # A living group with this ID may contain rank children after its leader exits.
    # If the leader identity changed, do not guess whether the group ID was reused.
    if observed_start is not None and expected_start is not None and observed_start != expected_start:
        raise ValueError(
            "resume rejected: process-group identity is ambiguous after PID reuse; "
            "inspect the reservation and GPU before resuming"
        )
    return True


def _start_replay_process(command: list[str], *, cwd: Path, env: dict[str, str],
                          batch_dir: Path, reservation: dict[str, Any]) -> subprocess.Popen:
    """Start one isolated process group and durably record its identity."""
    reservation = _persist_reservation_state(
        batch_dir, reservation, "spawn-intent", command=command,
        command_sha256=_sha(_canonical(command)),
    )
    try:
        process = subprocess.Popen(
            command, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, start_new_session=True,
        )
    except OSError as exc:
        _persist_reservation_state(
            batch_dir, reservation, "launch-failed-known-no-child",
            launch_error=f"{type(exc).__name__}: {exc}",
        )
        raise
    try:
        pgid = os.getpgid(process.pid)
    except ProcessLookupError:
        # A very short child may exit before getpgid; the new session's group ID
        # is still its PID, and recovery verifies that the group no longer exists.
        pgid = process.pid
    _persist_reservation_state(
        batch_dir, reservation, "child-started", pid=process.pid, pgid=pgid,
        process_start_ticks=_process_start_ticks(process.pid),
        child_started_at=datetime.now(timezone.utc).isoformat(),
    )
    return process


def _terminate_owned_process_group(process: subprocess.Popen, reservation: dict[str, Any],
                                   grace_s: float = 1.0) -> bool:
    """Boundedly stop only a child group whose leader identity still matches."""
    pid = int(reservation["pid"])
    pgid = int(reservation["pgid"])
    expected_start = reservation.get("process_start_ticks")
    observed_start = _process_start_ticks(pid)
    if observed_start is None or (expected_start is not None and observed_start != expected_start):
        return not _process_group_exists(pgid)
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    try:
        process.communicate(timeout=grace_s)
    except subprocess.TimeoutExpired:
        observed_start = _process_start_ticks(pid)
        if observed_start is not None and (expected_start is None or observed_start == expected_start):
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            process.communicate(timeout=grace_s)
        except subprocess.TimeoutExpired:
            pass
    deadline = time.monotonic() + grace_s
    while _process_group_exists(pgid) and time.monotonic() < deadline:
        time.sleep(min(0.02, max(0.0, deadline - time.monotonic())))
    return not _process_group_exists(pgid)


def _check_resume(batch_dir: Path, manifest: dict[str, Any], order: dict[str, Any]) -> list[dict[str, Any]]:
    saved_manifest = json.loads((batch_dir / "manifest.json").read_text())
    if _canonical(saved_manifest) != _canonical(manifest):
        raise ValueError("resume rejected: batch manifest, source, or execution environment changed")
    if _sha((batch_dir / "order.json").read_bytes()) != manifest["order_sha256"]:
        raise ValueError("resume rejected: frozen order table changed")
    suite_path = batch_dir / "inputs/suite-manifest.json"
    if _sha(suite_path.read_bytes()) != manifest["suite_manifest_sha256"]:
        raise ValueError("resume rejected: frozen suite manifest changed")
    _verify_frozen_inputs(batch_dir / "inputs", json.loads(suite_path.read_text()))
    archived_root = batch_dir / "source/repository"
    for item in manifest["source_snapshot"]["files"]:
        archived = archived_root / item["path"]
        if not archived.is_file() or _sha(archived.read_bytes()) != item["sha256"]:
            raise ValueError(f"resume rejected: archived source bytes changed: {archived}")
    ledger = _read_ledger(batch_dir / "runs.jsonl")
    planned = {run["run_id"] for block in order["blocks"] for run in block["arms"]}
    if any(row.get("run_id") not in planned for row in ledger):
        raise ValueError("resume rejected: ledger contains an unplanned run ID")
    current_snapshot = source_snapshot()
    if current_snapshot["digest"] != saved_manifest["source_snapshot"]["digest"]:
        raise ValueError("resume rejected: current source bytes differ from the archived revision")
    if _canonical(_cuda_environment()) != _canonical(saved_manifest["environment_at_plan"]):
        raise ValueError("resume rejected: GPU/software/visibility environment changed")
    for row in ledger:
        for path_key, hash_key in (("raw_path", "raw_sha256"), ("stdout_path", "stdout_sha256"),
                                   ("stderr_path", "stderr_sha256")):
            file_path = batch_dir / row[path_key]
            expected = row.get(hash_key)
            if expected is not None and (not file_path.is_file() or _sha(file_path.read_bytes()) != expected):
                raise ValueError(f"resume rejected: attempted artifact changed: {file_path}")
        reservation_path = row.get("reservation_path")
        if reservation_path:
            file_path = batch_dir / reservation_path
            if not file_path.is_file() or _sha(file_path.read_bytes()) != row.get("reservation_sha256"):
                raise ValueError(f"resume rejected: attempt reservation changed: {file_path}")
            reservation = json.loads(file_path.read_text())
            if _reservation_has_live_child(reservation):
                raise ValueError(
                    "resume rejected: a recorded replay or rank child is still running; "
                    f"refusing to reuse GPUs for {file_path}"
                )
    ledger = _recover_orphaned_attempts(batch_dir, manifest, order, ledger)
    return ledger


def _recover_orphaned_attempts(batch_dir: Path, manifest: dict[str, Any], order: dict[str, Any],
                               ledger: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Record abandoned reservations as failed attempts so resume can restart whole blocks."""
    recorded = {row.get("reservation_path") for row in ledger if row.get("reservation_path")}
    planned = {row["run_id"]: row for block in order["blocks"] for row in block["arms"]}
    ledger_path = batch_dir / "runs.jsonl"
    orphans = sorted(path for path in (batch_dir / "logs" / "reservations").glob("*.json")
                     if path.relative_to(batch_dir).as_posix() not in recorded)
    if not orphans:
        return ledger
    orphan_records = []
    for reservation_path in orphans:
        reservation = json.loads(reservation_path.read_text())
        if _reservation_has_live_child(reservation):
            raise ValueError(
                "resume rejected: replay or rank child is still running; "
                f"refusing to reuse GPUs for {reservation_path}"
            )
        orphan_records.append((reservation_path, reservation))
    expected_contract = _sha(_canonical({
        "execution_contract": manifest["execution_contract"],
        "replay_settings": manifest["replay_settings"],
    }))
    with ledger_path.open("a") as stream:
        for reservation_path, reservation in orphan_records:
            run_id = reservation.get("run_id")
            run = planned.get(run_id)
            if run is None or reservation.get("block_id") != run.get("block_id"):
                raise ValueError(f"orphan reservation does not match a planned run: {reservation_path}")
            relative_reservation = reservation_path.relative_to(batch_dir).as_posix()
            artifact_paths = {}
            for key in ("raw_path", "stdout_path", "stderr_path"):
                relative = Path(str(reservation.get(key, "")))
                if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                    raise ValueError(f"unsafe artifact path in reservation: {reservation_path}")
                artifact_paths[key] = batch_dir / relative
            raw_path = artifact_paths["raw_path"]
            result = None
            if raw_path.is_file():
                try:
                    result = json.loads(raw_path.read_text())
                except json.JSONDecodeError:
                    result = None
            record = {
                **{key: run[key] for key in (
                    "run_id", "block_id", "arm", "scenario", "workload_seed", "repeat", "epoch",
                    "input_sha256", "execution_sample_hash", "estimate_view_hash",
                    "compute_profile_sha256", "comm_profile_sha256")},
                "block_attempt": int(reservation["block_attempt"]),
                "attempt": int(reservation["attempt"]),
                "command": None, "started_at": reservation.get("reserved_at"),
                "ended_at": datetime.now(timezone.utc).isoformat(), "wall_time_s": None,
                "returncode": None, "failure_class": "parent_interrupted_before_ledger",
                "validation_status": (result or {}).get("validation", {}).get("status", "missing"),
                "source_snapshot_sha256": manifest["source_snapshot"]["digest"],
                "contract_sha256": expected_contract,
                "raw_path": reservation["raw_path"],
                "stdout_path": reservation["stdout_path"],
                "stderr_path": reservation["stderr_path"],
                "reservation_path": relative_reservation,
                "reservation_sha256": _sha(reservation_path.read_bytes()),
                "raw_sha256": _sha(raw_path.read_bytes()) if raw_path.is_file() else None,
                "stdout_sha256": (_sha(artifact_paths["stdout_path"].read_bytes())
                                  if artifact_paths["stdout_path"].is_file() else None),
                "stderr_sha256": (_sha(artifact_paths["stderr_path"].read_bytes())
                                  if artifact_paths["stderr_path"].is_file() else None),
                "raw_bytes": raw_path.stat().st_size if raw_path.is_file() else 0,
                "child_identity": {
                    key: reservation.get(key) for key in (
                        "pid", "pgid", "process_start_ticks", "launch_state",
                        "child_started_at", "child_returncode")
                },
                "error": ("parent stopped before writing the completion ledger; the full paired block "
                          "will restart after the recorded child group was confirmed absent"),
            }
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            ledger.append(record)
            recorded.add(relative_reservation)
    return ledger


def _verify_frozen_inputs(payload_root: Path, suite: dict[str, Any]) -> None:
    for sample in suite.get("samples", []):
        path = payload_root / sample["path"]
        if not path.is_file() or _sha(path.read_bytes()) != sample.get("input_sha256"):
            raise ValueError(f"frozen workload input missing or changed: {path}")
        for field, hash_field in (("fifo_order_path", "fifo_order_sha256"),
                                  ("ltf_order_path", "ltf_order_sha256")):
            relative = sample.get(field)
            expected = sample.get(hash_field)
            if relative is None:
                continue
            order_path = payload_root / relative
            if expected is None or not order_path.is_file() or _sha(order_path.read_bytes()) != expected:
                raise ValueError(f"frozen static order missing or changed: {order_path}")
    for key in ("compute", "communication"):
        profile = suite.get("profiles", {}).get(key)
        if not profile:
            raise ValueError(f"frozen {key} profile is missing")
        path = payload_root / profile["path"]
        if not path.is_file() or _sha(path.read_bytes()) != profile.get("sha256"):
            raise ValueError(f"frozen {key} profile missing or changed: {path}")


def _accepted_blocks(ledger: list[dict[str, Any]], arms: tuple[str, ...]) -> dict[str, int]:
    by_attempt: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in ledger:
        by_attempt.setdefault((row["block_id"], int(row["block_attempt"])), []).append(row)
    accepted: dict[str, int] = {}
    for (block_id, block_attempt), rows in by_attempt.items():
        if (len(rows) == len(arms) and {row["arm"] for row in rows} == set(arms)
                and all(row.get("validation_status") == "ok"
                        and row.get("failure_class") == "none"
                        and row.get("returncode") == 0 for row in rows)):
            previous = accepted.get(block_id)
            if previous is not None and previous != block_attempt:
                raise ValueError(f"multiple accepted attempts found for block {block_id}")
            accepted[block_id] = block_attempt
    return accepted


def _classify_failure(output: str, returncode: int | None, timed_out: bool) -> str:
    if timed_out:
        return "replay_timeout"
    if returncode == 0:
        return "none"
    if any(marker.lower() in output.lower() for marker in ENV_RETRY_MARKERS):
        return "environment_port_conflict"
    return "application_or_correctness_failure"


def _verified_pre_task_port_conflict(
    result: dict[str, Any] | None, *, returncode: int | None, timed_out: bool,
    process_group_exited: bool,
) -> bool:
    """Trust structured startup evidence, not PyTorch's version-specific error wording."""
    if timed_out or returncode in (0, None) or not process_group_exited or result is None:
        return False
    if result.get("ranks"):
        return False
    attempts = result.get("config", {}).get("rendezvous_startup_attempts")
    if not isinstance(attempts, list) or not attempts:
        return False
    return all(
        any(
            "EADDRINUSE" in str(error)
            and any(marker in str(error).lower() for marker in (
                "distnetworkerror", "server socket", "tcpstore",
            ))
            for error in attempt.get("errors", [])
        )
        for attempt in attempts
    )


def _run_child(command: list[str], *, cwd: Path, env: dict[str, str], timeout_s: float,
               batch_dir: Path, reservation: dict[str, Any]) -> tuple[int | None, str, str, bool, bool]:
    process = _start_replay_process(command, cwd=cwd, env=env, batch_dir=batch_dir,
                                    reservation=reservation)
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout_s)
        returncode = process.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        group_exited = _terminate_owned_process_group(process, reservation)
        if not group_exited:
            stderr += "\nreplay timed out; owned process group could not be confirmed exited"
            return None, stdout, stderr, True, False
        # communicate() reaps the process after termination when the leader identity
        # is still ours. Its captured output includes data written during shutdown.
        try:
            tail_out, tail_err = process.communicate(timeout=1.0)
        except subprocess.TimeoutExpired:
            tail_out, tail_err = "", ""
        stdout += tail_out or ""
        stderr += tail_err or ""
        stderr += f"\nparent timeout after {timeout_s}s"
        _persist_reservation_state(
            batch_dir, reservation, "child-exited", child_returncode=process.returncode,
            child_exited_at=datetime.now(timezone.utc).isoformat(), timed_out=True,
        )
        return process.returncode, stdout, stderr, True, True
    group_exited = not _process_group_exists(int(reservation["pgid"]))
    if group_exited:
        _persist_reservation_state(
            batch_dir, reservation, "child-exited", child_returncode=returncode,
            child_exited_at=datetime.now(timezone.utc).isoformat(),
        )
    else:
        stderr += ("\nreplay parent exited while its process group still has members; "
                   "refusing to start another GPU replay")
        returncode = returncode if returncode not in (None, 0) else 1
    return returncode, stdout, stderr, timed_out, group_exited


def _run_command(run: dict[str, Any], suite: dict[str, Any], batch_dir: Path,
                 timeout: float, warmup: int, *, output_path: Path) -> list[str]:
    if run["arm"] not in ARM_CONFIG:
        raise ValueError(f"unsupported seven-arm implementation {run['arm']}")
    engine, policy, order_key = ARM_CONFIG[run["arm"]]
    sample_path = batch_dir / "inputs" / run["sample_path"]
    args = [sys.executable, "-m", "examples.jobpacer.scripts.run_phase3",
            "--policy", policy, "--comm-engine", engine, "--startup-attempts", "1",
            "--dag", str(sample_path.resolve()), "--backend", "nccl", "--world-size", "2",
            "--epoch", str(run["epoch"]), "--timeout", str(timeout),
            "--setup-timeout", str(max(20.0, timeout)),
            "--warmup-iterations", str(warmup), "--observation-mode", json.loads((batch_dir / "manifest.json").read_text())["replay_settings"].get("observation_mode", "full"),
            "--poll-interval", "0.001", "--dag-poll-interval", "0.001",
            "--wait-budget-s", "0.02", "--matmul-precision", "highest",
            "--output", str(output_path.resolve())]
    profile = suite["profiles"]
    args.extend(("--comm-profile", str((batch_dir / "inputs" / profile["communication"]["path"]).resolve()),
                 "--compute-profile", str((batch_dir / "inputs" / profile["compute"]["path"]).resolve())))
    if order_key:
        order_path = batch_dir / "inputs" / run[order_key]
        args.extend(("--static-order", str(order_path.resolve())))
    return args


def _execute_one(run: dict[str, Any], suite: dict[str, Any], batch_dir: Path,
                 timeout: float, warmup: int, block_attempt: int,
                 ledger_file) -> dict[str, Any]:
    reservation = _reserve_run_attempt(run, batch_dir, block_attempt)
    attempt = reservation["attempt"]
    raw_path = batch_dir / reservation["raw_path"]
    stdout_path = batch_dir / reservation["stdout_path"]
    stderr_path = batch_dir / reservation["stderr_path"]
    command = _run_command(run, suite, batch_dir, timeout, warmup, output_path=raw_path)
    started = time.time()
    start_utc = datetime.fromtimestamp(started, timezone.utc).isoformat()
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(ROOT), env.get("PYTHONPATH", "")))
    timed_out = False
    process_group_exited = False
    try:
        returncode, stdout, stderr, timed_out, process_group_exited = _run_child(
            command, cwd=ROOT, env=env, timeout_s=timeout + max(20.0, timeout),
            batch_dir=batch_dir, reservation=reservation,
        )
    except OSError as exc:
        if reservation.get("launch_state") != "launch-failed-known-no-child":
            # A spawn-intent without a durable process identity is intentionally
            # left ambiguous; resume will refuse to reuse the GPUs automatically.
            raise
        returncode = 127
        stdout = ""
        stderr = f"{type(exc).__name__}: {exc}"
        process_group_exited = True
    if not process_group_exited and reservation.get("launch_state") != "launch-failed-known-no-child":
        stderr += "\nchild process group was not confirmed exited"
    stdout_path.write_text(stdout)
    stderr_path.write_text(stderr)
    result = None
    if raw_path.is_file():
        try:
            result = json.loads(raw_path.read_text())
        except json.JSONDecodeError:
            result = None
    validation = (result or {}).get("validation", {}).get("status")
    if returncode == 0 and validation != "ok":
        returncode = 1
    error_text = f"{stdout}\n{stderr}"
    classification = _classify_failure(error_text, returncode, timed_out)
    verified_port_conflict = _verified_pre_task_port_conflict(
        result, returncode=returncode, timed_out=timed_out,
        process_group_exited=process_group_exited,
    )
    if verified_port_conflict:
        classification = "environment_port_conflict"
    elif classification == "environment_port_conflict":
        classification = "application_or_cleanup_failure"
    sample_meta = run
    record = {
        "run_id": run["run_id"], "block_id": run["block_id"],
        "block_attempt": block_attempt, "arm": run["arm"],
        "scenario": run["scenario"], "workload_seed": run["workload_seed"],
        "repeat": run["repeat"], "epoch": run["epoch"], "attempt": attempt,
        "command": command, "started_at": start_utc,
        "ended_at": datetime.now(timezone.utc).isoformat(), "wall_time_s": time.time() - started,
        "returncode": returncode, "failure_class": classification,
        "child_process_group_exited": process_group_exited,
        "child_identity": {key: reservation.get(key) for key in (
            "pid", "pgid", "process_start_ticks", "launch_state", "child_returncode")},
        "validation_status": validation if validation is not None else "missing",
        "input_sha256": sample_meta["input_sha256"],
        "execution_sample_hash": sample_meta["execution_sample_hash"],
        "estimate_view_hash": sample_meta["estimate_view_hash"],
        "compute_profile_sha256": sample_meta["compute_profile_sha256"],
        "comm_profile_sha256": sample_meta["comm_profile_sha256"],
        "source_snapshot_sha256": json.loads((batch_dir / "manifest.json").read_text())["source_snapshot"]["digest"],
        "contract_sha256": _sha(_canonical({
            "execution_contract": json.loads((batch_dir / "manifest.json").read_text())["execution_contract"],
            "replay_settings": json.loads((batch_dir / "manifest.json").read_text())["replay_settings"],
        })),
        "raw_path": raw_path.relative_to(batch_dir).as_posix(),
        "stdout_path": stdout_path.relative_to(batch_dir).as_posix(),
        "stderr_path": stderr_path.relative_to(batch_dir).as_posix(),
        "reservation_path": reservation["reservation_path"],
        "reservation_sha256": _sha((batch_dir / reservation["reservation_path"]).read_bytes()),
        "raw_sha256": _sha(raw_path.read_bytes()) if raw_path.is_file() else None,
        "stdout_sha256": _sha(stdout_path.read_bytes()), "stderr_sha256": _sha(stderr_path.read_bytes()),
        "raw_bytes": raw_path.stat().st_size if raw_path.is_file() else 0,
        "error": error_text[-4000:] if classification != "none" else None,
    }
    ledger_file.write(json.dumps(record, sort_keys=True) + "\n")
    ledger_file.flush()
    os.fsync(ledger_file.fileno())
    return record


def run_batch(batch_dir: Path, *, timeout: float, warmup: int, allow_pilot: bool,
              resume: bool, allow_formal_matrix: bool = False) -> int:
    batch_dir = batch_dir.resolve(strict=True)
    manifest = json.loads((batch_dir / "manifest.json").read_text())
    order = json.loads((batch_dir / "order.json").read_text())
    verify_order_table(order)
    suite = json.loads((batch_dir / "inputs/suite-manifest.json").read_text())
    _verify_frozen_inputs(batch_dir / "inputs", suite)
    if manifest["arms"] != order["arms"] or manifest["suite_manifest_sha256"] != _sha((batch_dir / "inputs/suite-manifest.json").read_bytes()):
        raise ValueError("batch manifest differs from frozen order or suite")
    if manifest["source_snapshot"]["digest"] != source_snapshot()["digest"]:
        raise ValueError("source changed since batch plan; create a new revision")
    stage = manifest.get("stage")
    arm_set = set(order["arms"])
    if stage in {"pilot-six-arm", "pilot-block-smoke"}:
        if not allow_pilot or arm_set != set(MECHANISM_PILOT_ARMS):
            raise ValueError("six-arm mechanism pilot requires --allow-pilot")
    elif stage == "pilot-seven-arm-rehearsal":
        if not allow_pilot or arm_set != set(ARMS):
            raise ValueError("seven-arm rehearsal requires --allow-pilot and all frozen arms")
    elif stage == "formal":
        if not allow_formal_matrix or arm_set != set(ARMS):
            raise ValueError("formal matrix is locked; it requires --allow-formal-matrix")
    else:
        raise ValueError(f"batch stage {stage!r} is not executable")
    if BARE_ARM in arm_set:
        _verify_bare_qualification(suite)
    if stage == "formal":
        _verify_readiness(suite)
    if stage != "formal" and (len(order["blocks"]) > 54 or manifest.get("repeats_per_sample", 0) > 3):
        raise ValueError("pilot batches are limited to three pilot seeds, three repeats, and 54 blocks")
    current_environment = _cuda_environment()
    if _canonical(current_environment) != _canonical(manifest["environment_at_plan"]):
        raise ValueError("GPU/software/visibility changed after the suite order was frozen")
    if current_environment.get("visible_device_count") != 2:
        raise ValueError("GPU suite requires exactly two currently visible GPUs")
    if timeout != manifest["replay_settings"]["timeout_s"]:
        raise ValueError("timeout changed since the batch order was frozen")
    if warmup != manifest["replay_settings"]["warmup_iterations"]:
        raise ValueError("warmup changed since the batch order was frozen")
    ledger = _check_resume(batch_dir, manifest, order)
    if ledger and not resume:
        raise ValueError("batch already has attempts; pass --resume to continue this exact frozen batch")
    accepted = _accepted_blocks(ledger, tuple(order["arms"]))
    latest_block_attempt: dict[str, int] = {}
    for row in ledger:
        latest_block_attempt[row["block_id"]] = max(latest_block_attempt.get(row["block_id"], 0),
                                                      int(row["block_attempt"]))
    ledger_path = batch_dir / "runs.jsonl"
    ledger_path.touch(exist_ok=True)
    with ledger_path.open("a") as ledger_file:
        for block in order["blocks"]:
            block_id = block["block_id"]
            if block_id in accepted:
                continue
            previous_attempts = latest_block_attempt.get(block_id, 0)
            previous_rows = [row for row in ledger if row.get("block_id") == block_id
                             and int(row.get("block_attempt", 0)) == previous_attempts]
            if previous_rows:
                classes = {row.get("failure_class") for row in previous_rows}
                if classes - {"none", "environment_port_conflict", "parent_interrupted_before_ledger"}:
                    raise RuntimeError(f"block {block_id} has a non-retryable prior failure")
                if "environment_port_conflict" in classes and previous_attempts >= 2:
                    raise RuntimeError(f"block {block_id} exhausted its environment retry")
                if "environment_port_conflict" not in classes and not resume:
                    raise RuntimeError(f"incomplete block {block_id} requires explicit --resume")
            next_block_attempt = previous_attempts + 1
            if next_block_attempt > 2:
                raise RuntimeError(f"block {block_id} already exhausted its one whole-block retry")
            results = []
            for run in block["arms"]:
                record = _execute_one(run, suite, batch_dir, timeout, warmup,
                                      next_block_attempt, ledger_file)
                results.append(record)
                if record["failure_class"] not in {"none", "environment_port_conflict"}:
                    raise RuntimeError(
                        f"stopping after {record['failure_class']} in {block_id}/{run['arm']}; "
                        "preserve the block and start a new source revision after correction"
                    )
            if all(row["failure_class"] == "none" for row in results):
                accepted[block_id] = next_block_attempt
                continue
            if any(row["failure_class"] not in {"none", "environment_port_conflict"}
                   for row in results):
                raise RuntimeError(f"block {block_id} has a non-retryable failure")
            if next_block_attempt == 2:
                raise RuntimeError(f"block {block_id} failed its single whole-block environment retry")
            latest_block_attempt[block_id] = next_block_attempt
            retry_rows = []
            for run in block["arms"]:
                record = _execute_one(run, suite, batch_dir, timeout, warmup,
                                      next_block_attempt + 1, ledger_file)
                retry_rows.append(record)
                if record["failure_class"] not in {"none", "environment_port_conflict"}:
                    raise RuntimeError(f"stopping after non-retryable failure in retry of {block_id}")
            if any(row["failure_class"] != "none" for row in retry_rows):
                raise RuntimeError(f"block {block_id} failed its single whole-block environment retry")
            accepted[block_id] = next_block_attempt + 1
            latest_block_attempt[block_id] = next_block_attempt + 1
    return 0


def _bootstrap_median(values: list[float], *, samples: int, seed: int) -> tuple[float, float] | None:
    if not values or not samples:
        return None
    import random
    rng = random.Random(seed)
    boot = [statistics.median(values[rng.randrange(len(values))] for _ in values)
            for _ in range(samples)]
    boot.sort()
    return boot[int(0.025 * (len(boot) - 1))], boot[int(0.975 * (len(boot) - 1))]


def _rank0_decisions(result: dict[str, Any]) -> list[dict[str, Any]]:
    rank = next((item for item in result.get("ranks", []) if item.get("rank") == 0), {})
    return rank.get("decision_records", [])


def _static_head_hol(order: list[str], decisions: list[dict[str, Any]]) -> bool:
    snapshots = [row for row in decisions if row.get("kind") == "policy_snapshot"]
    dispatches = sorted((row for row in decisions
                         if row.get("kind") == "decision" and row.get("decision") == "dispatch"),
                        key=lambda row: row.get("now", 0.0))
    for idle in decisions:
        if idle.get("kind") != "idle_interval" or idle.get("reason") != "STATIC_HEAD_BLOCKED":
            continue
        if float(idle.get("duration", 0.0)) <= 0:
            continue
        prior = [row for row in dispatches if row.get("now", 0.0) < idle.get("start", 0.0)]
        cursor = len(prior)
        if cursor >= len(order):
            continue
        head = order[cursor]
        for snapshot in snapshots:
            now = snapshot.get("now", -1.0)
            if idle.get("start", 0.0) <= now <= idle.get("end", 0.0):
                eligible = {item.get("task_id") for item in snapshot.get("eligible", [])}
                if eligible and head not in eligible:
                    return True
    return False


def _dynamic_fifo_bypass(order: list[str], decisions: list[dict[str, Any]]) -> bool:
    dispatched: set[str] = set()
    for row in sorted((item for item in decisions
                       if item.get("kind") == "decision" and item.get("decision") == "dispatch"),
                      key=lambda item: item.get("now", 0.0)):
        head = next((task_id for task_id in order if task_id not in dispatched), None)
        eligible = set(row.get("eligible", []))
        selected = row.get("task_id")
        if head is not None and head not in eligible and selected in eligible and selected != head:
            return True
        if isinstance(selected, str):
            dispatched.add(selected)
    return False


def _ltf_frontier_choice(decisions: list[dict[str, Any]]) -> tuple[bool, bool]:
    snapshots = [row for row in decisions if row.get("kind") == "policy_snapshot"]
    saw_distinct = False
    selected_highest = False
    for decision in decisions:
        if decision.get("kind") != "decision" or decision.get("decision") != "dispatch":
            continue
        eligible = set(decision.get("eligible", []))
        if len(eligible) < 2:
            continue
        now = decision.get("now", 0.0)
        matching = [row for row in snapshots
                    if row.get("now", float("inf")) <= now
                    and {item.get("task_id") for item in row.get("eligible", [])} == eligible]
        if not matching:
            continue
        snapshot = max(matching, key=lambda row: row.get("now", 0.0))
        scores = {item["task_id"]: float(item["estimated_comm_s"])
                  + float(item["remaining_tail_s"])
                  for item in snapshot.get("eligible", [])}
        if len(set(scores.values())) < 2:
            continue
        saw_distinct = True
        highest = max(scores.values())
        if scores.get(decision.get("task_id")) == highest:
            selected_highest = True
    return saw_distinct, selected_highest


def _formal_matrix_checks(manifest: dict[str, Any], suite: dict[str, Any],
                          order: dict[str, Any], *, accepted_blocks: int,
                          accepted_runs: int, validation_errors: list[str]) -> dict[str, bool]:
    expected_blocks = {
        (scenario, seed, repeat)
        for scenario in SCENARIOS for seed in FORMAL_WORKLOAD_SEEDS for repeat in range(5)
    }
    actual_blocks = {
        (row.get("scenario"), row.get("workload_seed"), row.get("repeat"))
        for row in order.get("blocks", [])
    }
    expected_samples = {(scenario, seed) for scenario in SCENARIOS for seed in FORMAL_WORKLOAD_SEEDS}
    actual_samples = {
        (row.get("scenario"), row.get("workload_seed")) for row in suite.get("samples", [])
    }
    arms = list(order.get("arms", []))
    qualification = suite.get("bare_qualification", {})
    evidence = qualification.get("evidence") or {}
    qualified_bare = (qualification.get("arm") == BARE_ARM
                      and qualification.get("status") == "qualified"
                      and evidence.get("contract_version") == "bare-ordered-v2-layered-round-robin")
    exact_original_arms = arms == list(ARMS)
    order_matches_manifest = arms == manifest.get("arms")
    return {
        "formal_stage": manifest.get("stage") == "formal",
        "formal_scope": manifest.get("scope") == "formal" and suite.get("scope") == "formal",
        "six_scenarios": set(row[0] for row in actual_blocks) == set(SCENARIOS),
        "five_formal_seeds": set(row[1] for row in actual_blocks) == set(FORMAL_WORKLOAD_SEEDS),
        "five_repeats": manifest.get("repeats_per_sample") == 5,
        "exact_150_block_grid": len(order.get("blocks", [])) == 150 and actual_blocks == expected_blocks,
        "exact_1050_planned_runs": (len(order.get("arms", [])) == 7
                                     and order.get("run_count") == 1050
                                     and manifest.get("planned_blocks") == 150
                                     and manifest.get("planned_runs") == 1050),
        "exact_30_formal_samples": len(suite.get("samples", [])) == 30 and actual_samples == expected_samples,
        "bare_qualified": qualified_bare,
        "contract_version": suite.get("contract_version") == CONTRACT_VERSION,
        "exact_frozen_arm_ids": order_matches_manifest and exact_original_arms,
        "all_150_blocks_independently_accepted": accepted_blocks == 150,
        "all_1050_runs_independently_accepted": accepted_runs == 1050,
        "all_independent_validation_passed": not validation_errors,
    }


def analyze_batch(batch_dir: Path, *, bootstrap_samples: int = 2000, analysis_seed: int = 20260928) -> dict[str, Any]:
    batch_dir = batch_dir.resolve(strict=True)
    manifest = json.loads((batch_dir / "manifest.json").read_text())
    order = json.loads((batch_dir / "order.json").read_text())
    ledger = _read_ledger(batch_dir / "runs.jsonl")
    validation_errors = []
    suite_for_check = {}
    bad_blocks: set[str] = set()
    try:
        if _sha((batch_dir / "order.json").read_bytes()) != manifest["order_sha256"]:
            raise ValueError("order table hash differs from batch manifest")
        suite_for_check = json.loads((batch_dir / "inputs/suite-manifest.json").read_text())
        if _sha((batch_dir / "inputs/suite-manifest.json").read_bytes()) != manifest["suite_manifest_sha256"]:
            raise ValueError("suite manifest hash differs from batch manifest")
        _verify_frozen_inputs(batch_dir / "inputs", suite_for_check)
        archived_root = batch_dir / "source/repository"
        for source in manifest["source_snapshot"]["files"]:
            path = archived_root / source["path"]
            if not path.is_file() or _sha(path.read_bytes()) != source["sha256"]:
                raise ValueError(f"archived source bytes changed: {path}")
        recorded_reservations = {row.get("reservation_path") for row in ledger
                                 if row.get("reservation_path")}
        orphaned = sorted(path.relative_to(batch_dir).as_posix()
                          for path in (batch_dir / "logs" / "reservations").glob("*.json")
                          if path.relative_to(batch_dir).as_posix() not in recorded_reservations)
        if orphaned:
            raise ValueError("attempt reservations lack completion ledger records: " + ", ".join(orphaned))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        validation_errors.append(f"frozen batch integrity failure: {exc}")
        bad_blocks.update(block["block_id"] for block in order["blocks"])
    planned = {run["run_id"]: run for block in order["blocks"] for run in block["arms"]}
    expected_contract_hash = _sha(_canonical({
        "execution_contract": manifest["execution_contract"],
        "replay_settings": manifest["replay_settings"],
    }))
    for record in ledger:
        run = planned.get(record.get("run_id"))
        if run is None:
            validation_errors.append(f"ledger has an unplanned run: {record.get('run_id')}")
            continue
        record_errors = []
        artifacts = [("raw_path", "raw_sha256"), ("stdout_path", "stdout_sha256"),
                     ("stderr_path", "stderr_sha256")]
        if record.get("reservation_path"):
            artifacts.append(("reservation_path", "reservation_sha256"))
        for path_key, hash_key in artifacts:
            artifact = batch_dir / record[path_key]
            expected_hash = record.get(hash_key)
            if expected_hash is not None and (not artifact.is_file() or _sha(artifact.read_bytes()) != expected_hash):
                record_errors.append(f"attempt artifact hash mismatch: {artifact}")
        if record.get("contract_sha256") != expected_contract_hash:
            record_errors.append(f"contract hash mismatch: {record.get('run_id')}")
        for field in ("input_sha256", "execution_sample_hash", "estimate_view_hash",
                      "compute_profile_sha256", "comm_profile_sha256"):
            if record.get(field) != run.get(field):
                record_errors.append(f"attempt {field} mismatch: {record.get('run_id')}")
        if record.get("source_snapshot_sha256") != manifest["source_snapshot"]["digest"]:
            record_errors.append(f"attempt source hash mismatch: {record.get('run_id')}")
        validation_errors.extend(record_errors)
        if record_errors:
            bad_blocks.add(record["block_id"])
    by_block_attempt: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in ledger:
        by_block_attempt.setdefault((row["block_id"], int(row["block_attempt"])), []).append(row)
    accepted: dict[str, tuple[int, dict[str, Any]]] = {}
    for block in order["blocks"]:
        if block["block_id"] in bad_blocks:
            continue
        options = []
        for attempt in sorted({key[1] for key in by_block_attempt if key[0] == block["block_id"]}):
            rows = by_block_attempt[(block["block_id"], attempt)]
            if (len(rows) == len(order["arms"]) and {row["arm"] for row in rows} == set(order["arms"])
                    and all(row.get("validation_status") == "ok"
                            and row.get("failure_class") == "none"
                            and row.get("returncode") == 0 for row in rows)):
                options.append((attempt, {row["arm"]: row for row in rows}))
        if len(options) > 1:
            validation_errors.append(f"multiple complete attempts for {block['block_id']}")
        if options:
            accepted[block["block_id"]] = options[-1]
    rows_by_arm: dict[str, list[dict[str, Any]]] = {arm: [] for arm in order["arms"]}
    point_rows = []
    job_rows = []
    pilot_evidence: dict[str, dict[str, Any]] = {}
    mechanism_rows = []
    dynamic_diagnostics = []
    accepted_attempts = []
    for block in order["blocks"]:
        selected = accepted.get(block["block_id"])
        if not selected:
            continue
        block_error_start = len(validation_errors)
        mechanism = {"scenario": block["scenario"], "seed": block["workload_seed"],
                     "repeat": block["repeat"], "competition": False, "hol": False, "bypass": False}
        block_attempt, arms = selected
        accepted_attempts.append({"block_id": block["block_id"], "block_attempt": block_attempt})
        if any(row.get("input_sha256") != block.get("input_sha256")
               or row.get("execution_sample_hash") != block.get("execution_sample_hash")
               or row.get("estimate_view_hash") != block.get("estimate_view_hash")
               or row.get("compute_profile_sha256") != block.get("compute_profile_sha256")
               or row.get("comm_profile_sha256") != block.get("comm_profile_sha256")
               for row in arms.values()):
            validation_errors.append(f"paired input/profile hashes differ in {block['block_id']}")
        for arm, row in arms.items():
            path = batch_dir / row["raw_path"]
            result = json.loads(path.read_text())
            if row.get("raw_sha256") != _sha(path.read_bytes()):
                validation_errors.append(f"raw hash mismatch: {path}")
                bad_blocks.add(block["block_id"])
            decisions = _rank0_decisions(result)
            if arm in {"new-dynamic-fifo", "new-dynamic-ltf"}:
                from examples.jobpacer.experiments.seven_arm.gates import decision_evidence
                diagnostic = decision_evidence(decisions, "fifo" if arm.endswith("fifo") else "ltf")
                rank0 = next((rank for rank in result.get("ranks", []) if rank.get("rank") == 0), {})
                dynamic_diagnostics.append({"block_id": block["block_id"], "arm": arm,
                                            "scenario": block["scenario"], **diagnostic,
                                            "minimal_counts": rank0.get("mechanism_counts")})
                if manifest["replay_settings"].get("observation_mode") == "full":
                    validation_errors.extend(diagnostic["errors"])
                if arm.endswith("ltf"):
                    mechanism["competition"] = diagnostic["competition_observed"]
            evidence = pilot_evidence.setdefault(block["scenario"], {
                "runs": 0, "validation_ok_runs": 0,
                "static_fifo_hol": 0, "dynamic_fifo_bypass": 0,
                "ltf_distinct_frontier": 0, "ltf_selected_highest": 0,
            })
            evidence["runs"] += 1
            if result.get("validation", {}).get("status") == "ok":
                evidence["validation_ok_runs"] += 1
            if arm == "new-static-fifo" and block.get("fifo_sequence"):
                mechanism["hol"] = _static_head_hol(block["fifo_sequence"], decisions)
                evidence["static_fifo_hol"] += int(mechanism["hol"])
            if arm == "new-dynamic-fifo" and block.get("fifo_sequence"):
                mechanism["bypass"] = _dynamic_fifo_bypass(block["fifo_sequence"], decisions)
                evidence["dynamic_fifo_bypass"] += int(mechanism["bypass"])
            if arm == "new-dynamic-ltf":
                distinct, highest = _ltf_frontier_choice(decisions)
                evidence["ltf_distinct_frontier"] += int(distinct)
                evidence["ltf_selected_highest"] += int(distinct and highest)
            try:
                from examples.jobpacer.analysis.runtime_results import expected_dag_results, validate_results
                from examples.jobpacer.runtime.runtime_adapter import load_dag
                dag_path = batch_dir / "inputs" / block["sample_path"]
                dag = load_dag(dag_path, world_size=2)
                expected_data = expected_dag_results(dag.graph, 2)
                replay_check = validate_results(
                    result.get("ranks", []), 2,
                    expected=expected_data["expected"],
                    expected_nodes=expected_data["expected_nodes"],
                    digests={dag.manifest_digest},
                    expected_config={"comm_engine": ARM_CONFIG[arm][0], "policy": ARM_CONFIG[arm][1],
                                     "max_inflight": None if arm == BARE_ARM else 1,
                                     "observation_mode": manifest["replay_settings"].get("observation_mode", "full"),
                                     "estimate_view_hash": block["estimate_view_hash"], "input_hash": dag.input_hash},
                )
                if replay_check.get("status") != "ok":
                    validation_errors.append(
                        f"launch/task projection failed for {block['block_id']}/{arm}: "
                        + "; ".join(replay_check.get("errors", [])))
                expected_static = block.get("fifo_sequence") if arm.endswith("static-fifo") or arm == BARE_ARM else (
                    block.get("ltf_sequence") if arm.endswith("static-ltf") else None)
                if expected_static is not None:
                    for rank_result in result.get("ranks", []):
                        if rank_result.get("launch_sequence") != expected_static:
                            validation_errors.append(
                                f"static sequence mismatch for {block['block_id']}/{arm}/"
                                f"rank-{rank_result.get('rank')}")
                if result.get("validation", {}).get("status") != "ok":
                    validation_errors.append(f"replay validation not ok: {block['block_id']}/{arm}")
                rank_uuids = [item.get("device_uuid") for item in sorted(
                    result.get("ranks", []), key=lambda value: value.get("rank", -1))]
                expected_uuids = manifest.get("environment_at_plan", {}).get("device_uuids", [])
                if expected_uuids and rank_uuids != expected_uuids:
                    validation_errors.append(f"rank-to-GPU UUID mapping mismatch: {block['block_id']}/{arm}")
                gpu_checks = [job.get("gpu_buffer_validation", {})
                              for rank_result in result.get("ranks", [])
                              for job in rank_result.get("jobs", [])]
                if any(not checks or not all(checks.values()) for checks in gpu_checks):
                    validation_errors.append(f"GPU buffer validation failed: {block['block_id']}/{arm}")
            except (OSError, ValueError, KeyError, TypeError) as exc:
                validation_errors.append(f"could not independently validate {block['block_id']}/{arm}: {exc}")
            performance = result.get("performance", {})
            makespan = performance.get("workload_makespan_us")
            jcts = [item.get("makespan_us") for item in performance.get("job_makespans", [])]
            mean_jct = statistics.mean(jcts) if jcts else None
            measurement = {"block_id": block["block_id"], "scenario": block["scenario"],
                           "workload_seed": block["workload_seed"], "repeat": block["repeat"],
                           "arm": arm, "block_attempt": block_attempt,
                           "makespan_s": makespan / 1_000_000 if makespan is not None else None,
                           "mean_jct_s": mean_jct / 1_000_000 if mean_jct is not None else None,
                           "job_jct_s": {item["job_id"]: item["makespan_us"] / 1_000_000
                                          for item in performance.get("job_makespans", [])}}
            rows_by_arm.setdefault(arm, []).append(measurement)
            point_rows.append(measurement)
            job_rows.extend({
                "block_id": block["block_id"], "scenario": block["scenario"],
                "workload_seed": block["workload_seed"], "repeat": block["repeat"],
                "arm": arm, "job_id": job_id, "jct_s": jct_s,
            } for job_id, jct_s in measurement["job_jct_s"].items())
        if len(validation_errors) == block_error_start:
            mechanism["hol_and_bypass"] = mechanism["hol"] and mechanism["bypass"]
            mechanism_rows.append(mechanism)
        if len(validation_errors) > block_error_start:
            accepted.pop(block["block_id"], None)
            accepted_attempts = [item for item in accepted_attempts
                                 if item["block_id"] != block["block_id"]]
            point_rows = [item for item in point_rows if item["block_id"] != block["block_id"]]
            job_rows = [item for item in job_rows if item["block_id"] != block["block_id"]]
            for arm in rows_by_arm:
                rows_by_arm[arm] = [item for item in rows_by_arm[arm]
                                    if item["block_id"] != block["block_id"]]
    primary = (("new-static-fifo", "new-dynamic-fifo"), ("new-static-ltf", "new-dynamic-ltf"))
    paired: dict[str, Any] = {}
    secondary = (("old-static-fifo", "new-dynamic-fifo"), ("old-static-ltf", "new-dynamic-ltf"),
                 ("new-dynamic-fifo", "new-dynamic-ltf")) + tuple((BARE_ARM, arm) for arm in ARMS if arm != BARE_ARM)
    for baseline, candidate in primary + secondary:
        if baseline not in order["arms"] or candidate not in order["arms"]:
            continue
        candidate_points = {row["block_id"]: row for row in rows_by_arm[candidate]}
        baseline_points = {row["block_id"]: row for row in rows_by_arm[baseline]}
        comparison: dict[str, Any] = {"baseline": baseline, "candidate": candidate,
                                     "role": "primary" if (baseline, candidate) in primary else "secondary"}
        for metric in ("makespan_s", "mean_jct_s"):
            by_seed: dict[tuple[str, int], list[tuple[float, float]]] = {}
            repeat_points = []
            for block_id in sorted(set(candidate_points) & set(baseline_points)):
                base = baseline_points[block_id]
                cand = candidate_points[block_id]
                if base[metric] is None or cand[metric] is None or cand[metric] == 0:
                    continue
                delta = cand[metric] - base[metric]
                ratio = base[metric] / cand[metric]
                key = (base["scenario"], int(base["workload_seed"]))
                by_seed.setdefault(key, []).append((delta, ratio))
                repeat_points.append({"block_id": block_id, "delta": delta, "ratio": ratio})
            seed_rows = [{"scenario": scenario, "workload_seed": workload_seed,
                          "median_delta": statistics.median(item[0] for item in values),
                          "median_ratio": statistics.median(item[1] for item in values),
                          "paired_repeats": len(values)}
                         for (scenario, workload_seed), values in sorted(by_seed.items())]
            scenarios = sorted({row["scenario"] for row in seed_rows})
            scenario_deltas = {
                scenario: [row["median_delta"] for row in seed_rows if row["scenario"] == scenario]
                for scenario in scenarios}
            comparison[metric] = {
                "complete_repeat_pairs": len(repeat_points),
                "seed_block_medians": seed_rows,
                "scenario_median_delta": {scenario: statistics.median(values)
                                           for scenario, values in scenario_deltas.items()},
                "scenario_bootstrap_95pct_delta": {
                    scenario: _bootstrap_median(values, samples=bootstrap_samples,
                                                seed=analysis_seed + index)
                    for index, (scenario, values) in enumerate(scenario_deltas.items())},
                "paired_repeat_points": repeat_points,
            }
        paired[f"{baseline}__to__{candidate}"] = comparison
    failure_counts = {}
    for row in ledger:
        current = failure_counts.setdefault(row["arm"], {"attempts": 0, "failures": 0, "classes": {}})
        current["attempts"] += 1
        if row.get("failure_class") != "none":
            current["failures"] += 1
            kind = row.get("failure_class", "unknown")
            current["classes"][kind] = current["classes"].get(kind, 0) + 1
    required_runs = len(order["blocks"]) * len(order["arms"])
    pilot_gates = {}
    for scenario in ("L0-balanced", "L1-skew-tail", "D0-fork-join",
                     "D1-asymmetric-frontiers", "D2-cross-job-skew", "D3-order-and-sinks"):
        evidence = pilot_evidence.get(scenario, {"runs": 0, "validation_ok_runs": 0})
        if scenario == "L0-balanced":
            passed = evidence.get("runs", 0) > 0 and evidence.get("validation_ok_runs", 0) == evidence.get("runs")
            criterion = "all arms validate; no performance direction required"
        elif scenario in {"L1-skew-tail", "D2-cross-job-skew"}:
            passed = (evidence.get("static_fifo_hol", 0) > 0
                      and evidence.get("dynamic_fifo_bypass", 0) > 0)
            criterion = "static FIFO head HOL with another eligible task; dynamic FIFO dispatches a non-head candidate"
        elif scenario == "D1-asymmetric-frontiers":
            passed = (evidence.get("ltf_distinct_frontier", 0) > 0
                      and evidence.get("ltf_selected_highest", 0) > 0)
            criterion = "dynamic LTF observes distinct comm+tail scores among candidates and selects a highest-scoring task"
        else:
            passed = evidence.get("runs", 0) > 0 and evidence.get("validation_ok_runs", 0) == evidence.get("runs")
            criterion = "all arms validate the DAG, GPU buffers, communication projection, and terminal completion"
        pilot_gates[scenario] = {"passed": bool(passed), "criterion": criterion, **evidence}
    from examples.jobpacer.experiments.seven_arm.gates import mechanism_gates
    pilot_gates.update(mechanism_gates(mechanism_rows))
    # D2 opportunity is descriptive, never a forced benefit/selection gate.
    d2 = pilot_evidence.get("D2-cross-job-skew", {})
    pilot_gates["D2-cross-job-skew"]["passed"] = bool(d2.get("runs")) and d2.get("runs") == d2.get("validation_ok_runs")
    pilot_scope = manifest.get("scope") == "pilot"
    pilot_gates_passed = pilot_scope and all(row["passed"] for row in pilot_gates.values())
    formal_completion_checks = _formal_matrix_checks(
        manifest, json.loads((batch_dir / "inputs/suite-manifest.json").read_text()), order,
        accepted_blocks=len(accepted), accepted_runs=len(point_rows),
        validation_errors=validation_errors,
    )
    decomposition = []
    points = {(row["block_id"], row["arm"]): row for row in point_rows}
    for block in order["blocks"]:
        for policy in ("fifo", "ltf"):
            triple = [points.get((block["block_id"], arm)) for arm in
                      (f"old-static-{policy}", f"new-static-{policy}", f"new-dynamic-{policy}")]
            if not all(triple):
                continue
            old, static, dynamic = triple
            for metric in ("makespan_s", "mean_jct_s"):
                if any(row[metric] is None for row in triple):
                    continue
                decomposition.append({"block_id": block["block_id"], "policy": policy, "metric": metric,
                    "old_minus_dynamic": old[metric] - dynamic[metric],
                    "static_minus_dynamic": static[metric] - dynamic[metric],
                    "static_minus_old": static[metric] - old[metric]})
    readiness_passed = False
    try:
        from examples.jobpacer.experiments.seven_arm.gates import verify_readiness
        verify_readiness(suite_for_check, manifest["source_snapshot"]["digest"])
        readiness_passed = True
    except (ValueError, OSError, KeyError):
        pass
    arm_seed_summaries = {}
    for arm, points_for_arm in rows_by_arm.items():
        groups = {}
        for point in points_for_arm:
            groups.setdefault((point["scenario"], point["workload_seed"]), []).append(point)
        arm_seed_summaries[arm] = [
            {"scenario": scenario, "workload_seed": seed, "repeats": len(points),
             **{metric: statistics.median(point[metric] for point in points)
                for metric in ("makespan_s", "mean_jct_s") if all(point[metric] is not None for point in points)}}
            for (scenario, seed), points in sorted(groups.items())]
    analysis = {
        "schema": "jobpacer-gpu-seven-arm-analysis", "schema_version": 1,
        "batch_id": manifest["batch_id"], "planned_blocks": len(order["blocks"]),
        "planned_runs": required_runs, "accepted_complete_blocks": len(accepted),
        "accepted_runs": len(point_rows), "missing_blocks": sorted(
            set(block["block_id"] for block in order["blocks"]) - set(accepted)),
        "attempted_runs": len(ledger), "failure_counts_by_arm": failure_counts,
        "accepted_attempts": accepted_attempts, "paired_comparisons": paired,
        "pilot_scope": pilot_scope,
        "dynamic_dispatch_diagnostics": dynamic_diagnostics,
        "paired_cost_identity": decomposition,
        "arm_raw_distributions": rows_by_arm,
        "matrix_complete": len(accepted) == len(order["blocks"]) and len(point_rows) == required_runs,
        "semantic_checks_passed": bool(point_rows) and not validation_errors,
        "mechanism_gates_passed": pilot_gates_passed,
        "measurement_checks_passed": readiness_passed,
        "arm_seed_summaries": arm_seed_summaries,
        "net_performance_result": "see paired comparisons; intervals are exploratory, no multiplicity correction",
        "pilot_mechanism_gates": pilot_gates,
        "pilot_gates_passed": pilot_gates_passed,
        "validation_errors": validation_errors,
        "formal_completion_checks": formal_completion_checks,
        "complete_formal_matrix": all(formal_completion_checks.values()),
        "interpretation": "paired seed-block medians; limited to these frozen workload samples",
    }
    gate_base = {"source_snapshot_sha256": manifest["source_snapshot"]["digest"],
                 "generator_version": suite_for_check.get("generator_version"),
                 "profiles": {key: value["sha256"] for key, value in suite_for_check.get("profiles", {}).items()},
                 "artifacts": [{"path": str((batch_dir / row["raw_path"]).resolve()), "sha256": row["raw_sha256"]}
                               for row in ledger if row.get("raw_sha256")]}
    for gate, checks in (
        ("mechanism", {"seed_thresholds": pilot_gates_passed,
                       "semantics": analysis["semantic_checks_passed"], "matrix": analysis["matrix_complete"]}),
        ("rehearsal", {"seven_arms": set(order["arms"]) == set(ARMS),
                       "all_scenarios": {row["scenario"] for row in point_rows} == set(SCENARIOS),
                       "semantics": analysis["semantic_checks_passed"], "matrix": analysis["matrix_complete"]}),
    ):
        _write_json(batch_dir / f"{gate}-audit.json", {**gate_base, "gate": gate,
                    "checks": checks, "passed": all(checks.values())})
    _write_json(batch_dir / "analysis.json", analysis)
    _write_json(batch_dir / "tables" / "analysis.json", analysis)
    if point_rows:
        fields = ("block_id", "scenario", "workload_seed", "repeat", "arm", "block_attempt",
                  "makespan_s", "mean_jct_s")
        with (batch_dir / "tables" / "accepted-points.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in point_rows:
                writer.writerow({key: row.get(key) for key in fields})
    if job_rows:
        fields = ("block_id", "scenario", "workload_seed", "repeat", "arm", "job_id", "jct_s")
        with (batch_dir / "tables" / "job-jct.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(job_rows)
    checksums = []
    for path in sorted(item for item in batch_dir.rglob("*")
                       if item.is_file() and item.name != "SHA256SUMS.txt"):
        checksums.append(f"{_sha(path.read_bytes())}  {path.relative_to(batch_dir).as_posix()}")
    (batch_dir / "SHA256SUMS.txt").write_text("\n".join(checksums) + "\n")
    return analysis
