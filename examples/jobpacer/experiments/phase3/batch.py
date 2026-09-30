"""Freeze, run, and resume paired Phase 3 GPU experiment blocks."""

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

from examples.jobpacer.experiments.phase3.suite import (
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
    ROOT / "examples/jobpacer/gloo",
    ROOT / "examples/jobpacer/gpu",
    ROOT / "examples/jobpacer/diagnostics",
    ROOT / "examples/jobpacer/scripts",
    ROOT / "examples/jobpacer/experiments",
    ROOT / "examples/jobpacer/analysis/runtime_results.py",
    ROOT / "examples/jobpacer/analysis/phase3_results.py",
    ROOT / "examples/jobpacer/paths.py",
    ROOT / "examples/jobpacer/runtime/comm_profile.py",
    ROOT / "examples/jobpacer/gloo/workloads.py",
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
    seen_paths: set[str] = set()
    for source in SOURCE_PATHS:
        candidates = sorted(source.rglob("*.py")) if source.is_dir() else [source]
        for path in candidates:
            if not path.is_file():
                continue
            relative = path.relative_to(ROOT).as_posix()
            if relative in seen_paths:
                continue
            seen_paths.add(relative)
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
    from examples.jobpacer.experiments.phase3.gates import verify_readiness
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
    args = [sys.executable, "-m", "examples.jobpacer.runtime.replay_launcher",
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
