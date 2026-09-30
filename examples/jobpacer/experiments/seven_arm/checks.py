"""Read readiness evidence and route explicitly requested seven-arm diagnostics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from examples.jobpacer.experiments.seven_arm.batch import source_snapshot
from examples.jobpacer.experiments.seven_arm.gates import READINESS_GATES, REQUIRED_CHECKS


def check_status(suite_manifest_path: Path, *,
                 audit_paths: Mapping[str, Path] | None = None,
                 qualification_path: Path | None = None) -> dict[str, Any]:
    """Inspect existing qualification/readiness files without probing or launching GPUs."""
    suite_manifest_path = suite_manifest_path.resolve(strict=True)
    suite = json.loads(suite_manifest_path.read_text())
    digest = source_snapshot()["digest"]
    profiles = {key: value["sha256"] for key, value in suite.get("profiles", {}).items()}
    errors: list[str] = []

    qualification = suite.get("bare_qualification", {})
    evidence = qualification.get("evidence") or {}
    evidence_path = qualification_path or (Path(evidence["evidence_path"])
                                            if evidence.get("evidence_path") else None)
    qualification_status = {"passed": False, "path": str(evidence_path) if evidence_path else None}
    if evidence_path is None:
        qualification_status["error"] = "bare qualification evidence is not attached"
    else:
        try:
            raw = evidence_path.read_bytes()
            audit = json.loads(raw)
            from examples.jobpacer.experiments.seven_arm.batch import _canonical, _sha
            from examples.jobpacer.runtime.dag_comm_adapters import BARE_CONTRACT_VERSION
            expected_profile_sha = _sha(_canonical({
                key: value["sha256"] for key, value in sorted(suite["profiles"].items())
            }))
            attached_path = Path(evidence["evidence_path"]) if evidence.get("evidence_path") else None
            checks = {
                "file_hash": (not evidence.get("evidence_sha256")
                              or hashlib.sha256(raw).hexdigest() == evidence["evidence_sha256"]),
                "passed": audit.get("passed") is True,
                "arm": audit.get("arm") == "bare-ordered",
                "contract": audit.get("contract_version") == BARE_CONTRACT_VERSION,
                "profiles": audit.get("profile_sha256") == expected_profile_sha,
                "source": audit.get("source_snapshot_sha256") == digest,
                "suite_binding": (qualification.get("status") == "qualified"
                                  and attached_path is not None
                                  and attached_path.resolve() == evidence_path.resolve()),
            }
            qualification_status.update({"passed": all(checks.values()), "checks": checks})
            if not qualification_status["passed"]:
                qualification_status["error"] = "bare qualification is stale, failed, or belongs to another source snapshot"
        except (OSError, json.JSONDecodeError) as exc:
            qualification_status["error"] = str(exc)

    supplied_audits = dict(audit_paths or {})
    attached_audits = suite.get("readiness", {})
    gate_status: dict[str, dict[str, Any]] = {}
    for gate in READINESS_GATES:
        attached = attached_audits.get(gate, {})
        path = supplied_audits.get(gate)
        if path is None and attached.get("path"):
            path = Path(attached["path"])
        if path is None:
            gate_status[gate] = {"passed": False, "error": "evidence is not attached or supplied"}
            continue
        try:
            raw = path.read_bytes()
            audit = json.loads(raw)
            bound_path = Path(attached["path"]) if attached.get("path") else None
            bound_hash_matches = (
                bound_path is None or path.resolve() != bound_path.resolve()
                or not attached.get("sha256")
                or hashlib.sha256(raw).hexdigest() == attached["sha256"]
            )
            checks = {
                "file_hash": bound_hash_matches,
                "gate": audit.get("gate") == gate,
                "passed": audit.get("passed") is True,
                "source": audit.get("source_snapshot_sha256") == digest,
                "generator": audit.get("generator_version") == suite.get("generator_version"),
                "profiles": audit.get("profiles") == profiles,
                "checks": (isinstance(audit.get("checks"), dict)
                           and REQUIRED_CHECKS[gate] <= set(audit["checks"])
                           and all(value is True for value in audit["checks"].values())),
                "artifacts": bool(audit.get("artifacts")),
            }
            if checks["artifacts"]:
                checks["artifact_hashes"] = all(
                    Path(item["path"]).is_file()
                    and hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() == item["sha256"]
                    for item in audit["artifacts"]
                )
            gate_status[gate] = {"passed": all(checks.values()), "path": str(path), "checks": checks}
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            gate_status[gate] = {"passed": False, "path": str(path), "error": str(exc)}

    ready = (suite.get("scope") == "formal"
             and suite.get("status") == "frozen-inputs-ready-for-formal"
             and qualification_status["passed"]
             and all(row["passed"] for row in gate_status.values()))
    if not qualification_status["passed"]:
        errors.append(str(qualification_status.get("error", "bare qualification pending")))
    errors.extend(f"{gate}: {row.get('error', 'audit failed')}" for gate, row in gate_status.items()
                  if not row["passed"])
    return {"status": "ready-for-formal" if ready else "not-ready",
            "suite_manifest": str(suite_manifest_path), "source_snapshot_sha256": digest,
            "bare_qualification": qualification_status, "gates": gate_status,
            "errors": errors, "replay_started": False}


def run_measurement_check(action: str, *, output: Path,
                          suite_manifest: Path | None = None,
                          pairs: int = 5, timeout: float = 60.0) -> dict[str, Any]:
    from examples.jobpacer.diagnostics import gpu_measurement as measurement

    if action == "plan":
        if suite_manifest is None:
            raise ValueError("measurement plan requires a suite manifest")
        return measurement.plan(suite_manifest, output, pairs=pairs, timeout=timeout)
    if action == "run":
        return measurement.run(output)
    if action == "analyze":
        return measurement.analyze(output)
    raise ValueError(f"unsupported measurement action: {action}")


def run_recovery_check(suite_manifest: Path, output: Path) -> dict[str, Any]:
    from examples.jobpacer.diagnostics.gpu_recovery import run

    return run(suite_manifest.resolve(), output.resolve())


def run_mechanism_check(output: Path, *, message_bytes: int = 1 << 20,
                        repeats: int = 5, warmup: int = 5,
                        timeout: float = 30.0) -> dict[str, Any]:
    from examples.jobpacer.diagnostics.bare_nccl_mechanism import run_mechanism_check as run

    return run(output, message_bytes=message_bytes, repeats=repeats,
               warmup=warmup, timeout=timeout)
