"""Attach independently archived readiness audits and freeze the formal launch plan."""
from __future__ import annotations

import json
from pathlib import Path

from examples.jobpacer.experiments.seven_arm.gates import READINESS_GATES, verify_readiness
from examples.jobpacer.experiments.seven_arm.batch import (
    _canonical, _sha, _verify_bare_qualification, _verify_frozen_inputs,
    _write_json, analyze_batch, source_snapshot,
)
from examples.jobpacer.experiments.seven_arm.suite import ARMS, MECHANISM_PILOT_ARMS


def seal_formal_inputs(suite_dir: Path, pilot_batch_dir: Path) -> dict:
    """Attach a completed independent pilot to formal candidate inputs."""
    suite_dir = suite_dir.resolve(strict=True)
    pilot_batch_dir = pilot_batch_dir.resolve(strict=True)
    formal_path = suite_dir / "suite-manifest.json"
    formal = json.loads(formal_path.read_text())
    pilot = json.loads((pilot_batch_dir / "inputs/suite-manifest.json").read_text())
    batch = json.loads((pilot_batch_dir / "manifest.json").read_text())
    order = json.loads((pilot_batch_dir / "order.json").read_text())
    analysis = analyze_batch(pilot_batch_dir)
    if formal.get("scope") != "formal" or formal.get("status") != "profiled-awaiting-E2-pilot":
        raise ValueError("formal input suite is not in the profiled-awaiting-pilot state")
    if pilot.get("scope") != "pilot" or batch.get("stage") not in {"pilot-six-arm", "pilot-seven-arm-rehearsal"}:
        raise ValueError("seal requires a separately seeded six-arm pilot batch")
    if (set(batch.get("arms", [])) not in (set(MECHANISM_PILOT_ARMS), set(ARMS))
            or order.get("block_count") != 54 or order.get("run_count") != 54 * len(batch["arms"])):
        raise ValueError("pilot must contain six scenarios, three pilot seeds, three repeats, and six implemented arms")
    if analysis.get("accepted_complete_blocks") != 54:
        raise ValueError("pilot is missing complete paired blocks")
    if analysis.get("validation_errors") or not analysis.get("pilot_gates_passed"):
        raise ValueError("pilot data validation or one or more scenario mechanism gates failed")
    if set(pilot.get("workload_seeds", [])) & set(formal.get("workload_seeds", [])):
        raise ValueError("pilot and formal workload seeds must be disjoint")
    if _canonical(pilot.get("profiles")) != _canonical(formal.get("profiles")):
        raise ValueError("pilot and formal suites must use identical frozen profiles")
    pilot_topology = {row["scenario"]: row.get("topology_hash") for row in pilot.get("samples", [])}
    formal_topology = {row["scenario"]: row.get("topology_hash") for row in formal.get("samples", [])}
    if any(not pilot_topology.get(scenario) or pilot_topology.get(scenario) != formal_topology.get(scenario)
           for scenario in formal_topology):
        raise ValueError("pilot and formal scenario topology/shape/program roles differ")
    formal["status"] = "frozen-inputs-awaiting-readiness"
    formal["pilot_evidence"] = {
        "batch_id": batch["batch_id"],
        "batch_manifest_sha256": _sha((pilot_batch_dir / "manifest.json").read_bytes()),
        "order_sha256": _sha((pilot_batch_dir / "order.json").read_bytes()),
        "ledger_sha256": _sha((pilot_batch_dir / "runs.jsonl").read_bytes()),
        "analysis_sha256": _sha((pilot_batch_dir / "analysis.json").read_bytes()),
        "accepted_blocks": analysis["accepted_complete_blocks"],
        "pilot_gates": analysis["pilot_mechanism_gates"],
        "scope": "six-arm mechanism pilot; bare backend, measurement and seven-arm rehearsal gates remain required",
    }
    _write_json(formal_path, formal)
    return formal


def finalize(suite_path: Path, qualification_path: Path, audits: dict[str, Path]) -> dict:
    suite = json.loads(suite_path.read_text())
    if suite.get("scope") != "formal" or suite.get("status") != "frozen-inputs-awaiting-readiness":
        raise ValueError("seal the independent pilot before final readiness attachment")
    if set(audits) != set(READINESS_GATES):
        raise ValueError(f"required audits: {READINESS_GATES}")
    _verify_frozen_inputs(suite_path.parent, suite)
    qualification = json.loads(qualification_path.read_text())
    suite["bare_qualification"] = {
        "arm": "bare-ordered", "status": "qualified", "evidence": {
            "evidence_path": str(qualification_path.resolve()),
            "evidence_sha256": _sha(qualification_path.read_bytes()),
            **{key: qualification[key] for key in ("contract_version", "source_snapshot_sha256",
                                                    "profile_sha256", "device_uuids")},
        },
    }
    suite["readiness"] = {gate: {"path": str(path.resolve()), "sha256": _sha(path.read_bytes())}
                          for gate, path in audits.items()}
    _verify_bare_qualification(suite)
    verify_readiness(suite, source_snapshot()["digest"])
    suite["status"] = "frozen-inputs-ready-for-formal"
    _write_json(suite_path, suite)
    return {"status": suite["status"], "suite_manifest": str(suite_path.resolve()),
            "sha256": _sha(suite_path.read_bytes()), "formal_runs_started": 0}
