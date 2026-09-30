"""Evidence checks for v2 preparation; never infer readiness from matrix size."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


def decision_evidence(records: list[dict[str, Any]], policy: str) -> dict[str, Any]:
    """Use the snapshot at the decision's exact coordinator timestamp."""
    snapshots = {}
    rows = []
    errors = []
    for record in records:
        if record.get("kind") == "policy_snapshot":
            snapshots[record.get("now")] = record
        if record.get("kind") != "decision" or record.get("decision") != "dispatch":
            continue
        snapshot = snapshots.get(record.get("now"))
        if snapshot is None:
            errors.append("dispatch has no matching coordinator snapshot")
            continue
        eligible = snapshot.get("eligible", [])
        if not eligible or {item["task_id"] for item in eligible} != set(record.get("eligible", [])):
            errors.append("dispatch and snapshot candidate sets differ")
            continue
        fifo = min(eligible, key=lambda item: (item["eligible_seq"], item["task_id"]))
        ltf = min(eligible, key=lambda item: (
            -(item["estimated_comm_s"] + item["remaining_tail_s"]), item["eligible_seq"], item["task_id"]))
        scores = {item["task_id"]: item["estimated_comm_s"] + item["remaining_tail_s"] for item in eligible}
        expected = fifo["task_id"] if policy == "fifo" else ltf["task_id"]
        correct = record["task_id"] == expected
        if not correct:
            errors.append(f"policy choice mismatch at decision {record.get('decision_seq')}")
        rows.append({"decision_seq": record.get("decision_seq"), "now": record["now"],
                     "candidates": eligible, "scores": scores, "selected": record["task_id"],
                     "fifo": fifo["task_id"], "ltf": ltf["task_id"], "correct": correct,
                     "competition": len(set(scores.values())) > 1 and fifo["task_id"] != ltf["task_id"]})
    return {"dispatches": rows, "errors": errors,
            "candidate_counts": dict(Counter(str(len(row["candidates"])) for row in rows)),
            "competition_observed": any(row["competition"] and row["correct"] for row in rows)}


def mechanism_gates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Rows represent complete, independently validated seed/repeat blocks."""
    result = {}
    for scenario, flag in (("D1-asymmetric-frontiers", "competition"), ("L1-skew-tail", "hol_and_bypass")):
        selected = [row for row in rows if row["scenario"] == scenario]
        seeds = sorted({row["seed"] for row in selected})
        counts = {str(seed): len({row["repeat"] for row in selected
                                 if row["seed"] == seed and row.get(flag)}) for seed in seeds}
        complete = len(seeds) >= 3 and all(
            {row["repeat"] for row in selected if row["seed"] == seed} >= {0, 1, 2} for seed in seeds)
        result[scenario] = {"passed": complete and sum(value >= 2 for value in counts.values()) >= 2,
                            "complete_three_seed_diagnostics": complete, "triggered_repeats_by_seed": counts,
                            "criterion": "at least two of three independent seeds trigger in at least two of three repeats"}
    return result


READINESS_GATES = ("software", "backend", "mechanism", "measurement", "rehearsal", "recovery", "budget")

REQUIRED_CHECKS = {
    "software": {"targeted_unit", "gloo", "nccl", "diff_clean"},
    "backend": {"normal_paths", "failure_paths", "multi_inflight", "supported_backend"},
    "mechanism": {"seed_thresholds", "semantics", "matrix"},
    "measurement": {"all_planned_runs", "semantic_checks", "five_pairs_per_path_and_kind", "minimal_mechanism_preserved"},
    "rehearsal": {"seven_arms", "all_scenarios", "semantics", "matrix"},
    "recovery": {"startup_failure", "bounded_cleanup", "fresh_rendezvous", "whole_block_retry", "interruption_resume", "hash_checks"},
    "budget": {"seven_arm_history", "wall_estimate", "disk_estimate", "available_disk"},
}


def verify_readiness(suite: Mapping[str, Any], source_digest: str) -> None:
    """Require archived, source-bound evidence for every stage before formal launch."""
    evidence = suite.get("readiness", {})
    for gate in READINESS_GATES:
        record = evidence.get(gate)
        if not isinstance(record, dict) or not record.get("path"):
            raise ValueError(f"formal readiness gate {gate} is pending")
        path = Path(record["path"])
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != record.get("sha256"):
            raise ValueError(f"formal readiness gate {gate} evidence missing or changed")
        audit = json.loads(path.read_text())
        if (audit.get("gate") != gate or audit.get("passed") is not True
                or audit.get("source_snapshot_sha256") != source_digest
                or audit.get("generator_version") != suite.get("generator_version")
                or audit.get("profiles") != {key: value["sha256"] for key, value in suite["profiles"].items()}):
            raise ValueError(f"formal readiness gate {gate} failed or belongs to another revision")
        artifacts = audit.get("artifacts")
        checks = audit.get("checks")
        if (not artifacts or not checks or not REQUIRED_CHECKS[gate] <= set(checks)
                or not all(value is True for value in checks.values())):
            raise ValueError(f"formal readiness gate {gate} lacks checks and underlying artifacts")
        for item in artifacts:
            artifact = Path(item["path"])
            if not artifact.is_file() or hashlib.sha256(artifact.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError(f"formal readiness gate {gate} underlying artifact changed")
