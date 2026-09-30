"""Attach independently archived readiness audits and freeze the formal launch plan."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from examples.jobpacer.scripts.gpu_seven_arm_gates import READINESS_GATES, verify_readiness
from examples.jobpacer.scripts.run_gpu_seven_arm import (
    _sha, _verify_bare_qualification, _verify_frozen_inputs, _write_json, source_snapshot,
)


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-manifest", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--audit", action="append", required=True, help="GATE=PATH; once for each gate")
    args = parser.parse_args()
    audits = {}
    for item in args.audit:
        gate, path = item.split("=", 1)
        if gate in audits:
            parser.error(f"duplicate gate: {gate}")
        audits[gate] = Path(path)
    print(json.dumps(finalize(args.suite_manifest, args.qualification, audits), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
