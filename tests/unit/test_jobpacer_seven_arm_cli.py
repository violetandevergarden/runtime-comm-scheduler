from __future__ import annotations

import json
from pathlib import Path

from examples.jobpacer.scripts import run_gpu_seven_arm
from examples.jobpacer.experiments.seven_arm import batch, checks


def test_unified_cli_advertises_all_six_workflow_commands():
    help_text = run_gpu_seven_arm.build_parser().format_help()
    for command in ("prepare", "qualify", "check", "run", "analyze", "finalize"):
        assert command in help_text


def test_check_status_reads_evidence_without_starting_diagnostics(tmp_path, monkeypatch, capsys):
    suite_path = tmp_path / "suite-manifest.json"
    evidence_path = tmp_path / "software-audit.json"
    evidence_path.write_text("{}\n")
    suite = {"scope": "formal", "status": "frozen-inputs-awaiting-readiness",
             "profiles": {}, "readiness": {
                 "software": {"path": str(evidence_path), "sha256": "stale-hash"},
             }, "bare_qualification": {}}
    suite_path.write_text(json.dumps(suite))
    before = suite_path.read_bytes()
    monkeypatch.setattr(checks, "source_snapshot", lambda: {"digest": "test-source"})

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("status mode must not launch a diagnostic")

    monkeypatch.setattr(checks, "run_measurement_check", must_not_run)
    monkeypatch.setattr(checks, "run_recovery_check", must_not_run)
    monkeypatch.setattr(checks, "run_mechanism_check", must_not_run)
    assert run_gpu_seven_arm.main([
        "check", "--status", "--suite-manifest", str(suite_path),
    ]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "not-ready"
    assert result["replay_started"] is False
    assert result["gates"]["software"]["checks"]["file_hash"] is False
    assert suite_path.read_bytes() == before
    assert sorted(path.name for path in tmp_path.iterdir()) == ["software-audit.json", "suite-manifest.json"]


def test_run_plan_only_creates_a_plan_and_never_calls_batch_executor(tmp_path, monkeypatch, capsys):
    suite_path = tmp_path / "suite-manifest.json"
    suite_path.write_text("{}")
    batch_dir = tmp_path / "batch"
    calls = []

    def create(*args, **kwargs):
        calls.append((args, kwargs))
        return {"stage": "pilot-block-smoke", "planned_blocks": 1, "planned_runs": 6}

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("plan-only must not start replay")

    monkeypatch.setattr(batch, "create_batch", create)
    monkeypatch.setattr(batch, "run_batch", must_not_run)
    assert run_gpu_seven_arm.main([
        "run", "--plan-only", "--suite-manifest", str(suite_path),
        "--batch-dir", str(batch_dir), "--arms", "a,b", "--repeats", "1",
    ]) == 0
    assert len(calls) == 1
    assert calls[0][0] == (suite_path, batch_dir)
    result = json.loads(capsys.readouterr().out)
    assert result["batch_dir"] == str(batch_dir.resolve())
    assert result["runs"] == 6


def test_analyze_dispatch_never_calls_batch_executor(monkeypatch, capsys, tmp_path):
    batch_dir = tmp_path / "existing-batch"
    monkeypatch.setattr(batch, "analyze_batch", lambda *_args, **_kwargs: {
        "accepted_complete_blocks": 2, "planned_blocks": 3,
        "complete_formal_matrix": False, "validation_errors": [],
    })

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("analyze must not start replay")

    monkeypatch.setattr(batch, "run_batch", must_not_run)
    assert run_gpu_seven_arm.main(["analyze", "--batch-dir", str(batch_dir)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result == {"accepted_blocks": 2, "planned_blocks": 3,
                      "complete_formal_matrix": False, "validation_errors": 0}


def test_source_snapshot_hashes_new_implementation_and_entrypoint_only():
    paths = {row["path"] for row in batch.source_snapshot()["files"]}
    assert "examples/jobpacer/experiments/seven_arm/batch.py" in paths
    assert "examples/jobpacer/experiments/seven_arm/checks.py" in paths
    assert "examples/jobpacer/experiments/seven_arm/suite.py" in paths
    assert "examples/jobpacer/experiments/__init__.py" in paths
    assert "examples/jobpacer/gpu/gpu_dag_resources.py" in paths
    assert "examples/jobpacer/scripts/run_gpu_seven_arm.py" in paths
    seven_arm_scripts = {path.name for path in Path("examples/jobpacer/scripts").glob("*seven_arm*.py")}
    assert seven_arm_scripts == {"run_gpu_seven_arm.py"}
