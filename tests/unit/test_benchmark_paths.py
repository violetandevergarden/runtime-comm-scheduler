from pathlib import Path

import json
import pytest

from examples.jobpacer import paths as benchmark_paths
from examples.jobpacer.paths import (
    is_formal_experiment_input, resolve_migrated_path, repository_path,
)
from examples.jobpacer.analysis.visualize import load_manifest


ROOT = Path(__file__).resolve().parents[2]
PHASE3_MAP = ROOT / "benchmark/phase3/results/migration-map.json"
PHASE3_MOVED = ROOT / "benchmark/phase3/experiments/dag-semantics/smoke/multi-group.json"
PHASE12_MAP = ROOT / "benchmark/phase1.2/results/migration-map.json"
PHASE12_MOVED = ROOT / "benchmark/phase1.2/results/capacity/20260919T143653Z-9536db/raw"


def test_repository_path_is_independent_of_current_directory():
    assert repository_path("benchmark/phase3/README.md") == ROOT / "benchmark/phase3/README.md"


def test_migration_map_resolves_file_and_directory_suffix(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark_paths, "REPOSITORY_ROOT", tmp_path)
    target_file = tmp_path / "benchmark/new/file.json"
    target_dir = tmp_path / "benchmark/new/batch/raw"
    target_file.parent.mkdir(parents=True)
    target_file.write_text("{}")
    target_dir.mkdir(parents=True)
    mapping = tmp_path / "migration-map.json"
    mapping.write_text(json.dumps({"entries": [
        {"old_path": "benchmark/old/file.json", "new_path": "benchmark/new/file.json"},
        {"old_path": "benchmark/old/batch", "new_path": "benchmark/new/batch", "directory": True},
    ]}))
    assert resolve_migrated_path("benchmark/old/file.json", mapping) == target_file
    assert resolve_migrated_path("benchmark/old/batch/raw", mapping) == target_dir


@pytest.mark.skipif(not (PHASE3_MAP.is_file() and PHASE3_MOVED.is_file()),
                    reason="archived Phase 3 migration map/input is absent from this checkout")
def test_phase3_smoke_input_legacy_path_resolves_to_moved_file():
    mapping = PHASE3_MAP
    resolved = resolve_migrated_path("benchmark/phase3/multi-group.json", mapping)
    assert resolved == PHASE3_MOVED
    assert resolved.is_file()


@pytest.mark.skipif(not (PHASE12_MAP.is_file() and PHASE12_MOVED.is_dir()),
                    reason="archived Phase 1.2 migration map/result fixture is absent from this checkout")
def test_phase12_legacy_result_directory_resolves_with_suffix():
    mapping = PHASE12_MAP
    resolved = resolve_migrated_path(
        "benchmark/phase1.2/result/batches/20260919T143653Z-9536db/raw",
        mapping,
    )
    assert resolved == PHASE12_MOVED
    assert resolved.is_dir()


@pytest.mark.skipif(not (PHASE12_MAP.is_file() and (PHASE12_MOVED.parent / "manifest.json").is_file()),
                    reason="archived Phase 1.2 batch fixture is absent from this checkout")
def test_visualizer_reads_a_phase12_batch_through_its_legacy_path():
    manifest = load_manifest("benchmark/phase1.2/result/batches/20260919T143653Z-9536db")
    assert manifest["complete"] is True
    assert manifest["batch_id"] == "20260919T143653Z-9536db"


def test_preserved_smoke_fixtures_are_not_classified_as_formal_inputs():
    smoke = ROOT / "benchmark/phase3/experiments/dag-semantics/smoke/linear.json"
    formal = ROOT / "benchmark/phase3/experiments/baseline/L0-balanced.json"
    input_root = ROOT / "benchmark/phase3/experiments"
    assert not is_formal_experiment_input(smoke, input_root)
    assert is_formal_experiment_input(formal, input_root)
