from pathlib import Path

from examples.jobpacer.analysis.benchmark_paths import (
    is_formal_experiment_input, resolve_migrated_path, repository_path,
)
from examples.jobpacer.analysis.visualize import load_manifest


ROOT = Path(__file__).resolve().parents[2]


def test_repository_path_is_independent_of_current_directory():
    assert repository_path("benchmark/phase3/README.md") == ROOT / "benchmark/phase3/README.md"


def test_phase3_smoke_input_legacy_path_resolves_to_moved_file():
    mapping = ROOT / "benchmark/phase3/results/migration-map.json"
    resolved = resolve_migrated_path("benchmark/phase3/multi-group.json", mapping)
    assert resolved == ROOT / "benchmark/phase3/experiments/dag-semantics/smoke/multi-group.json"
    assert resolved.is_file()


def test_phase12_legacy_result_directory_resolves_with_suffix():
    mapping = ROOT / "benchmark/phase1.2/results/migration-map.json"
    resolved = resolve_migrated_path(
        "benchmark/phase1.2/result/batches/20260919T143653Z-9536db/raw",
        mapping,
    )
    assert resolved == ROOT / "benchmark/phase1.2/results/capacity/20260919T143653Z-9536db/raw"
    assert resolved.is_dir()


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
