import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from examples.jobpacer.workloads import Workload
from examples.jobpacer.workload_builder import (
    convert_document,
)


def test_conversion_counts_intermediate_compute_once_and_keeps_collectives():
    source = {
        "id": "two-collectives",
        "metadata": {"seed": 7, "duration_unit_s": 0.001},
        "tasks": [
            {"id": "c0", "kind": "compute", "duration": 2, "dependencies": []},
            {"id": "m0", "kind": "communication", "duration": 1, "num_bytes": 4096, "dependencies": ["c0"]},
            {"id": "c1", "kind": "compute", "duration": 3, "dependencies": ["m0"]},
            {"id": "m1", "kind": "communication", "duration": 2, "num_bytes": 8192, "dependencies": ["c1"]},
            {"id": "c2", "kind": "compute", "duration": 4, "dependencies": ["m1"]},
        ],
    }

    workload = Workload.from_dict(convert_document(source))
    first, second = workload.jobs[0].communications
    assert (first.num_bytes, second.num_bytes) == (4096, 8192)
    assert first.producer_compute_s == 0.002
    assert first.consumer_compute_s == 0.003
    assert second.producer_compute_s == 0
    assert second.consumer_compute_s == 0.004
    assert sum(
        item.producer_compute_s + item.consumer_compute_s
        for item in workload.jobs[0].communications
    ) == pytest.approx(0.009)


def test_adjacent_communications_are_not_merged():
    source = {
        "tasks": [
            {"id": "m0", "kind": "communication", "duration": 0, "num_bytes": 4096, "dependencies": []},
            {"id": "m1", "kind": "communication", "duration": 0, "num_bytes": 8192, "dependencies": ["m0"]},
        ]
    }
    communications = convert_document(source)["jobs"][0]["communications"]
    assert [item["num_bytes"] for item in communications] == [4096, 8192]


def test_checked_in_benchmarks_match_converter_output():
    root = Path(__file__).parents[2] / "benchmark" / "phase1.2"
    for source_path in sorted((root / "dag").glob("*.json")):
        converted = convert_document(json.loads(source_path.read_text()))
        expected = json.loads((root / "workloads" / source_path.name).read_text())
        assert converted == expected
        Workload.from_dict(converted)
