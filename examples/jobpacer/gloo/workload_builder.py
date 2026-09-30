#!/usr/bin/env python3
"""Convert chain DAG JSON files into JobPacer workload manifests.

Input ``duration`` values are seconds unless ``metadata.duration_unit_s`` is
provided. Communication tasks must carry their real ``num_bytes``; execution
time is not a reliable way to infer message size.

Usage:
    python examples/jobpacer/workload_builder.py INPUT_DIR OUTPUT_DIR
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_OP = "all_reduce"
DEFAULT_SEED = 42


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def preprocess_zero_duration(data: dict[str, Any]) -> dict[str, Any]:
    """Remove zero-duration compute nodes and reconnect their chain."""

    tasks = {task["id"]: dict(task) for task in data["tasks"]}
    if len(tasks) != len(data["tasks"]):
        raise ValueError("task ids must be unique")

    while True:
        zero_ids = [
            task_id
            for task_id, task in tasks.items()
            if task["kind"] == "compute" and task["duration"] == 0
        ]
        if not zero_ids:
            break

        successors = {task_id: [] for task_id in tasks}
        for task in tasks.values():
            for dependency in task["dependencies"]:
                if dependency not in tasks:
                    raise ValueError(
                        f"task {task['id']!r} has unknown dependency {dependency!r}"
                    )
                successors[dependency].append(task["id"])

        for task_id in zero_ids:
            if task_id not in tasks:
                continue
            task = tasks[task_id]
            for successor_id in successors[task_id]:
                successor = tasks[successor_id]
                successor["dependencies"] = [
                    dependency
                    for dependency in successor["dependencies"]
                    if dependency != task_id
                ]
                for dependency in task["dependencies"]:
                    if dependency not in successor["dependencies"]:
                        successor["dependencies"].append(dependency)
            del tasks[task_id]

    return {**data, "tasks": list(tasks.values())}


def build_chains(data: dict[str, Any]) -> list[list[dict[str, Any]]]:
    """Validate that the graph is disjoint chains and return those chains."""

    tasks = {task["id"]: task for task in data["tasks"]}
    if len(tasks) != len(data["tasks"]):
        raise ValueError("task ids must be unique")

    successors: dict[Any, list[Any]] = {task_id: [] for task_id in tasks}
    for task in tasks.values():
        dependencies = task.get("dependencies", [])
        if len(dependencies) > 1:
            raise ValueError(f"task {task['id']!r} is not part of a chain")
        for dependency in dependencies:
            if dependency not in tasks:
                raise ValueError(
                    f"task {task['id']!r} has unknown dependency {dependency!r}"
                )
            successors[dependency].append(task["id"])

    branching = [task_id for task_id, values in successors.items() if len(values) > 1]
    if branching:
        raise ValueError(f"tasks branch and cannot be represented as jobs: {branching}")

    chains: list[list[dict[str, Any]]] = []
    visited: set[Any] = set()
    for root in (task for task in data["tasks"] if not task.get("dependencies")):
        chain = []
        current = root["id"]
        while True:
            if current in visited:
                raise ValueError(f"cycle or shared task detected at {current!r}")
            visited.add(current)
            chain.append(tasks[current])
            if not successors[current]:
                break
            current = successors[current][0]
        chains.append(chain)

    if visited != set(tasks):
        raise ValueError("graph contains a cycle")
    return chains


def merge_adjacent_compute(chain: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge compute spans only; every communication remains a collective."""

    merged: list[dict[str, Any]] = []
    for task in chain:
        if task["kind"] not in {"compute", "communication"}:
            raise ValueError(f"unknown task kind {task['kind']!r}")
        if task["duration"] < 0:
            raise ValueError(f"task {task['id']!r} has negative duration")
        if merged and task["kind"] == merged[-1]["kind"] == "compute":
            merged[-1]["duration"] += task["duration"]
        else:
            merged.append(dict(task))
    return merged


def _message_bytes(task: dict[str, Any]) -> int:
    num_bytes = task.get("num_bytes")
    if not isinstance(num_bytes, int) or isinstance(num_bytes, bool):
        raise ValueError(
            f"communication task {task['id']!r} must define integer num_bytes"
        )
    if num_bytes <= 0 or num_bytes % 4:
        raise ValueError(
            f"communication task {task['id']!r} num_bytes must be a positive multiple of 4"
        )
    return num_bytes


def chain_to_comm_blocks(
    chain: list[dict[str, Any]], duration_unit_s: float = 1.0
) -> list[dict[str, Any]]:
    """Convert one chain without counting any compute span twice.

    Compute before the first communication becomes its producer window. A
    compute span after a communication becomes that communication's consumer
    overlap window; it is not repeated as the next communication's producer.
    """

    blocks: list[dict[str, Any]] = []
    prefix_compute_s = 0.0
    for task in chain:
        duration_s = float(task["duration"]) * duration_unit_s
        if task["kind"] == "compute":
            if blocks:
                blocks[-1]["consumer_compute_s"] += duration_s
            else:
                prefix_compute_s += duration_s
            continue

        blocks.append(
            {
                "id": len(blocks),
                "num_bytes": _message_bytes(task),
                "op": task.get("op", DEFAULT_OP),
                "producer_compute_s": prefix_compute_s,
                "consumer_compute_s": 0.0,
                "estimated_comm_s": duration_s,
            }
        )
        prefix_compute_s = 0.0

    if not blocks:
        raise ValueError("each chain must contain at least one communication")
    return blocks


def round_s(value: float, ndigits: int = 10) -> float:
    return round(value, ndigits)


def convert_document(data: dict[str, Any]) -> dict[str, Any]:
    metadata = data.get("metadata", {})
    duration_unit_s = float(metadata.get("duration_unit_s", 1.0))
    if duration_unit_s <= 0:
        raise ValueError("metadata.duration_unit_s must be positive")

    prepared = preprocess_zero_duration(data)
    chains = [merge_adjacent_compute(chain) for chain in build_chains(prepared)]
    jobs = []
    for job_index, chain in enumerate(chains):
        communications = chain_to_comm_blocks(chain, duration_unit_s)
        for communication in communications:
            for field in (
                "producer_compute_s",
                "consumer_compute_s",
                "estimated_comm_s",
            ):
                communication[field] = round_s(communication[field])
        jobs.append(
            {
                "job_id": f"job-{job_index}",
                "ranks": None,
                "communications": communications,
            }
        )

    if not jobs:
        raise ValueError("input must contain at least one chain")
    return {
        "name": data.get("id", "workload"),
        "seed": metadata.get("seed", DEFAULT_SEED),
        "jobs": jobs,
    }


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(f"Usage: {argv[0]} <input_dir> <output_dir>")
        return 1

    input_dir, output_dir = map(Path, argv[1:])
    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return 1
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in json_files:
        output = output_dir / path.name
        output.write_text(
            json.dumps(convert_document(load_json(path)), indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[ok] {path.name} -> {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
