"""Rebuild derived CSV/JSON summaries from an intact Phase 3 runs.jsonl batch."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from examples.jobpacer.experiments import gloo_phase3_batch as batch


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rebuild(batch_dir: Path, *, replace_derived: bool) -> dict[str, Any]:
    batch_dir = batch_dir.resolve(strict=True)
    manifest = json.loads((batch_dir / "manifest.json").read_text(encoding="utf-8"))
    runs_path = batch_dir / "runs.jsonl"
    latest = batch._latest_records(runs_path)
    config = manifest.get("config", {})
    planned = len(config.get("epochs", ())) * int(config.get("repeats", 0)) * len(config.get("arms", ()))
    if planned <= 0 or len(latest) != planned:
        raise ValueError(f"runs.jsonl is incomplete: expected {planned} latest run records, found {len(latest)}")
    raw_dir = (batch_dir / "raw").resolve(strict=True)
    records = sorted(latest.values(), key=lambda record: record.get("run_id", ""))
    loaded: list[tuple[dict[str, Any], dict[str, Any] | None, Path]] = []
    for record in records:
        path = Path(record.get("result_path", "")).resolve(strict=True)
        if path.parent != raw_dir or path.suffix != ".json":
            raise ValueError(f"result path escapes batch raw/: {path}")
        if record.get("result_sha256") != _digest(path):
            raise ValueError(f"raw result SHA-256 mismatch: {path}")
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid raw result JSON {path}: {exc}") from exc
        loaded.append((record, result, path))

    summary_rows = [batch._measure_row(record, result, path)
                    for record, result, path in loaded]
    job_rows = [row for record, result, _path in loaded
                for row in batch._job_rows(record, result)]
    task_rows = [row for record, result, _path in loaded
                 for row in batch._task_timing_rows(record, result)]
    mechanism_rows = [batch._mechanism_row(record, result) for record, result, _path in loaded]
    mechanism_summary = batch._mechanism_summary(mechanism_rows, records)
    coordinator_task_rows: list[dict[str, Any]] = []
    coordinator_event_rows: list[dict[str, Any]] = []
    for record, result, _path in loaded:
        task_diag, event_diag = batch._coordinator_diagnostic_rows(record, result)
        coordinator_task_rows.extend(task_diag)
        coordinator_event_rows.extend(event_diag)

    paired_rows = []
    baselines = tuple(dict.fromkeys((config.get("baseline", "new-static_fifo"),
                                    *config.get("secondary_baselines", ()))))
    for baseline in baselines:
        paired_rows.extend(batch._paired_rows(
            summary_rows, baseline=baseline,
            tie_threshold=float(config.get("tie_threshold", 0.01)),
            bootstrap_samples=int(config.get("bootstrap_samples", 2000)),
            bootstrap_seed=int(config.get("order_seed", 0)),
        ))

    tables_dir = batch_dir / "tables"
    tables_dir.mkdir(exist_ok=True)
    derived = {
        "summary.csv": summary_rows,
        "jobs.csv": job_rows,
        "task-timings.csv": task_rows,
        "mechanisms.csv": mechanism_rows,
        "mechanism-summary.csv": mechanism_summary,
        "paired-summary.csv": paired_rows,
        "coordinator-task-timings.csv": coordinator_task_rows,
        "coordinator-events.csv": coordinator_event_rows,
    }
    output_paths = [batch_dir / name for name in derived]
    output_paths += [tables_dir / name for name in derived]
    existing = [path for path in output_paths if path.exists()]
    if existing and not replace_derived:
        raise FileExistsError("derived outputs already exist; pass --replace-derived to regenerate only tables")
    for name, rows in derived.items():
        batch._write_csv(batch_dir / name, rows)
        batch._write_csv(tables_dir / name, rows)
    analysis = {
        "unit": "seed-block after within-seed repeat median",
        "baselines": baselines,
        "compute_jitter": config.get("compute_jitter"),
        "interpretation": "fixed-input system-noise control" if config.get("compute_jitter") == 0
        else "compute-duration perturbation across seed blocks",
        "tie_threshold": config.get("tie_threshold", 0.01),
        "bootstrap_samples": config.get("bootstrap_samples", 2000),
        "comparisons": paired_rows,
        "mechanism_summary": mechanism_summary,
        "failed_runs": sum(row["status"] != "ok" for row in summary_rows),
        "derived_tables_rebuilt": True,
        "raw_runs_sha256": _digest(runs_path),
        "analysis_code_sha256": _digest(Path(__file__).resolve()),
        "batch_runner_code_sha256": _digest(Path(batch.__file__).resolve()),
    }
    analysis_path = batch_dir / "analysis.json"
    if analysis_path.exists() and not replace_derived:
        raise FileExistsError("analysis.json exists; pass --replace-derived to regenerate derived analysis")
    analysis_path.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (tables_dir / "analysis.json").write_bytes(analysis_path.read_bytes())
    return {"batch_dir": str(batch_dir), "runs": len(records),
            "failed": analysis["failed_runs"], "derived_tables": len(derived)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--replace-derived", action="store_true",
                        help="replace CSV/analysis outputs only; never changes manifest, runs.jsonl or raw/")
    args = parser.parse_args(argv)
    print(json.dumps(rebuild(args.batch_dir, replace_derived=args.replace_derived), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
