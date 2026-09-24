"""Resolve repository benchmark paths retained by historical migration maps."""

from __future__ import annotations

import json
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def repository_path(path: str | Path) -> Path:
    """Resolve a CLI path relative to the repository, not the caller's cwd."""
    source = Path(path).expanduser()
    return source if source.is_absolute() else REPOSITORY_ROOT / source


def is_formal_experiment_input(path: str | Path, input_root: str | Path) -> bool:
    """Identify formal suite inputs while leaving the preserved smoke fixtures exempt."""
    source = repository_path(path).resolve(strict=False)
    root = Path(input_root).resolve(strict=False)
    return (source.is_relative_to(root)
            and not source.is_relative_to(root / "dag-semantics" / "smoke"))


def resolve_migrated_path(path: str | Path, migration_map: str | Path) -> Path:
    """Return the moved path when *path* is listed in a benchmark migration map."""
    source = repository_path(path)
    resolved_source = source.resolve(strict=False)
    if resolved_source.exists():
        return resolved_source

    candidates = [source.as_posix()]
    try:
        candidates.insert(0, resolved_source.relative_to(REPOSITORY_ROOT).as_posix())
    except ValueError:
        pass

    try:
        document = json.loads(Path(migration_map).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return source

    matches: list[tuple[int, str, str, bool]] = []
    for entry in document.get("entries", []):
        old = str(entry.get("old_path", "")).rstrip("/")
        new = str(entry.get("new_path", "")).rstrip("/")
        if not old or not new:
            continue
        is_prefix = bool(entry.get("directory", False))
        for candidate in candidates:
            if candidate == old or (is_prefix and candidate.startswith(old + "/")):
                matches.append((len(old), old, new, is_prefix))
    if not matches:
        return source

    _, old, new, is_prefix = max(matches)
    matched = next(candidate for candidate in candidates
                   if candidate == old or (is_prefix and candidate.startswith(old + "/")))
    suffix = matched[len(old):].lstrip("/")
    target = REPOSITORY_ROOT / new
    if suffix:
        target /= suffix
    return target if target.exists() else source
