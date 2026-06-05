"""Cycle directory management for otf-mtp runs.

Each invocation of the CLI archives its inputs, outputs, and consumed dump
files into ./otf_cycles/cycle_N/, where N is the current highest cycle index.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

CYCLE_BASE = Path("./otf_cycles")
CYCLE_PREFIX = "cycle_"

_CYCLE_ARTIFACTS_MOVE = ("mlip_train.log", )
_CYCLE_ARTIFACTS_COPY = ("otf_state.json", )


_current: Path | None = None


def _last_cycle_number(base_dir: Path = CYCLE_BASE) -> int:
    """Return the highest cycle index found under *base_dir*, or -1 if none exist."""
    pattern = re.compile(rf"^{re.escape(CYCLE_PREFIX)}(\d+)$")
    indices = [int(m.group(1)) for p in base_dir.glob(f"{CYCLE_PREFIX}*") if p.is_dir() and (m := pattern.match(p.name))]
    return max(indices) if indices else -1


def current_cycle_dir() -> Path | None:
    """Return the active cycle directory, or None if next_cycle_dir() has not been called."""
    return _current


def next_cycle_dir(base_dir: Path = CYCLE_BASE) -> Path:
    """Create the next cycle directory, register it as current, and return it."""
    global _current
    _current = base_dir / f"{CYCLE_PREFIX}{_last_cycle_number(base_dir) + 1}"
    _current.mkdir(parents=True, exist_ok=True)
    return _current


def archive_cycle(cycle_dir: Path, potential: str, training_set: str, dump_files: list[str]) -> None:
    """Archive one OTF cycle's artifacts into *cycle_dir*.

    - Creates *cycle_dir*.
    - Copies *potential* and *training_set* as snapshots (input and post-run state).
    - Moves _CYCLE_ARTIFACTS_MOVE from cwd into *cycle_dir* (if they exist).
    - Copies _CYCLE_ARTIFACTS_COPY from cwd into *cycle_dir* (if they exist).
    - Moves each file in *dump_files* into *cycle_dir* (clears the dump dir for
      the next cycle).
    """
    cycle_dir.mkdir(parents=True, exist_ok=True)

    for src in (potential, training_set):
        p = Path(src)
        if p.exists():
            shutil.copy2(p, cycle_dir / p.name)

    for name in _CYCLE_ARTIFACTS_MOVE:
        src = Path(name)
        if src.exists():
            shutil.move(str(src), cycle_dir / name)

    for name in _CYCLE_ARTIFACTS_COPY:
        src = Path(name)
        if src.exists():
            shutil.copy2(src, cycle_dir / name)

    for dump in dump_files:
        p = Path(dump)
        if p.exists():
            shutil.move(str(p), cycle_dir / p.name)
