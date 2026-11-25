"""Create and archive the cycle directories of otf-engine runs.

Each OTF cycle has the directory ./otf_cycles/cycle_N/, N one above the highest
existing index. The cycle writes its log to otf.log there, archives its inputs,
outputs and consumed dump files there, and records its outcome in the file status.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

CYCLE_BASE = Path("./otf_cycles")
CYCLE_PREFIX = "cycle_"
STATUS_FILE = "status"
LOG_FILE = "otf.log"
LOG_FORMAT = "%(levelname)s %(module)s:%(funcName)s: %(message)s"
TRAIN_LOG = "mlip_train.log"

_CYCLE_ARTIFACTS_MOVE = (TRAIN_LOG, )
_CYCLE_ARTIFACTS_COPY = ("otf_state.json", )


def _cycle_indices(base_dir: Path = CYCLE_BASE) -> list[int]:
    """Return the indices of every cycle directory under *base_dir*."""
    pattern = re.compile(rf"^{re.escape(CYCLE_PREFIX)}(\d+)$")
    return [int(m.group(1)) for p in base_dir.glob(f"{CYCLE_PREFIX}*") if p.is_dir() and (m := pattern.match(p.name))]


def _last_cycle_number(base_dir: Path = CYCLE_BASE) -> int:
    """Return the highest cycle index under *base_dir*, or -1 if there is none."""
    indices = _cycle_indices(base_dir)
    return max(indices) if indices else -1


def last_successful_cycle_dir(base_dir: Path = CYCLE_BASE) -> Path | None:
    """Return the highest-numbered cycle directory whose status is ok, or None."""
    for index in sorted(_cycle_indices(base_dir), reverse=True):
        cycle_dir = base_dir / f"{CYCLE_PREFIX}{index}"
        status = cycle_dir / STATUS_FILE
        if status.exists() and status.read_text().strip() == "ok": return cycle_dir
    return None


def next_cycle_dir(base_dir: Path = CYCLE_BASE) -> Path:
    """Create the next cycle directory and return it."""
    cycle_dir = base_dir / f"{CYCLE_PREFIX}{_last_cycle_number(base_dir) + 1}"
    cycle_dir.mkdir(parents=True, exist_ok=True)
    return cycle_dir


def archive_cycle(cycle_dir: Path, potential_path: str, training_set_path: str, dump_files: list[str], ok: bool) -> None:
    """Archive the files of one OTF cycle into *cycle_dir* and record whether the cycle succeeded.

    - Create *cycle_dir*.
    - Copy *potential_path* and *training_set_path*.
    - Move the files of _CYCLE_ARTIFACTS_MOVE that exist from the working directory into *cycle_dir*.
    - Copy the files of _CYCLE_ARTIFACTS_COPY that exist from the working directory into *cycle_dir*.
    - Copy each file of *dump_files* into *cycle_dir* and truncate the original in place.
      Skip the entries that are not files.
    - Write STATUS_FILE last.
    """
    cycle_dir.mkdir(parents=True, exist_ok=True)

    for src in (potential_path, training_set_path):
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
        if p.is_file():
            shutil.copy2(p, cycle_dir / p.name)
            with p.open("r+b") as handle:
                handle.truncate(0)

    (cycle_dir / STATUS_FILE).write_text("ok\n" if ok else "failed\n")
