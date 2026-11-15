"""Number formatting shared by the potential file writers."""

from __future__ import annotations

from typing import Iterable


def format_floats(values: Iterable[float]) -> str:
    """*values* comma-separated, each to 17 significant digits, so a file read back holds the same doubles."""
    return ", ".join(f"{float(value):.17g}" for value in values)


def format_ints(values: Iterable[int]) -> str:
    """*values* comma-separated."""
    return ", ".join(str(int(value)) for value in values)
