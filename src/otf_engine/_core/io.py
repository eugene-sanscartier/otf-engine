"""Number formatting shared by the potential file writers."""

from __future__ import annotations

import numpy


def format_floats(values) -> str:
    """*values*, flattened in C order, comma-separated, each to 17 significant digits, so a file read back holds the same doubles."""
    return ", ".join(f"{float(value):.17g}" for value in numpy.ravel(values))


def format_ints(values) -> str:
    """*values*, flattened in C order, comma-separated."""
    return ", ".join(str(int(value)) for value in numpy.ravel(values))
