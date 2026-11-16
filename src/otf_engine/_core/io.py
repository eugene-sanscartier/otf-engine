"""Potential files: reading one as its gradient class, and the number formatting the writers share."""

from __future__ import annotations

import numpy


def read_potential(path: str):
    """The potential in the file *path*, as a PairETNGrad if the file's first line is ETN, else as a PairMTPGrad."""
    with open(path, "rb") as f:    # the #MVS_v1.1 section is binary
        etn = f.readline().strip() == b"ETN"
    if etn:
        from .._etn import PairETNGrad    # _mtp and _etn import _core, so _core imports them only when called
        potential = PairETNGrad(path)
    else:
        from .._mtp import PairMTPGrad
        potential = PairMTPGrad(path)
    return potential


def format_floats(values) -> str:
    """*values*, flattened in C order, comma-separated, each to 17 significant digits, so a file read back holds the same doubles."""
    return ", ".join(f"{float(value):.17g}" for value in numpy.ravel(values))


def format_ints(values) -> str:
    """*values*, flattened in C order, comma-separated."""
    return ", ".join(str(int(value)) for value in numpy.ravel(values))
