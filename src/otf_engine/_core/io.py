"""Potential files: reading one as its gradient class, and what the writers share: the first lines, the radial basis and the number formatting."""

from __future__ import annotations

import numpy


def read_potential(path: str):
    """The potential in the file *path*, as an ETN if the file's first line is ETN, else as an MTP."""
    with open(path, "rb") as f:    # the #MVS_v1.1 section is binary
        etn = f.readline().strip() == b"ETN"
    if etn:
        from .._etn import ETN    # _mtp and _etn import _core, so _core imports them only when called
        potential = ETN(path)
    else:
        from .._mtp import MTP
        potential = MTP(path)
    return potential


def write_header(stream, potential, model: str, version: str, with_species_count: bool) -> None:
    """The first lines of a *model* (MTP, ETN) file, as Pair::read_header reads them: version, name, scaling and, when asked, the species count."""
    stream.write(f"{model}\n")
    stream.write(f"version = {version}\n")
    stream.write(f"potential_name = {potential.get_potential_name()}\n")
    stream.write(f"scaling = {potential.get_scaling():.17g}\n")
    if with_species_count:
        stream.write(f"species_count = {potential.get_species_count()}\n")


def write_radial_basis(stream, potential) -> None:
    """The radial basis lines, as Pair::read_radial_basis reads them."""
    stream.write(f"radial_basis_type = {potential.get_radial_basis_type()}\n")
    stream.write(f"\tmin_dist = {potential.get_min_cutoff():.17g}\n")
    stream.write(f"\tmax_dist = {potential.get_max_cutoff():.17g}\n")
    stream.write(f"\tradial_basis_size = {potential.get_radial_basis_size()}\n")


def format_floats(values) -> str:
    """*values*, flattened in C order, comma-separated, each to 17 significant digits, so a file read back holds the same doubles."""
    return ", ".join(f"{float(value):.17g}" for value in numpy.ravel(values))


def format_ints(values) -> str:
    """*values*, flattened in C order, comma-separated."""
    return ", ".join(str(int(value)) for value in numpy.ravel(values))
