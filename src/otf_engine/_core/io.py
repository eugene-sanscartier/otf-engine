"""Read potential files, and write the header, radial basis and numbers that the MTP and ETN files share."""

from __future__ import annotations

import numpy


def potential_class(path: str):
    """Return the class of the potential in *path*: ETN if the first line is ETN, else MTP."""
    with open(path, "rb") as f:    # the #MVS_v1.1 section is binary
        etn = f.readline().strip() == b"ETN"
    if etn:
        from .._etn import ETN
        cls = ETN
    else:
        from .._mtp import MTP
        cls = MTP
    return cls


def read_potential(path: str):
    """Return the potential in *path*, as an ETN or an MTP."""
    potential = potential_class(path)(path)
    return potential


def write_header(stream, potential, model: str, version: str, with_species_count: bool) -> None:
    """Write the header of a *model* file, MTP or ETN."""
    stream.write(f"{model}\n")
    stream.write(f"version = {version}\n")
    stream.write(f"potential_name = {potential.get_potential_name()}\n")
    stream.write(f"scaling = {potential.get_scaling():.17g}\n")
    if with_species_count:
        stream.write(f"species_count = {potential.get_species_count()}\n")
    stream.write(f"potential_tag = {' '.join(potential.get_species())}\n")


def write_potential_tag(path: str, species: list) -> None:
    """Set the potential_tag line of the potential file *path* to *species*."""
    with open(path, "rb") as f:    # the #MVS_v1.1 section is binary
        data = f.read()
    if b"\npotential_tag" in data:
        start = data.index(b"\npotential_tag") + 1
        end = data.index(b"\n", start) + 1
    else:
        start = end = data.index(b"\nradial_basis_type") + 1
    data = data[:start] + f"potential_tag = {' '.join(species)}\n".encode() + data[end:]
    with open(path, "wb") as f:
        f.write(data)


def write_radial_basis(stream, potential) -> None:
    """Write the radial basis lines that Pair::read_radial_basis reads."""
    stream.write(f"radial_basis_type = {potential.get_radial_basis_type()}\n")
    stream.write(f"\tmin_dist = {potential.get_min_cutoff():.17g}\n")
    stream.write(f"\tmax_dist = {potential.get_max_cutoff():.17g}\n")
    stream.write(f"\tradial_basis_size = {potential.get_radial_basis_size()}\n")


def format_floats(values) -> str:
    """Flatten *values* in C order and join them with commas, each to 17 significant digits."""
    return ", ".join(f"{float(value):.17g}" for value in numpy.ravel(values))


def format_ints(values) -> str:
    """Flatten *values* in C order and join them with commas."""
    return ", ".join(str(int(value)) for value in numpy.ravel(values))
