"""MTP file input/output helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def _format_floats(values: Iterable[float]) -> str:
    return ", ".join(f"{float(value):.17g}" for value in values)


def _format_ints(values: Iterable[int]) -> str:
    return ", ".join(str(int(value)) for value in values)


def write_mtp(potential, filename, template_filename=None):
    """Write an MTP potential to an .mtp file.

    The current implementation writes the canonical file structure expected by
    MTPPotential.read_file(). If template_filename is provided, it is accepted
    for API compatibility but not required.
    """
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    radial_basis_coeffs = potential.get_radial_basis_coeffs()
    alpha_index_basic = potential.get_alpha_index_basic()
    alpha_index_times = potential.get_alpha_index_times()
    alpha_moment_mapping = potential.get_alpha_moment_mapping()
    species_coeffs = potential.get_species_coeffs()
    linear_coeffs = potential.get_linear_coeffs()

    species_count = potential.get_species_count()
    radial_func_count = potential.get_radial_func_count()
    radial_basis_size = potential.get_radial_basis_size()

    with path.open("w", encoding="utf-8") as stream:
        stream.write("MTP\n")
        stream.write("version = 1.1.0\n")
        stream.write(f"potential_name = {potential.get_potential_name()}\n")
        stream.write(f"scaling = {potential.get_scaling():.17g}\n")
        stream.write(f"species_count = {species_count}\n")
        stream.write("potential_tag = \n")
        stream.write("radial_basis_type = RBChebyshev\n")
        stream.write(f"\tmin_dist = {potential.get_min_cutoff():.17g}\n")
        stream.write(f"\tmax_dist = {potential.get_max_cutoff():.17g}\n")
        stream.write(f"\tradial_basis_size = {radial_basis_size}\n")
        stream.write(f"\tradial_funcs_count = {radial_func_count}\n")
        stream.write("\tradial_coeffs\n")

        pair_count = species_count * species_count
        for pair_index in range(pair_count):
            i = pair_index // species_count
            j = pair_index % species_count
            stream.write(f"\t\t{i}-{j}\n")
            block = radial_basis_coeffs[pair_index]
            for row in range(radial_func_count):
                values = block[row]
                stream.write(f"\t\t\t{{{_format_floats(values)}}}\n")

        stream.write(f"alpha_moments_count = {potential.get_alpha_moment_count()}\n")
        stream.write(f"alpha_index_basic_count = {potential.get_alpha_index_basic_count()}\n")
        stream.write("alpha_index_basic = {" + ", ".join("{" + _format_ints(row) + "}" for row in alpha_index_basic) + "}\n")
        stream.write(f"alpha_index_times_count = {potential.get_alpha_index_times_count()}\n")
        stream.write("alpha_index_times = {" + ", ".join("{" + _format_ints(row) + "}" for row in alpha_index_times) + "}\n")
        stream.write(f"alpha_scalar_moments = {potential.get_alpha_scalar_count()}\n")
        stream.write("alpha_moment_mapping = {" + _format_ints(alpha_moment_mapping) + "}\n")
        stream.write("species_coeffs = {" + _format_floats(species_coeffs) + "}\n")
        stream.write("moment_coeffs = {" + _format_floats(linear_coeffs) + "}\n")
