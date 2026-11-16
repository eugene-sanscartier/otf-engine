"""MTP file input/output helpers."""

from __future__ import annotations

from pathlib import Path

from .._core.io import format_floats, format_ints, write_header, write_radial_basis


def write_mtp(potential, path):
    """Write an MTP potential to an .mtp file.

    The file structure is the one PairMTP::read_file() expects. An untrained
    potential is written without its species count and coefficients, as mlip-3 writes it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    radial_basis_coeffs = potential.get_radial_basis_coeffs()
    alpha_index_basic = potential.get_alpha_index_basic()
    alpha_index_times = potential.get_alpha_index_times()
    alpha_moment_mapping = potential.get_alpha_moment_mapping()
    species_coeffs = potential.get_species_coeffs()
    linear_coeffs = potential.get_linear_coeffs()

    species_count = potential.get_species_count()
    radial_func_count = potential.get_radial_func_count()
    trained = potential.is_trained()

    with path.open("w", encoding="utf-8") as stream:
        write_header(stream, potential, "MTP", "1.1.0", with_species_count=trained)
        stream.write("potential_tag = \n")
        write_radial_basis(stream, potential)
        stream.write(f"\tradial_funcs_count = {radial_func_count}\n")
        stream.write("\tradial_coeffs\n")

        pair_count = species_count * species_count if trained else 0
        for pair_index in range(pair_count):
            i = pair_index // species_count
            j = pair_index % species_count
            stream.write(f"\t\t{i}-{j}\n")
            block = radial_basis_coeffs[pair_index]
            for row in range(radial_func_count):
                values = block[row]
                stream.write(f"\t\t\t{{{format_floats(values)}}}\n")

        stream.write(f"alpha_moments_count = {potential.get_alpha_moment_count()}\n")
        stream.write(f"alpha_index_basic_count = {potential.get_alpha_index_basic_count()}\n")
        stream.write("alpha_index_basic = {" + ", ".join("{" + format_ints(row) + "}" for row in alpha_index_basic) + "}\n")
        stream.write(f"alpha_index_times_count = {potential.get_alpha_index_times_count()}\n")
        stream.write("alpha_index_times = {" + ", ".join("{" + format_ints(row) + "}" for row in alpha_index_times) + "}\n")
        stream.write(f"alpha_scalar_moments = {potential.get_alpha_scalar_count()}\n")
        stream.write("alpha_moment_mapping = {" + format_ints(alpha_moment_mapping) + "}\n")
        if trained:
            stream.write("species_coeffs = {" + format_floats(species_coeffs) + "}\n")
            stream.write("moment_coeffs = {" + format_floats(linear_coeffs) + "}\n")
