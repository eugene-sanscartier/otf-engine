"""ETN file input/output helpers."""

from __future__ import annotations

from pathlib import Path

from .._core.io import format_floats, format_ints


def write_etn(potential, filename):
    """Write an ETN potential to an .etn file.

    The file structure is the one PairETN::read_file() expects. An untrained
    potential is written without its coefficients, as write_mtp writes one.
    """
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    l_count = len(potential.get_species_ranks())
    core_count = potential.get_core_count()

    with path.open("w", encoding="utf-8") as stream:
        stream.write("ETN\n")
        stream.write("version = 1.0.0\n")
        stream.write(f"potential_name = {potential.get_potential_name()}\n")
        stream.write(f"scaling = {potential.get_scaling():.17g}\n")
        stream.write(f"species_count = {potential.get_species_count()}\n")
        stream.write(f"radial_basis_type = {potential.get_radial_basis_type()}\n")
        stream.write(f"\tmin_dist = {potential.get_min_cutoff():.17g}\n")
        stream.write(f"\tmax_dist = {potential.get_max_cutoff():.17g}\n")
        stream.write(f"\tradial_basis_size = {potential.get_radial_basis_size()}\n")
        stream.write("species_ranks = {" + format_ints(potential.get_species_ranks()) + "}\n")
        stream.write("radial_funcs_counts = {" + format_ints(potential.get_radial_func_counts()) + "}\n")
        stream.write("radial_ranks = {" + format_ints(potential.get_radial_ranks()) + "}\n")
        stream.write(f"core_count = {core_count}\n")
        stream.write("bond_ranks\n")
        for ranks in potential.get_bond_ranks():
            stream.write("\t{" + format_ints(ranks) + "}\n")

        if not potential.is_trained():
            return

        stream.write("radial_coeffs\n")
        for l in range(l_count):
            stream.write("\t{" + format_floats(potential.get_radial_coeffs(l)) + "}\n")
        stream.write("species_pair_coeffs\n")
        for l in range(l_count):
            stream.write("\t{" + format_floats(potential.get_species_pair_coeffs(l)) + "}\n")
        stream.write("core_coeffs\n")
        for i in range(core_count):
            stream.write("\t{" + format_floats(potential.get_core_coeffs(i)) + "}\n")
        if core_count > 3:
            stream.write("core_coeffs_im\n")
            for i in range(1, core_count - 1):
                stream.write("\t{" + format_floats(potential.get_core_coeffs(i, imaginary=True)) + "}\n")
        stream.write("species_coeffs = {" + format_floats(potential.get_species_coeffs()) + "}\n")
