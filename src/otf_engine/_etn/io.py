"""ETN file input/output helpers."""

from __future__ import annotations

from pathlib import Path

from .._core.io import format_floats, format_ints, write_header, write_radial_basis


def write_etn(potential, path):
    """Write an ETN potential to an .etn file.

    The file structure is the one PairETN::read_file() expects. An untrained
    potential is written without its coefficients, as write_mtp writes one, and
    without its species count if it has none.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    l_count = len(potential.get_species_ranks())
    core_count = potential.get_core_count()

    with path.open("w", encoding="utf-8") as stream:
        # an untrained potential may have no species count until training
        write_header(stream, potential, "ETN", "1.0.0", with_species_count=potential.get_species_count() > 0)
        write_radial_basis(stream, potential)
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
