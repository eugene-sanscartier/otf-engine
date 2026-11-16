"""The ETN test potentials, their conversion to mlip-4's params vector, and the mlip-4 probes' compilation."""

import re
import subprocess
from pathlib import Path

import numpy

from conftest import ETN_REFERENCE

MIN_DIST, MAX_DIST, RADIAL_BASIS_SIZE = 1.0, 5.0, 6
# name: species count, species ranks, radial function counts, radial ranks (per l of the input), bond ranks
POTENTIALS = {
    "etn-1": (3, [2, 1], [6, 4], [3, 2], [[1], [1]]),
    "etn-2": (3, [2, 1], [6, 4], [3, 2], [[1], [2, 2, 1, 1], [1]]),
    "etn-3": (3, [2, 2, 1], [6, 4, 3], [3, 2, 2], [[1], [3, 2, 1], [3, 2, 1], [1]]),
    "etn-4": (3, [2, 0, 1, 1, 1], [6, 4, 3, 2, 2], [3, 2, 0, 1, 1], [[1], [2, 0, 1, 1, 1], [2, 1, 1, 1, 1], [2, 0, 1, 1, 1], [1]]),
    "etn-5": (3, [2, 2, 1], [6, 4, 3], [3, 2, 2], [[1], [2, 2, 1], [2, 2, 1], [2, 2, 1], [2, 2, 1], [1]]),
    "etn-3-single": (1, [2, 1, 1], [6, 4, 3], [3, 2, 1], [[1], [2, 2, 1], [2, 2, 1], [1]]),
}


def untrained_etn(name, species_count=None):
    """The untrained ETN of a test potential, built from its shape, with its species count unless one is given."""
    from otf_engine._etn import ETN

    table_species_count, species_ranks, radial_funcs_counts, radial_ranks, bond_ranks = POTENTIALS[name]
    count = table_species_count if species_count is None else species_count
    return ETN(species_ranks=species_ranks, radial_funcs_counts=radial_funcs_counts, radial_ranks=radial_ranks, bond_ranks=bond_ranks, species_count=count, min_dist=MIN_DIST, max_dist=MAX_DIST, radial_basis_size=RADIAL_BASIS_SIZE)


def coeff_blocks(pot):
    """The coefficient blocks of pot as (kind, argument), in mlip-4's params order."""
    l_count, core_count = len(pot.get_species_ranks()), pot.get_core_count()
    blocks = [("radial", l) for l in range(l_count)] + [("species_pair", l) for l in range(l_count)]
    blocks += [("core", (i, False)) for i in range(core_count)]
    blocks += [("core", (i, True)) for i in range(1, core_count - 1) if core_count > 3]
    return blocks + [("species", None)]


def get_block(pot, kind, arg):
    """A coefficient block of pot, flat in mlip-4's params order."""
    block = {"radial": lambda: pot.get_radial_coeffs(arg), "species_pair": lambda: pot.get_species_pair_coeffs(arg),
             "core": lambda: pot.get_core_coeffs(*arg), "species": pot.get_species_coeffs}[kind]()
    return numpy.ravel(block)


def params(pot):
    """The coefficients of pot as mlip-4's params vector."""
    return numpy.concatenate([get_block(pot, kind, arg) for kind, arg in coeff_blocks(pot)])


def set_params(pot, values):
    """Split mlip-4's params vector into pot's coefficient blocks."""
    offset = 0
    for kind, arg in coeff_blocks(pot):
        size = len(get_block(pot, kind, arg))
        block = values[offset : offset + size]
        offset += size
        if kind == "radial":
            pot.set_radial_coeffs(arg, block)
        elif kind == "species_pair":
            pot.set_species_pair_coeffs(arg, block)
        elif kind == "core":
            pot.set_core_coeffs(arg[0], block, imaginary=arg[1])
        else:
            pot.set_species_coeffs(block)


def species_of(pot, types):
    """The species the potential sees: every atom is species 0 for a single-species potential."""
    return types if pot.get_species_count() > 1 else numpy.zeros_like(types)


def compile_probe(build_dir, workdir, source):
    """Compile a probe against mlip-4's build, with the include flags of its ETN module."""
    flags = (Path(build_dir) / "src/python/CMakeFiles/mlip_etn.dir/flags.make").read_text()
    includes = re.search(r"^CXX_INCLUDES = (.*)$", flags, re.M).group(1).split()
    probe = Path(workdir) / Path(source).stem
    subprocess.run(["g++", "-O2", "-std=gnu++17", *includes, str(source), str(Path(build_dir) / "lib_mlip_4_interface.a"), "-lpthread", "-ldl", "-o", str(probe)], check=True)
    return probe


def parse_tensors(printed):
    """The "NAME SHAPE... : VALUES" lines of an mlip-4 probe, as {NAME: array}."""
    tensors = {}
    for line in printed.splitlines():
        head, values = line.split(" :")
        name, *shape = head.split()
        tensors[name] = numpy.array([float(x) for x in values.split()]).reshape([int(n) for n in shape])
    return tensors


def reference_potential(name):
    return ETN_REFERENCE / f"{name}.etn"
