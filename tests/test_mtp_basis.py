"""MTP's built from a level or from basis functions, against mlip-4's MTP of the same level.

mtp_reference/mlip4_levels.npz holds, for every level on an RBCinf basis, mlip-4's energy, site energies, forces and
virial on two cells, at the coefficients of coeffs(). mlip-4 orders its params as PairMTP orders its coefficients,
radial, species then basis. Each array is compared relative to its largest entry.
"""

import subprocess

import numpy
import pytest
from ase import Atoms

from conftest import MTP_REFERENCE, StoredOutputs
from etn_potentials import compile_probe, parse_tensors
from otf_engine._core import neighbors

LEVELS = list(range(6, 30, 2))
SPECIES_COUNT = 3
MIN_DIST, MAX_DIST, RADIAL_BASIS_SIZE = 1.0, 5.0, 6
TOL = 1e-11
KEYS = ("energy", "eatom", "forces", "virial")


def build(level, species_count=SPECIES_COUNT):
    from otf_engine._mtp import MTP

    return MTP(level=level, species_count=species_count, radial_basis_type="RBCinf", min_dist=MIN_DIST, max_dist=MAX_DIST, radial_basis_size=RADIAL_BASIS_SIZE)


def coeffs(pot):
    """The coefficients the comparison uses, 0.1 sin(10 + k^2), spread and free of a random stream."""
    k = numpy.arange(pot.get_coeff_count(), dtype=numpy.float64)
    return 0.1 * numpy.sin(10 + k * k)


@pytest.fixture(scope="module")
def cells(structures):
    """Structure 1 and a pair 3.7 A apart, with their species."""
    pair = Atoms(numbers=[1, 2], positions=[[1.0, 1.0, 1.0], [4.7, 1.0, 1.0]], cell=9.0 * numpy.eye(3), pbc=True)
    return [(structures[1], numpy.asarray(structures[1].arrays["type_index"], dtype=numpy.int32)), (pair, numpy.array([0, 2], dtype=numpy.int32))]


def run_probe(probe, level, coeffs, cells, workdir):
    """mlip-4's results for its MTP of level with params coeffs on cells."""
    floats = lambda values: " ".join(repr(float(x)) for x in values)
    lines = [f"{level} {SPECIES_COUNT} {RADIAL_BASIS_SIZE} {MIN_DIST!r} {MAX_DIST!r}", str(len(coeffs)), floats(coeffs), str(len(cells))]
    for atoms, types in cells:
        lines += [str(len(atoms))] + [floats(row) for row in atoms.cell.array]
        lines += [f"{t} {floats(p)}" for t, p in zip(types, atoms.positions)]
    probe_file = workdir / "probe.txt"
    probe_file.write_text("\n".join(lines) + "\n")
    return parse_tensors(subprocess.run([str(probe), str(probe_file)], check=True, capture_output=True, text=True).stdout)


@pytest.fixture(scope="module")
def golden(recapture, mlip4_build, cells, tmp_path_factory):
    """mlip-4's results per level; under --recapture rerun first."""
    stored = StoredOutputs(MTP_REFERENCE / "mlip4_levels.npz", recapture)
    if recapture:
        stored.arrays.clear()
        workdir = tmp_path_factory.mktemp("mtp_levels")
        probe = compile_probe(mlip4_build, workdir, MTP_REFERENCE / "mlip4_mtp_probe.cpp")
        for level in LEVELS:
            for key, value in run_probe(probe, level, coeffs(build(level)), cells, workdir).items():
                stored[f"{level}/{key}"] = value
        stored.save()
    return stored


@pytest.mark.parametrize("level", LEVELS)
def test_level_against_mlip4(level, golden, cells):
    """The MTP of a level gives mlip-4's energy, site energies, forces and virial at the same coefficients."""
    pot = build(level)
    pot.set_coeffs(coeffs(pot))
    failures = []
    for s, (atoms, types) in enumerate(cells):
        result = pot.compute(neighbors(atoms, MAX_DIST, types=types))
        for key in KEYS:
            want, got = golden[f"{level}/s{s}/{key}"], numpy.asarray(result[key], dtype=numpy.float64)
            rel = numpy.abs(got - want).max() / numpy.abs(want).max()
            if rel > TOL:
                failures += [f"s{s}/{key}: {rel:.1e}"]
    assert not failures, failures


def test_basis_functions_give_level():
    """A level's basis functions, given as basis_functions in another order, build the level's MTP."""
    from otf_engine._mtp import MTP, level_basis_functions

    functions = level_basis_functions(16)[::-1]
    pot = MTP(basis_functions=functions, species_count=SPECIES_COUNT, radial_basis_type="RBCinf", min_dist=MIN_DIST, max_dist=MAX_DIST, radial_basis_size=RADIAL_BASIS_SIZE)
    level = build(16)
    for get in ("get_alpha_index_basic", "get_alpha_index_times", "get_alpha_moment_mapping", "get_coeffs"):
        numpy.testing.assert_array_equal(getattr(pot, get)(), getattr(level, get)(), err_msg=get)


def test_untrained_round_trip(tmp_path):
    """A built MTP is untrained, and write_mtp then MTP gives it back as an untrained file gives it."""
    from otf_engine._mtp import MTP, write_mtp

    built = build(10, species_count=0)
    assert not built.is_trained()
    write_mtp(built, tmp_path / "untrained.mtp")
    back = MTP(str(tmp_path / "untrained.mtp"))
    assert not back.is_trained()
    assert back.get_radial_func_count() == built.get_radial_func_count() == 3
    for get in ("get_alpha_index_basic", "get_alpha_index_times", "get_alpha_moment_mapping", "get_coeffs"):
        numpy.testing.assert_array_equal(getattr(back, get)(), getattr(built, get)(), err_msg=get)


@pytest.mark.parametrize("kwargs, message", [
    ({"level": 7}, "level 7 does not exist"),
    ({}, "either level or basis_functions"),
    ({"level": 6, "basis_functions": [[0, 0]]}, "either level or basis_functions"),
    ({"basis_functions": [[0, 1]]}, "a basis function is a scalar"),
    ({"basis_functions": [[0, 0, 1, 2, 1, 1]]}, "symmetric"),
    ({"basis_functions": [[0, 0, 0]]}, "3 entries"),
])
def test_invalid(kwargs, message):
    """A level outside the table, both or neither of level and basis_functions, and a malformed basis function raise."""
    from otf_engine._mtp import MTP

    with pytest.raises((RuntimeError, ValueError), match=message):
        MTP(min_dist=MIN_DIST, max_dist=MAX_DIST, radial_basis_size=RADIAL_BASIS_SIZE, **kwargs)
