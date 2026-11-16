"""The structures' and potentials' files, the neighbor lists, and the calculator's stress."""

import io

import numpy
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from conftest import DATA, MTP_REFERENCE
from otf_engine.io_cfg import read_cfg, write_cfg


def round_trip(structures, species=None):
    text = io.StringIO()
    write_cfg(text, structures)
    text.seek(0)
    return read_cfg(text, species)


def test_cfg_round_trip(structures):
    """write_cfg then read_cfg gives back the cell, species, positions, energy, forces and stress, to the 6 decimals written."""
    back = round_trip(structures)
    assert len(back) == len(structures)
    for a, b in zip(structures, back):
        numpy.testing.assert_array_equal(b.arrays["type_index"], a.arrays["type_index"])
        numpy.testing.assert_allclose(b.cell[:], a.cell[:], atol=5e-7)
        numpy.testing.assert_allclose(b.positions, a.positions, atol=5e-7)
        assert abs(b.calc.results["energy"] - a.calc.results["energy"]) <= 5e-7
        numpy.testing.assert_allclose(b.calc.results["forces"], a.calc.results["forces"], atol=5e-7)
        assert ("stress" in b.calc.results) == ("stress" in a.calc.results)
        if "stress" in a.calc.results:
            numpy.testing.assert_allclose(b.calc.results["stress"] * b.get_volume(), a.calc.results["stress"] * a.get_volume(), atol=5e-7)


def test_cfg_left_handed_stress(structures):
    """A left-handed cell keeps the sign of its stress, and of the virial Sample takes, through write_cfg and read_cfg."""
    from otf_engine._core import Sample

    atoms = structures[1].copy()
    atoms.calc = structures[1].calc
    atoms.arrays["type_index"] = structures[1].arrays["type_index"]
    atoms.set_cell(atoms.cell[[1, 0, 2]])    # rows swapped: the same lattice, det < 0
    assert numpy.linalg.det(atoms.cell[:]) < 0
    back = round_trip([atoms])[0]
    numpy.testing.assert_allclose(back.calc.results["stress"], atoms.calc.results["stress"], atol=1e-6)
    numpy.testing.assert_allclose(Sample(back, 5.0, 0).virial, Sample(atoms, 5.0, 0).virial, atol=1e-5)


def test_cfg_without_energy(structures):
    """A structure without an energy reads back without one, not with a zero energy to fit."""
    atoms = structures[1].copy()
    atoms.arrays["type_index"] = structures[1].arrays["type_index"]
    atoms.calc = SinglePointCalculator(atoms, forces=structures[1].calc.results["forces"])
    back = round_trip([atoms])[0]
    assert "energy" not in back.calc.results


def test_cfg_species_maps(structures):
    """read_cfg takes the species as a list, a dict by index or a dict by name, and sets the same symbols."""
    names = ["Ni", "Si", "H"]
    by_list = round_trip(structures, names)
    for species in ({0: "Ni", 1: "Si", 2: "H"}, {"Ni": 0, "Si": 1, "H": 2}):
        assert [a.get_chemical_symbols() for a in round_trip(structures, species)] == [a.get_chemical_symbols() for a in by_list]
    assert by_list[1].get_chemical_symbols() == [names[t] for t in structures[1].arrays["type_index"]]


def test_sample_virial():
    """Sample gives a cfg's PlusStress (xx yy zz yz xz xy in the file) as the virial (xx yy zz xy xz yz)."""
    from otf_engine._core import Sample

    text = (DATA / "structures.cfg").read_text()
    block = text.split("BEGIN_CFG")[2]
    plus_stress = numpy.array(block.split("PlusStress:")[1].splitlines()[1].split(), dtype=float)
    atoms = read_cfg(io.StringIO("BEGIN_CFG" + block))[0]
    virial = Sample(atoms, 5.0, 0).virial
    numpy.testing.assert_allclose(virial, plus_stress[[0, 1, 2, 5, 4, 3]], rtol=1e-12)


@pytest.mark.parametrize("name", ["potential-16.mtp", "template-08.almtp"])
def test_mtp_round_trip(name, tmp_path):
    """write_mtp then PairMTP gives back a trained potential's species and coefficients exactly, and an untrained one stays
    untrained, written without its species as mlip-3 writes it."""
    from otf_engine._mtp import MTP, write_mtp

    potential = MTP(str(MTP_REFERENCE / name))
    write_mtp(potential, str(tmp_path / "back.mtp"))
    back = MTP(str(tmp_path / "back.mtp"))
    assert back.is_trained() == potential.is_trained()
    assert (back.get_min_cutoff(), back.get_max_cutoff(), back.get_scaling(), back.get_radial_basis_size(), back.get_alpha_scalar_count()) == (potential.get_min_cutoff(), potential.get_max_cutoff(), potential.get_scaling(), potential.get_radial_basis_size(), potential.get_alpha_scalar_count())
    if potential.is_trained():
        assert back.get_species_count() == potential.get_species_count()
        numpy.testing.assert_array_equal(back.get_coeffs(), potential.get_coeffs())


@pytest.mark.parametrize("case", ["sheared", "short", "open"])
def test_neighbors_against_ase(case, structures):
    """neighbors lists ASE's displacements, images included: in a sheared cell, a cell shorter than the cutoff, and without periodicity."""
    from ase.neighborlist import neighbor_list

    from otf_engine._core.neighbors import neighbors

    atoms = structures[1].copy()
    atoms.arrays["type_index"] = structures[1].arrays["type_index"]
    if case == "short":
        atoms = atoms[:2]
        atoms.set_cell(atoms.cell[:] * 0.5, scale_atoms=True)
    if case == "open":
        atoms.pbc = False
    ours = numpy.asarray(neighbors(atoms, 5.0).displacements).reshape(-1, 3)
    theirs = neighbor_list("D", atoms, 5.0)
    assert len(ours) == len(theirs)
    by_rows = lambda D: D[numpy.lexsort(numpy.round(D, 9).T[::-1])]
    numpy.testing.assert_allclose(by_rows(ours), by_rows(theirs), atol=1e-12)


def test_calculator_stress(structures):
    """PotentialCalculator's stress, ASE's convention, matches the numerical derivative of the energy under strain."""
    from ase.calculators.fd import calculate_numerical_stress

    from otf_engine._core.calculator import PotentialCalculator
    from otf_engine._mtp import PairMTP

    atoms = structures[1].copy()
    atoms.arrays["type_index"] = structures[1].arrays["type_index"]
    atoms.calc = PotentialCalculator(PairMTP(str(MTP_REFERENCE / "potential-16.mtp")))
    numpy.testing.assert_allclose(atoms.get_stress(), calculate_numerical_stress(atoms, eps=1e-6), rtol=1e-5, atol=1e-8)
