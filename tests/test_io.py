"""The structures' and potentials' files, the neighbor lists, and the calculator's stress."""

import io
import shutil

import numpy
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from conftest import DATA, ETN_REFERENCE, MTP_REFERENCE, two_species, write_dump
from otf_engine.io_cfg import read_cfg, write_cfg


def named(structures, species):
    """Return copies of *structures* whose atoms carry elements in place of type_index."""
    copies = []
    for atoms in structures:
        copy = atoms.copy()
        copy.calc = atoms.calc
        copy.set_chemical_symbols([species[t] for t in atoms.arrays["type_index"]])
        del copy.arrays["type_index"]
        copies += [copy]
    return copies


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
    """A left-handed cell keeps the sign of its stress, and of the virial FitStructure takes, through write_cfg and read_cfg."""
    from otf_engine._core import FitStructure

    atoms = structures[1].copy()
    atoms.calc = structures[1].calc
    atoms.arrays["type_index"] = structures[1].arrays["type_index"]
    atoms.set_cell(atoms.cell[[1, 0, 2]])    # rows swapped: the same lattice, det < 0
    assert numpy.linalg.det(atoms.cell[:]) < 0
    back = round_trip([atoms])[0]
    numpy.testing.assert_allclose(back.calc.results["stress"], atoms.calc.results["stress"], atol=1e-6)
    numpy.testing.assert_allclose(FitStructure(back, 5.0, 0).virial, FitStructure(atoms, 5.0, 0).virial, atol=1e-5)


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


def test_fit_structure_virial():
    """FitStructure gives a cfg's PlusStress (xx yy zz yz xz xy in the file) as the virial (xx yy zz xy xz yz)."""
    from otf_engine._core import FitStructure

    text = (DATA / "structures.cfg").read_text()
    block = text.split("BEGIN_CFG")[2]
    plus_stress = numpy.array(block.split("PlusStress:")[1].splitlines()[1].split(), dtype=float)
    atoms = read_cfg(io.StringIO("BEGIN_CFG" + block))[0]
    virial = FitStructure(atoms, 5.0, 0).virial
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


@pytest.mark.parametrize("path", [MTP_REFERENCE / "potential-16.mtp", ETN_REFERENCE / "etn-3.etn"])
def test_potential_tag(path, tmp_path):
    """A potential's species survive write and read."""
    from otf_engine._core import read_potential

    potential = read_potential(str(path))
    assert potential.get_species() == []
    potential.set_species(["Si", "H", "Ni"])
    potential.write(str(tmp_path / path.name))
    assert read_potential(str(tmp_path / path.name)).get_species() == ["Si", "H", "Ni"]


def test_training_set_extxyz(structures, tmp_path):
    """write_training_set then read_training_set give back an extxyz's structures."""
    from otf_engine.training_set import read_training_set, write_training_set

    written = named(structures, ["Ni", "Si", "H"])
    write_training_set(tmp_path / "train.xyz", written[:2])
    write_training_set(tmp_path / "train.xyz", written[2:])
    back = read_training_set(tmp_path / "train.xyz")

    assert len(back) == len(written)
    for a, b in zip(written, back):
        assert b.get_chemical_symbols() == a.get_chemical_symbols()
        assert "type_index" not in b.arrays
        numpy.testing.assert_array_equal(b.cell[:], a.cell[:])
        numpy.testing.assert_allclose(b.positions, a.positions, atol=1e-8)
        assert b.calc.results["energy"] == a.calc.results["energy"]
        numpy.testing.assert_allclose(b.calc.results["forces"], a.calc.results["forces"], atol=1e-8)
        assert ("stress" in b.calc.results) == ("stress" in a.calc.results)
        if "stress" in a.calc.results:
            numpy.testing.assert_array_equal(b.calc.results["stress"], a.calc.results["stress"])


def test_species_indices():
    """species_indices gives each atom its species index."""
    from otf_engine._core import species_indices

    atoms = Atoms("NiSi2H", positions=numpy.zeros((4, 3)))
    assert species_indices(atoms, ["Si", "H", "Ni"]).tolist() == [2, 0, 0, 1]
    with pytest.raises(ValueError, match="H is not in species"):
        species_indices(atoms, ["Si", "Ni"])
    atoms.arrays["type"] = numpy.array([2, 1, 1, 3])
    assert species_indices(atoms).tolist() == [1, 0, 0, 2]
    atoms.arrays["type_index"] = numpy.array([3, 2, 1, 0])
    assert species_indices(atoms, ["Si", "H", "Ni"]).tolist() == [3, 2, 1, 0]


def test_train_writes_species(structures, tmp_path):
    """train writes the training set's elements into an untagged potential."""
    from otf_engine._core import read_potential
    from otf_engine.train import train

    training_set = named(structures, ["Ni", "Si", "H"])
    train(str(MTP_REFERENCE / "template-08.almtp"), training_set, str(tmp_path / "trained.almtp"), options={"iteration_limit": 0, "log": "none"}, ranks=1)
    first_appearance = list(dict.fromkeys(element for atoms in training_set for element in atoms.get_chemical_symbols()))
    assert read_potential(str(tmp_path / "trained.almtp")).get_species() == first_appearance


@pytest.mark.parametrize("lammps_species, element, species", [(["Si", "Ni"], False, ["Si", "Ni"]), (["Si", "Ni"], True, None)], ids=["species", "element column"])
def test_grade_dump_species(lammps_species, element, species, structures, candidates, tmp_path):
    """grade_dump indexes and grades a dump whose LAMMPS types run opposite to the potential's species."""
    from otf_engine._core.io import write_potential_tag
    from otf_engine.otf_cycle import grade_dump
    from otf_engine.selection import update_active_set

    shutil.copy(DATA / "two-species-08.mtp", tmp_path / "potential.almtp")
    write_potential_tag(str(tmp_path / "potential.almtp"), ["Ni", "Si"])
    update_active_set(str(tmp_path / "potential.almtp"), two_species(structures))
    frames = named(two_species(candidates), ["Ni", "Si"])
    write_dump(tmp_path / "in_order.lammps", frames, ["Ni", "Si"])
    write_dump(tmp_path / "dump.lammps", frames, lammps_species, element)

    reference = grade_dump(str(tmp_path / "in_order.lammps"), str(tmp_path / "potential.almtp"), ["Ni", "Si"])
    graded = grade_dump(str(tmp_path / "dump.lammps"), str(tmp_path / "potential.almtp"), species)
    for a, b, c in zip(two_species(candidates), reference, graded):
        numpy.testing.assert_array_equal(c.arrays["type_index"], a.arrays["type_index"])
        assert c.info["features"]["MV_grade"] == b.info["features"]["MV_grade"]


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
    """PairCalculator's stress, ASE's convention, matches the numerical derivative of the energy under strain."""
    from ase.calculators.fd import calculate_numerical_stress

    from otf_engine._core.calculator import PairCalculator
    from otf_engine._mtp import PairMTP

    atoms = structures[1].copy()
    atoms.arrays["type_index"] = structures[1].arrays["type_index"]
    atoms.calc = PairCalculator(PairMTP(str(MTP_REFERENCE / "potential-16.mtp")))
    numpy.testing.assert_allclose(atoms.get_stress(), calculate_numerical_stress(atoms, eps=1e-6), rtol=1e-5, atol=1e-8)
