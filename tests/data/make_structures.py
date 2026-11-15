"""Writes the synthetic structures of the tests, labelled by the level-16 teacher potential with its species energies left out.

structures.cfg: 6 sheared periodic cells of 8 to 12 atoms of species 0, 1 and 2, the first without stress.
candidates.cfg: 7 cells to grade and select against the first: structures 1 and 4 rattled by 0.2 A, 2 by 0.01 A and 5
by 0.002 A, 3 as it is, and two new.
two-species-08.mtp: a trained 2-species potential, to which a training set with species 2 adds one.

    python tests/data/make_structures.py
"""

from pathlib import Path

import numpy
from ase import Atoms

from otf_engine._core.calculator import PotentialCalculator
from otf_engine._mtp import PairMTP, write_mtp
from otf_engine.io_cfg import write_cfg
from otf_engine.train import fit, training_structure

DATA = Path(__file__).resolve().parent
TEACHER = DATA.parent / "mtp_reference" / "potential-16.mtp"
DENSITY = 0.075    # atoms per cubic angstrom, NiSi's
MIN_DISTANCE = 1.9


def random_cell(rng, n_atoms, species):
    """A sheared cell of n_atoms at DENSITY, no two atoms (images included) closer than MIN_DISTANCE."""
    side = (n_atoms / DENSITY) ** (1 / 3)
    cell = side * (numpy.eye(3) + rng.uniform(-0.05, 0.05, (3, 3)))
    atoms = Atoms(numbers=numpy.ones(n_atoms, dtype=int), cell=cell, pbc=True)

    positions = []
    while len(positions) < n_atoms:
        trial = rng.uniform(0, 1, 3) @ cell
        atoms.positions[: len(positions) + 1] = numpy.array(positions + [trial])
        distances = atoms[: len(positions) + 1].get_distances(len(positions), range(len(positions)), mic=True) if positions else []
        if all(d >= MIN_DISTANCE for d in distances):
            positions += [trial]
    atoms.positions = numpy.array(positions)
    atoms.arrays["type_index"] = numpy.asarray(species, dtype=numpy.int32)
    atoms.numbers = atoms.arrays["type_index"] + 1
    return atoms


def labelled(atoms, teacher, stress=True):
    """atoms with the teacher's forces, stress and energy, the energy less the teacher's species energies: a cohesive
    energy of a few eV per atom, which an untrained potential starts nearer to than the reference code's absolute one."""
    atoms.calc = PotentialCalculator(teacher)
    properties = ["energy", "forces"] + (["stress"] if stress else [])
    atoms.calc.calculate(atoms, properties)
    results = {key: atoms.calc.results[key] for key in properties}
    results["energy"] -= numpy.sum(numpy.asarray(teacher.get_species_coeffs())[atoms.arrays["type_index"]])
    atoms.calc.results = results
    return atoms


def main():
    rng = numpy.random.default_rng(20261115)
    teacher = PairMTP(str(TEACHER))

    sizes = [8, 9, 10, 11, 12, 12]
    structures = []
    for k, n in enumerate(sizes):
        species = rng.permutation(numpy.arange(n) % 3)
        structures += [labelled(random_cell(rng, n, species), teacher, stress=k != 0)]

    candidates = []
    for k, amplitude in ((1, 0.2), (4, 0.2), (2, 0.01), (5, 0.002), (3, 0.0)):
        rattled = structures[k].copy()
        rattled.arrays["type_index"] = structures[k].arrays["type_index"].copy()
        rattled.positions += rng.normal(0, amplitude, rattled.positions.shape)
        candidates += [labelled(rattled, teacher)]
    for n in (10, 12):
        candidates += [labelled(random_cell(rng, n, rng.permutation(numpy.arange(n) % 3)), teacher)]

    with open(DATA / "structures.cfg", "w") as f:
        write_cfg(f, structures)
    with open(DATA / "candidates.cfg", "w") as f:
        write_cfg(f, candidates)

    # the level-8 template with 4 radial basis functions, 43 coefficients for 2 species, fewer than the structures' atoms so
    # that their neighborhoods fill an active set, fitted for 10 steps to the structures with species 2 renamed 1
    two_species = []
    for atoms in structures:
        renamed = atoms.copy()
        renamed.calc = atoms.calc
        renamed.arrays["type_index"] = numpy.minimum(atoms.arrays["type_index"], 1)
        two_species += [renamed]
    template = DATA / "template-08-r4.almtp"
    template.write_text((DATA.parent / "mtp_reference" / "template-08.almtp").read_text().replace("radial_basis_size = 8", "radial_basis_size = 4"))
    potential = fit(str(template), [training_structure(atoms, 5.0, k) for k, atoms in enumerate(two_species)], {"iteration_limit": "10", "skip_preinit": "true", "random_seed": "1", "log": "none"}, ranks=1, balance_species=False)
    template.unlink()
    write_mtp(potential, str(DATA / "two-species-08.mtp"))


if __name__ == "__main__":
    main()
