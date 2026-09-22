"""One fitting sample: a structure's neighbor list and its reference values."""

from dataclasses import dataclass

import numpy
from numpy import float64, ndarray

from ._mtp import NeighList
from .neighbors import neighbors


@dataclass
class Sample:
    """Neighbor list plus the reference values to fit against.

    stress is in ASE Voigt order (xx, yy, zz, yz, xz, xy) and eV/A^3.
    """

    neighbors: NeighList
    n_atoms: int
    energy: float
    forces: ndarray | None = None
    stress: ndarray | None = None
    volume: float | None = None


def sample(atoms, cutoff: float) -> Sample:
    """Build the fitting sample for an ASE Atoms carrying reference values.

    Forces and stress are taken when the attached calculator has them.
    """
    energy = float(atoms.get_potential_energy())
    results = atoms.calc.results

    forces = numpy.asarray(results["forces"], dtype=float64) if "forces" in results else None
    stress = numpy.asarray(results["stress"], dtype=float64) if "stress" in results else None
    volume = float(atoms.get_volume()) if stress is not None else None

    return Sample(neighbors=neighbors(atoms, cutoff), n_atoms=len(atoms), energy=energy, forces=forces, stress=stress, volume=volume)
