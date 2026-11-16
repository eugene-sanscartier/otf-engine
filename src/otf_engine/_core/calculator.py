"""An ASE Calculator wrapping a Pair: an MTP or an ETN, plain or with its coefficient gradients (MTP, ETN)."""

import numpy
from numpy import float64
from ase.calculators.calculator import Calculator, all_changes

from .neighbors import neighbors
from .fit_structure import virial2stress


class PairCalculator(Calculator):
    """ASE Calculator for a Pair.

    Parameters
    ----------
    potential
        The potential object, with compute(neighbors) and get_max_cutoff().
    """

    implemented_properties = ["energy", "free_energy", "energies", "forces", "stress"]

    def __init__(self, potential, **kwargs):
        super().__init__(**kwargs)
        self.potential = potential
        self.cutoff = self.potential.get_max_cutoff()

    def neighbors(self, atoms):
        """Return the neighbor list for *atoms*."""
        return neighbors(atoms, self.cutoff)

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        if properties is None:
            properties = self.implemented_properties

        result = self.potential.compute(self.neighbors(atoms))

        energy = float(result["energy"])
        self.results["energy"] = energy
        self.results["free_energy"] = energy
        self.results["forces"] = numpy.array(result["forces"], dtype=float64)
        if "energies" in properties:
            self.results["energies"] = numpy.array(result["eatom"], dtype=float64)

        if "stress" in properties:
            self.results["stress"] = virial2stress(result["virial"], atoms.get_volume())
