"""An ASE Calculator wrapping a Pair: an MTP or an ETN, plain or with its coefficient gradients (PairMTPGrad, PairETNGrad)."""

import numpy
from numpy import float64
from ase.calculators.calculator import Calculator, all_changes

from .neighbors import neighbors


class PotentialCalculator(Calculator):
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
            vol = atoms.get_volume()
            # ASE stress convention: Voigt order (xx,yy,zz,yz,xz,xy), positive = tensile
            # Our virials are (xx,yy,zz,xy,xz,yz), sign: virial = -stress * vol
            v = result["virials"]
            self.results["stress"] = numpy.array([-v[0], -v[1], -v[2], -v[5], -v[4], -v[3]], dtype=float64) / vol
