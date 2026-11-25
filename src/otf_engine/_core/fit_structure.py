"""FitStructure, and the conversions between the virial and the ASE stress."""

from __future__ import annotations

import numpy
from numpy import float64

from . import _core
from .neighbors import neighbors, species_indices

# Permutation between the virial order xx yy zz xy xz yz and the ASE Voigt order xx yy zz yz xz xy; it is its own inverse.
_VOIGT_ORDER = [0, 1, 2, 5, 4, 3]


def virial2stress(virial, volume: float):
    """Return the ASE stress, -virial / volume in Voigt order, from the virial of a Pair."""
    stress = -numpy.asarray(virial, dtype=float64)[_VOIGT_ORDER] / volume
    return stress


def stress2virial(stress, volume: float):
    """Return the virial that the fits take, from the ASE stress."""
    virial = -numpy.asarray(stress, dtype=float64)[_VOIGT_ORDER] * volume
    return virial


class FitStructure(_core._ext.FitStructure):
    """Structure that the fits and error_table take: neighbor list, reference energy, forces, virial and number."""

    def __init__(self, atoms, cutoff: float, number: int, species=()):
        """Build the neighbor list of *atoms* within *cutoff* and take the reference values from *atoms.calc*.

        The energy, forces or virial is None where *atoms.calc* has no result for it.

        Args:
            number: Number that names the structure in the error tables.
            species: Element of each species index.
        """
        results = atoms.calc.results
        energy = float(results["energy"]) if "energy" in results else None
        forces = numpy.asarray(results["forces"], dtype=float64) if "forces" in results else None
        virial = stress2virial(results["stress"], atoms.get_volume()) if "stress" in results else None
        super().__init__(neighbors(atoms, cutoff, species_indices(atoms, species)), energy, forces, virial, number)
