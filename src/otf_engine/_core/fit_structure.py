"""FitStructure: a structure to fit, built from an ASE structure and its calculator's results."""

from __future__ import annotations

import numpy
from numpy import float64

from . import _core
from .neighbors import neighbors

# Between the virial's order, mlip-3's PlusStress xx yy zz xy xz yz, and ASE's Voigt order xx yy zz yz xz xy; its own inverse
_VOIGT_ORDER = [0, 1, 2, 5, 4, 3]


def stress_from_virial(virial, volume: float):
    """ASE's stress, -virial / volume in Voigt order, from the virial a Pair computes."""
    stress = -numpy.asarray(virial, dtype=float64)[_VOIGT_ORDER] / volume
    return stress


def virial_from_stress(stress, volume: float):
    """The virial the fits take, from ASE's stress."""
    virial = -numpy.asarray(stress, dtype=float64)[_VOIGT_ORDER] * volume
    return virial


class FitStructure(_core._ext.FitStructure):
    """A structure to fit, which the fits and error_table take: its neighbor list, its reference energy, forces and virial, and its number."""

    def __init__(self, atoms, cutoff: float, number: int):
        """*atoms*'s neighbor list at *cutoff*, and the energy, forces and stress of *atoms.calc*, None where absent; the error tables name it by *number*."""
        results = atoms.calc.results
        energy = float(results["energy"]) if "energy" in results else None
        forces = numpy.asarray(results["forces"], dtype=float64) if "forces" in results else None
        virial = virial_from_stress(results["stress"], atoms.get_volume()) if "stress" in results else None
        super().__init__(neighbors(atoms, cutoff), energy, forces, virial, number)
