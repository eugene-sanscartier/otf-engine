"""Sample: a structure to fit, built from an ASE structure and its calculator's results."""

from __future__ import annotations

import numpy
from numpy import float64

from . import _core
from .neighbors import neighbors


class Sample(_core._ext.Sample):
    """A structure to fit, which the trainers and error_table take: its neighbor list, its reference energy, forces and virial, and its number."""

    def __init__(self, atoms, cutoff: float, number: int):
        """*atoms*'s neighbor list at *cutoff*, and the energy, forces and stress of *atoms.calc*, None where absent; the error tables name it by *number*."""
        results = atoms.calc.results
        energy = float(results["energy"]) if "energy" in results else None
        forces = numpy.asarray(results["forces"], dtype=float64) if "forces" in results else None
        virial = None
        if "stress" in results:
            # ASE's stress is -virial / volume, in Voigt order xx yy zz yz xz xy; the trainers take mlip-3's PlusStress, xx yy zz xy xz yz
            v = -numpy.asarray(results["stress"], dtype=float64) * atoms.get_volume()
            virial = v[[0, 1, 2, 5, 4, 3]]
        super().__init__(neighbors(atoms, cutoff), energy, forces, virial, number)
