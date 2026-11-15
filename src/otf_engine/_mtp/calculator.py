"""
ASE Calculator for an MTP.

Usage
-----
from ase.build import bulk
from otf_engine._mtp import MTPCalculator

atoms = bulk("Si", cubic=True)
calc  = MTPCalculator("path/to/Si.mtp")
atoms.calc = calc
print(atoms.get_potential_energy())
print(atoms.get_forces())
"""

import numpy
from numpy import float64, ndarray

from .._core import PotentialCalculator
from ._mtp import MTPTraining


class MTPCalculator(PotentialCalculator):
    """ASE Calculator for the Moment Tensor Potential.

    Parameters
    ----------
    filename : str
        Path to the .mtp potential file (version 1.1.0).
    """

    def __init__(self, filename: str, **kwargs):
        super().__init__(MTPTraining(filename), **kwargs)

    def get_basis_values(self, atoms) -> ndarray:
        """Evaluate MTP basis functions for each atom.

        Returns an array of shape ``(n_atoms, alpha_scalar_count)`` where row
        ``i`` contains the scalar moment tensor values for that atom's
        neighbourhood.  Dot-product with ``get_linear_coeffs()`` gives the
        per-atom energy contributions (after adding ``species_coeffs``).
        """
        return numpy.array(self.potential.eval_basis(self.neighbors(atoms)), dtype=float64)

    def eval_grad(self, atoms) -> ndarray:
        """Per-atom information vector for extrapolation grade computation.

        Returns an array of shape ``(n_atoms, coeff_count)`` where each row is
        the gradient of the site energy w.r.t. all MTP coefficients:
        ``[radial_grads | species_one_hot | basis_values]``.
        """
        return numpy.array(self.potential.eval_grad(self.neighbors(atoms)), dtype=float64)
