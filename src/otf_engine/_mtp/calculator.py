"""
ASE Calculator wrapping the MTP potential.

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
from ase.calculators.calculator import Calculator, all_changes

from ._mtp import MTPTraining
from .neighbors import mtp_types, neighbors


class MTPCalculator(Calculator):
    """ASE Calculator for the Moment Tensor Potential.

    Parameters
    ----------
    filename : str
        Path to the .mtp potential file (version 1.1.0).
    """

    implemented_properties = ["energy", "free_energy", "energies", "forces", "stress"]

    def __init__(self, filename: str, **kwargs):
        super().__init__(**kwargs)
        self.potential = MTPTraining(filename)
        self.cutoff = self.potential.get_max_cutoff()

    def _atoms_to_types(self, atoms) -> ndarray:
        """Return the 0-indexed MTP type array for *atoms*."""
        return mtp_types(atoms)

    def neighbors(self, atoms):
        """Return the MTP neighbor list for *atoms*."""
        return neighbors(atoms, self.cutoff)

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        if properties is None:
            properties = self.implemented_properties

        want_virials = "stress" in properties
        want_eatom = "energies" in properties
        result = self.potential.compute(self.neighbors(atoms), compute_virials=want_virials, compute_eatom=want_eatom)

        energy = float(result["energy"])
        self.results["energy"] = energy
        self.results["free_energy"] = energy
        self.results["forces"] = numpy.array(result["forces"], dtype=float64)
        if want_eatom:
            self.results["energies"] = numpy.array(result["eatom"], dtype=float64)

        if want_virials:
            vol = atoms.get_volume()
            # ASE stress convention: Voigt order (xx,yy,zz,yz,xz,xy), positive = tensile
            # Our virials are (xx,yy,zz,xy,xz,yz), sign: virial = -stress * vol
            v = result["virials"]
            self.results["stress"] = numpy.array([-v[0], -v[1], -v[2], -v[5], -v[4], -v[3]], dtype=float64) / vol

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
