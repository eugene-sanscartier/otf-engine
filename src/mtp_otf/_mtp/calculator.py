"""
ASE Calculator wrapping MTPPotential.

Usage
-----
from ase.build import bulk
from mtp import MTPCalculator

atoms = bulk("Si", cubic=True)
calc  = MTPCalculator("path/to/Si.mtp")
atoms.calc = calc
print(atoms.get_potential_energy())
print(atoms.get_forces())
"""

import numpy
from numpy import int32, float64, ndarray
from ase.calculators.calculator import Calculator, all_changes
from ase.neighborlist import neighbor_list

from ._mtp import MTPPotential


class MTPCalculator(Calculator):
    """ASE Calculator for the Moment Tensor Potential.

    Parameters
    ----------
    filename : str
        Path to the .mtp potential file (version 1.1.0).
    """

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, filename: str, **kwargs):
        super().__init__(**kwargs)
        self.potential = MTPPotential(filename)
        self.cutoff = self.potential.get_max_cutoff()

    def _atoms_to_types(self, atoms) -> ndarray:
        """Return 0-indexed MTP type array for *atoms*.

        Uses ``atoms.arrays["type_index"]`` directly when present; otherwise
        assigns indices by order of first appearance of chemical symbols.
        """
        if "type_index" in atoms.arrays:
            return numpy.asarray(atoms.arrays["type_index"], dtype=int32)
        seen: dict[str, int] = {}
        for sym in atoms.get_chemical_symbols():
            if sym not in seen:
                seen[sym] = len(seen)
        return numpy.array([seen[s] for s in atoms.get_chemical_symbols()], dtype=int32)

    def _build_neighbor_list(self, atoms):
        """Return (ilist, numneigh, firstneigh_flat, displacements) as numpy arrays.

        Uses ASE's neighbor_list with query ``"ijD"`` to obtain PBC-corrected
        displacement vectors ``D = r_j - r_i + S @ cell``.  Every pair is
        listed in both directions — equivalent to LAMMPS REQ_FULL.
        """
        i_arr, j_arr, D_arr = neighbor_list("ijD", atoms, self.cutoff)

        n_atoms = len(atoms)
        ilist = numpy.arange(n_atoms, dtype=int32)
        numneigh = numpy.bincount(i_arr, minlength=n_atoms).astype(int32)

        # Sort so neighbor blocks are contiguous for each central atom
        order = numpy.argsort(i_arr, kind="stable")
        firstneigh = j_arr[order].astype(int32)
        displacements = numpy.ascontiguousarray(D_arr[order], dtype=float64)

        return ilist, numneigh, firstneigh, displacements

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        # if atoms is None:
        #     raise ValueError("MTPCalculator.calculate() requires an ASE Atoms object")

        if properties is None:
            properties = self.implemented_properties

        types = self._atoms_to_types(atoms)
        ilist, numneigh, firstneigh, displacements = self._build_neighbor_list(atoms)

        compute_virials = "stress" in properties
        result = self.potential.compute(types, ilist, numneigh, firstneigh, displacements, compute_virials=compute_virials, compute_eatom=False)

        self.results["energy"] = float(result["energy"])
        self.results["forces"] = numpy.array(result["forces"], dtype=float64)

        if compute_virials:
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

        Parameters
        ----------
        atoms : ase.Atoms

        Returns
        -------
        ndarray, shape (n_atoms, alpha_scalar_count)
        """
        types = self._atoms_to_types(atoms)
        ilist, numneigh, firstneigh, displacements = self._build_neighbor_list(atoms)
        return numpy.array(self.potential.eval_basis(types, ilist, numneigh, firstneigh, displacements), dtype=float64)

    def eval_grad(self, atoms) -> ndarray:
        """Per-atom information vector for extrapolation grade computation.

        Returns an array of shape ``(n_atoms, coeff_count)`` where each row is
        the gradient of the site energy w.r.t. all MTP coefficients:
        ``[radial_grads | species_one_hot | basis_values]``.
        Used with the MaxVol invA matrix to compute extrapolation grades.

        Parameters
        ----------
        atoms : ase.Atoms

        Returns
        -------
        ndarray, shape (n_atoms, coeff_count)
            coeff_count = radial_coeff_count + species_count + alpha_scalar_count
        """
        types = self._atoms_to_types(atoms)
        ilist, numneigh, firstneigh, displacements = self._build_neighbor_list(atoms)
        return numpy.array(self.potential.eval_grad(types, ilist, numneigh, firstneigh, displacements), dtype=float64)
