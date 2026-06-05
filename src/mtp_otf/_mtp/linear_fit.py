"""
Weighted least-squares fitter for MTP linear coefficients.

Fits `linear_coeffs` and `species_coeffs` simultaneously by solving:

    [ w_E * A_E ]           [ w_E * b_E   ]
    [ w_F * A_F ] @ x    =  [ w_F * F_DFT ]
    [ w_S * A_S ]           [ w_S * S_DFT ]

where  x = [linear_coeffs | species_coeffs].
"""

import numpy as np
from scipy.linalg import lstsq

from .design_matrix import build_design_matrix


class LinearFitter:
    """Fit MTP linear coefficients (and species reference energies) by
    weighted least squares.

    Parameters
    ----------
    pot : MTPPotential
        Potential whose linear_coeffs / species_coeffs will be updated in-place
        after calling fit().
    weight_energy / weight_forces / weight_stress : float
        Relative weights for the three observable types.
    include_forces / include_stress : bool
        Whether to include force / stress rows in the system.
    """

    def __init__(self, pot, *, weight_energy=1.0, weight_forces=0.01, weight_stress=0.001, weight_scaling=1, include_forces=True, include_stress=True):
        self.pot = pot
        self.weight_energy = weight_energy
        self.weight_forces = weight_forces
        self.weight_stress = weight_stress
        self.weight_scaling = weight_scaling
        self.include_forces = include_forces
        self.include_stress = include_stress

        # Results populated after fit()
        self.linear_coeffs_ = None
        self.species_coeffs_ = None
        self.residual_ = None
        self.rank_ = None

    def fit(self, dataset, comm=None):
        """Assemble design matrix and solve; update pot in-place.

        Parameters
        ----------
        dataset : list of entry dicts (see design_matrix module for format)
        comm : mpi4py communicator or None
            When provided each rank processes dataset[rank::size]; the local
            rows are allgathered before the linear solve; fitted coefficients
            are broadcast to all ranks.
        """
        if comm is not None:
            mpi_rank = comm.Get_rank()
            mpi_size = comm.Get_size()
            local_data = dataset[mpi_rank::mpi_size]
        else:
            local_data = dataset

        A_local, b_local = build_design_matrix(
            self.pot,
            local_data,
            weight_energy=self.weight_energy,
            weight_forces=self.weight_forces,
            weight_stress=self.weight_stress,
            weight_scaling=self.weight_scaling,
            include_forces=self.include_forces,
            include_stress=self.include_stress,
        )

        # Normal equations: assemble local (A^T A, A^T b), reduce, then solve.
        # This sends O(n_params²) data per rank instead of O(n_rows × n_params)
        # from allgather — much more scalable at high rank counts.
        M_local = A_local.T @ A_local
        v_local = A_local.T @ b_local

        if comm is not None:
            from mpi4py import MPI as _MPI
            M = np.zeros_like(M_local)
            v = np.zeros_like(v_local)
            comm.Reduce(M_local, M, op=_MPI.SUM, root=0)
            comm.Reduce(v_local, v, op=_MPI.SUM, root=0)
        else:
            M, v = M_local, v_local

        if comm is None or comm.Get_rank() == 0:
            # Small ridge for numerical stability (replaces SVD-based lstsq for
            # near-singular cases; negligible for typical MTP sizes post-Rescale)
            reg = 1e-12 * np.eye(len(M))
            x, residuals, rank_val, _ = lstsq(M + reg, v)
        else:
            x, residuals, rank_val = None, None, None

        if comm is not None:
            x = comm.bcast(x, root=0)

        n_basis = self.pot.get_alpha_scalar_count()
        self.linear_coeffs_ = x[:n_basis]
        self.species_coeffs_ = x[n_basis:]
        try:
            self.residual_ = float(residuals) if residuals is not None else None
        except (TypeError, IndexError):
            self.residual_ = None
        self.rank_ = rank_val

        self.pot.set_linear_coeffs(self.linear_coeffs_)
        self.pot.set_species_coeffs(self.species_coeffs_)

        return self

    def rmse(self, dataset):
        """Root-mean-square energy error (eV/atom) on dataset after fitting."""
        from .design_matrix import _eval_basis
        linear_coeffs = self.linear_coeffs_
        species_coeffs = self.species_coeffs_
        errors = []
        for entry in dataset:
            basis = _eval_basis(self.pot, entry)
            types = entry["types"]
            e_pred = (basis @ linear_coeffs).sum() + sum(species_coeffs[t] for t in types)
            e_ref = float(entry["energy"])
            errors.append((e_pred - e_ref) / len(types))
        return float(np.sqrt(np.mean(np.array(errors)**2)))
