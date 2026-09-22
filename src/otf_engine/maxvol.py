"""MaxVol (maximum-volume) D-optimality algorithm.

Mirrors the mlip-3 C++ implementation (src/maxvol.h, src/maxvol.cpp) and
the LAMMPS pair_mtp_extrapolation.cpp grade formula exactly:

    grade(v) = max_i |sum_j v[j] * invA[i,j]|
             = max |v @ invA.T|

where A (n×n) is the active-set matrix of selected information vectors and
invA = A^{-T} (column-ordered in mlip-3; stored row-major here for numpy).

Threshold 1.001 matches mlip-3's SELECT_THRESHOLD.
init_scale 1e-6 matches mlip-3's INIT_VALUE.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy
from numpy import intp, float64, ndarray


@dataclass
class Equations:
    """The MaxVol equations one structure contributes.

    grads   : (n_equations, coeff_count) d(observable)/d(coefficients)
    indices : (n_equations,) mlip equation index of each
    """

    grads: ndarray
    indices: ndarray


class MaxVol:
    """D-optimality active-set maintained via Sherman-Morrison rank-1 updates.

    Parameters
    ----------
    n : int
        Number of coefficients (CoeffCount of the potential).
    init_scale : float
        Initial diagonal value A = init_scale * I  (mlip-3: 1e-6).
    threshold : float
        Swap threshold: swap when grade(v) > threshold  (mlip-3: 1.001).
    """

    def __init__(self, n: int, init_scale: float = 1e-6, threshold: float = 1.001):
        self.threshold = threshold
        self.A = numpy.eye(n, dtype=float64) * init_scale
        # invA = (A^T)^{-1} = (init_scale * I)^{-1} = (1/init_scale) * I
        self.invA = numpy.eye(n, dtype=float64) / init_scale
        self.active_pool_ids = numpy.full(n, -1, dtype=intp)
        self.active_struct_indices = numpy.full(n, -1, dtype=intp)
        self.active_eqn_indices = numpy.full(n, -1, dtype=intp)

    @classmethod
    def from_arrays(cls, A: ndarray, invA: ndarray, threshold: float = 1.001) -> "MaxVol":
        """Restore from stored A and invA (read from #MVS_v1.1 section)."""
        obj = object.__new__(cls)
        obj.threshold = threshold
        obj.A = numpy.array(A, dtype=float64)
        obj.invA = numpy.array(invA, dtype=float64)
        n = obj.A.shape[0]
        obj.active_pool_ids = numpy.full(n, -1, dtype=intp)
        obj.active_struct_indices = numpy.full(n, -1, dtype=intp)
        obj.active_eqn_indices = numpy.full(n, -1, dtype=intp)
        return obj

    # ------------------------------------------------------------------
    # Grade
    # ------------------------------------------------------------------

    def grade(self, v: ndarray) -> float:
        """Extrapolation grade for information vector v (length n).

        grade = max |v @ invA.T|

        Direct transcription of the LAMMPS C++ loop:
            for i in range(n):
                grade_i = sum(v[j] * invA[i,j] for j in range(n))
            return max(abs(grade_i))
        """
        return float(numpy.abs(v @ self.invA.T).max())

    # ------------------------------------------------------------------
    # Greedy swap (Sherman-Morrison)
    # ------------------------------------------------------------------

    def select_candidates(self, per_struct: list[Equations], pool_id: int = -1, max_swaps: int = 99999) -> ndarray:
        """Batch MaxVol selection mirroring mlip-3's MaximizeVol.

        At each step: find the globally highest-grade equation across ALL
        structures, swap it in, repeat until none grades above threshold or
        max_swaps is exhausted.  Vectorised: all grades computed as one
        matrix-multiply per sweep (O(n_equations × n²), in NumPy).

        Parameters
        ----------
        per_struct : list of Equations
            The equations each structure contributes.
        pool_id : int
            Pool identifier recorded with each active equation.
        max_swaps : int
            Maximum number of swaps (mlip-3 default: 99999).

        Returns the structure index of each active equation.
        """
        if not per_struct:
            return self.active_struct_indices.copy()

        grads = []
        struct_of_eqn = []
        index_of_eqn = []
        for struct_index, eqns in enumerate(per_struct):
            grads.extend(eqns.grads)
            struct_of_eqn.extend([struct_index] * len(eqns.indices))
            index_of_eqn.extend(int(index) for index in eqns.indices)

        if not grads:
            return self.active_struct_indices.copy()
        grads = numpy.asarray(grads, dtype=float64)
        n_swaps = 0

        while n_swaps < max_swaps:
            # grades[j] = max_i |grads[j] @ invA.T|  — one matmul for all equations
            BinvAT = grads @ self.invA.T  # (n_equations, n)
            grades = numpy.abs(BinvAT).max(axis=1)  # (n_equations,)

            best_j = int(grades.argmax())
            if grades[best_j] <= self.threshold:
                break  # converged

            swapped, _k, displaced_grad, displaced_pool_id, displaced_struct_index, displaced_index = self.try_swap(
                grads[best_j],
                pool_id=pool_id,
                struct_index=struct_of_eqn[best_j],
                eqn_index=index_of_eqn[best_j],
            )
            if not swapped:
                break
            # mlip-3 swaps the entering equation with the displaced active one,
            # so future iterations grade the updated candidate matrix.
            grads[best_j] = displaced_grad
            struct_of_eqn[best_j] = displaced_struct_index
            index_of_eqn[best_j] = displaced_index
            n_swaps += 1

        return self.active_struct_indices.copy()

    def restore_active(self, cfg_indices: ndarray, eqn_indices: ndarray, pool_id: int) -> None:
        """Restore the provenance of each active equation from saved indices."""
        cfg_indices = numpy.asarray(cfg_indices, dtype=intp)
        self.active_pool_ids[:] = numpy.where(cfg_indices >= 0, pool_id, -1)
        self.active_struct_indices[:] = cfg_indices
        self.active_eqn_indices[:] = numpy.asarray(eqn_indices, dtype=intp)

    def try_swap(self, v: ndarray, pool_id: int = -1, struct_index: int = -1, eqn_index: int = -1) -> tuple[bool, int, ndarray | None, int, int, int]:
        """Swap v into A if grade(v) > threshold; Woodbury rank-1 update of invA.

        Mirrors mlip-3 MaxVol::UpdateInvA() exactly:
          w     = v @ invA.T              BinvA row for v
          k     = argmax |w|              j0: column / row-of-A to replace
          dv    = v - A[k]
          tmp   = 1 / w[k]
          buf3  = tmp * invA @ dv         (invA matrix-vector, NOT invA.T)
          invA -= outer(buf3, invA[k])    invA[k] = row k of invA

        Returns
        -------
        tuple
            ``(swapped, active_index, displaced_grad, displaced_pool_id,
            displaced_struct_index, displaced_eqn_index)`` — the last four
            describe the equation that v evicted from the active set.
        """
        w = v @ self.invA.T  # grade elements
        k = int(numpy.abs(w).argmax())
        if numpy.abs(w[k]) <= self.threshold:
            return False, -1, None, -1, -1, -1

        dv = v - self.A[k]
        tmp = 1.0 / w[k]
        buf3 = tmp * (self.invA @ dv)  # invA * dv
        self.invA -= numpy.outer(buf3, self.invA[k].copy())
        displaced_grad = self.A[k].copy()
        displaced_pool_id = int(self.active_pool_ids[k])
        displaced_struct_index = int(self.active_struct_indices[k])
        displaced_eqn_index = int(self.active_eqn_indices[k])
        self.A[k] = v
        self.active_pool_ids[k] = pool_id
        self.active_struct_indices[k] = struct_index
        self.active_eqn_indices[k] = eqn_index
        return True, k, displaced_grad, displaced_pool_id, displaced_struct_index, displaced_eqn_index
