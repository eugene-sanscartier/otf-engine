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

import numpy as np


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
        self.A = np.eye(n, dtype=np.float64) * init_scale
        # invA = (A^T)^{-1} = (init_scale * I)^{-1} = (1/init_scale) * I
        self.invA = np.eye(n, dtype=np.float64) / init_scale
        self.active_labels = np.full(n, -1, dtype=np.intp)
        self.active_eqn_indices = np.full(n, -1, dtype=np.intp)

    @classmethod
    def from_arrays(cls, A: np.ndarray, invA: np.ndarray, threshold: float = 1.001) -> "MaxVol":
        """Restore from stored A and invA (read from #MVS_v1.1 section)."""
        obj = object.__new__(cls)
        obj.threshold = threshold
        obj.A = np.array(A, dtype=np.float64)
        obj.invA = np.array(invA, dtype=np.float64)
        obj.active_labels = np.full(obj.A.shape[0], -1, dtype=np.intp)
        obj.active_eqn_indices = np.full(obj.A.shape[0], -1, dtype=np.intp)
        return obj

    # ------------------------------------------------------------------
    # Grade
    # ------------------------------------------------------------------

    def grade(self, v: np.ndarray) -> float:
        """Extrapolation grade for information vector v (length n).

        grade = max |v @ invA.T|

        Direct transcription of the LAMMPS C++ loop:
            for i in range(n):
                grade_i = sum(v[j] * invA[i,j] for j in range(n))
            return max(abs(grade_i))
        """
        return float(np.abs(v @ self.invA.T).max())

    # ------------------------------------------------------------------
    # Greedy swap (Sherman-Morrison)
    # ------------------------------------------------------------------

    def select_candidates(self, rows_per_struct: list[tuple], labels: np.ndarray | None = None, eqn_indices_per_struct: list[np.ndarray] | None = None, max_swaps: int = 99999) -> np.ndarray:
        """Batch MaxVol selection mirroring mlip-3's MaximizeVol.

        At each step: find the globally highest-grade row across ALL structures,
        swap it in, repeat until no row has grade > threshold or max_swaps is
        exhausted.  Vectorised: all grades computed as one matrix-multiply per
        sweep  (O(n_rows × n²) but in NumPy, not Python loops).

        Parameters
        ----------
        rows_per_struct : list of (np.ndarray, object) tuples
            Each tuple: ``(rows, payload)`` where ``rows`` has shape
            ``(n_rows_for_struct, n)``. The payload is ignored here and kept
            only for call-site compatibility.
        max_swaps : int
            Maximum number of swaps (mlip-3 default: 99999).

        Returns
        -------
        np.ndarray
            Sorted unique labels currently present in the active set and drawn
            from *labels* for this call.
        """
        if not rows_per_struct:
            return np.empty(0, dtype=np.intp)

        # Stack all rows and build a flat index → struct mapping
        all_rows = np.vstack([rows for rows, _payload in rows_per_struct])  # (total_rows, n)
        if all_rows.shape[0] == 0:
            return np.empty(0, dtype=np.intp)
        if labels is None:
            labels = np.arange(len(rows_per_struct), dtype=np.intp)
        if eqn_indices_per_struct is None:
            eqn_indices_per_struct = [np.full(len(rows), -1, dtype=np.intp) for rows, _payload in rows_per_struct]
        row_labels = np.concatenate([np.full(len(rows), labels[i], dtype=np.intp) for i, (rows, _payload) in enumerate(rows_per_struct)])
        row_eqn_indices = np.concatenate([np.asarray(eqn_indices_per_struct[i], dtype=np.intp) for i in range(len(rows_per_struct))])
        n_swaps = 0

        while n_swaps < max_swaps:
            # grades[j] = max_i |all_rows[j] @ invA.T|  — one matmul for all rows
            BinvAT = all_rows @ self.invA.T  # (total_atoms, n)
            grades = np.abs(BinvAT).max(axis=1)  # (total_atoms,)

            best_j = int(grades.argmax())
            if grades[best_j] <= self.threshold:
                break  # converged

            swapped, _k, displaced_row, displaced_label, displaced_eqn_index = self.try_swap(all_rows[best_j], int(row_labels[best_j]), int(row_eqn_indices[best_j]))
            if not swapped:
                break
            # mlip-3 swaps the entering B row with the displaced active row,
            # so future iterations grade the updated candidate matrix.
            all_rows[best_j] = displaced_row
            row_labels[best_j] = displaced_label
            row_eqn_indices[best_j] = displaced_eqn_index
            n_swaps += 1

        active = self.active_labels[self.active_labels >= 0]
        if active.size == 0:
            return np.empty(0, dtype=np.intp)
        label_set = set(int(x) for x in labels)
        return np.array(sorted({int(x) for x in active if int(x) in label_set}), dtype=np.intp)

    def try_swap(self, v: np.ndarray, label: int = -1, eqn_index: int = -1) -> tuple[bool, int, np.ndarray | None, int, int]:
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
            ``(swapped, active_row_index, displaced_row, displaced_label, displaced_eqn_index)``.
        """
        w = v @ self.invA.T  # grade elements
        k = int(np.abs(w).argmax())
        if np.abs(w[k]) <= self.threshold:
            return False, -1, None, -1, -1

        dv = v - self.A[k]
        tmp = 1.0 / w[k]
        buf3 = tmp * (self.invA @ dv)  # invA * dv
        self.invA -= np.outer(buf3, self.invA[k].copy())
        displaced_row = self.A[k].copy()
        displaced_label = int(self.active_labels[k])
        displaced_eqn_index = int(self.active_eqn_indices[k])
        self.A[k] = v
        self.active_labels[k] = label
        self.active_eqn_indices[k] = eqn_index
        return True, k, displaced_row, displaced_label, displaced_eqn_index
