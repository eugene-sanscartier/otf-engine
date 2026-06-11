"""
Design matrix assembly for MTP linear fitting.

Each dataset entry is a dict with keys:
    types        : int32  (n_atoms,)
    ilist        : int32  (n_atoms,)
    numneigh     : int32  (n_atoms,)
    firstneigh   : int32  (sum_numneigh,)
    displacements: float64 (sum_numneigh, 3)
    energy       : float
    forces       : float64 (n_atoms, 3)   — optional
    stress       : float64 (6,)           — optional, ASE Voigt convention

All forces/stresses are optional; pass weight=0 to exclude them.
"""

import numpy as np
from numpy import float64


def _eval_basis(pot, entry):
    """Return basis matrix (n_atoms, n_basis) for one dataset entry."""
    return np.asarray(pot.eval_basis(
        entry["types"],
        entry["ilist"],
        entry["numneigh"],
        entry["firstneigh"],
        entry["displacements"],
    ), dtype=float64)


def _eval_force_and_stress_columns(pot, entry, n_basis):
    """
    Return force design matrix (n_atoms, 3, n_basis) and
    virial design matrix (6, n_basis) for one entry.

    Uses the column-by-column trick: set linear_coeffs = e_k, call compute(),
    read forces and virials.  Cost: n_basis compute() calls.
    """
    original = pot.get_linear_coeffs().copy()
    original_sc = pot.get_species_coeffs().copy()

    # Zero species_coeffs so only the linear part contributes to forces/virials
    pot.set_species_coeffs(np.zeros_like(original_sc))

    n_atoms = len(entry["types"])
    A_F = np.zeros((n_atoms, 3, n_basis))
    A_S = np.zeros((6, n_basis))

    e_k = np.zeros(n_basis)
    for k in range(n_basis):
        e_k[k] = 1.0
        pot.set_linear_coeffs(e_k)
        result = pot.compute(
            entry["types"],
            entry["ilist"],
            entry["numneigh"],
            entry["firstneigh"],
            entry["displacements"],
            compute_virials=True,
            compute_eatom=False,
        )
        A_F[:, :, k] = result["forces"]
        # virials are (xx,yy,zz,xy,xz,yz); convert to ASE Voigt (xx,yy,zz,yz,xz,xy)
        v = result["virials"]
        A_S[:, k] = [-v[0], -v[1], -v[2], -v[5], -v[4], -v[3]]
        e_k[k] = 0.0

    # Restore
    pot.set_linear_coeffs(original)
    pot.set_species_coeffs(original_sc)
    return A_F, A_S


def build_design_matrix(pot, dataset, weight_energy=1.0, weight_forces=0.01, weight_stress=0.001, weight_scaling=1, include_forces=True, include_stress=True):
    """
    Assemble the full weighted design matrix and right-hand-side vector
    for linear MTP fitting.

    The unknown vector has length  n_basis + species_count:
        x = [linear_coeffs | species_coeffs]

    Parameters
    ----------
    pot : MTPPotential
    dataset : list of entry dicts (see module docstring)
    weight_energy / weight_forces / weight_stress : float
        Base weights for the three observable types.
    weight_scaling : int
        Exponent for per-config size normalisation: energy and stress rows are
        divided by N^weight_scaling.  weight_scaling=1 (default) → divide by N,
        matching mlip-3's wgt_scale_power_energy/stress = 1 convention.
        Forces are never scaled (mlip-3 wgt_scale_power_forces default = 0).
    include_forces / include_stress : bool
        Whether to add force/stress rows (requires n_basis compute() calls
        per structure; can be slow for large n_basis).

    Returns
    -------
    A : ndarray (n_rows, n_basis + species_count)
    b : ndarray (n_rows,)
    """
    n_basis = pot.get_alpha_scalar_count()
    n_species = pot.get_species_count()
    n_params = n_basis + n_species

    rows_A, rows_b = [], []

    for entry in dataset:
        types = entry["types"]
        n_atoms = len(types)
        basis = _eval_basis(pot, entry)  # (n_atoms, n_basis)

        scale = n_atoms**weight_scaling

        # --- Energy row ---
        w_e = weight_energy / scale
        row_e = np.zeros(n_params)
        row_e[:n_basis] = basis.sum(axis=0)
        # species one-hot columns
        for t in types:
            row_e[n_basis + t] += 1.0
        rows_A.append(w_e * row_e)
        rows_b.append(np.array([w_e * float(entry["energy"])]))

        # --- Force rows ---
        if include_forces and "forces" in entry:
            A_F, A_S = _eval_force_and_stress_columns(pot, entry, n_basis)
            w_f = weight_forces  # forces: no per-size scaling (weight_scaling_forces=0)
            # shape (n_atoms*3, n_params): force rows have no species column contribution
            A_F_flat = A_F.reshape(n_atoms * 3, n_basis)
            B_F = np.zeros((n_atoms * 3, n_params))
            B_F[:, :n_basis] = A_F_flat
            rows_A.append(w_f * B_F)
            rows_b.append(w_f * entry["forces"].ravel())

            # --- Stress rows ---
            if include_stress and "stress" in entry:
                vol = entry.get("volume")
                if vol is None:
                    raise ValueError("Entry must include 'volume' when include_stress=True")
                w_s = weight_stress / scale
                B_S = np.zeros((6, n_params))
                B_S[:, :n_basis] = A_S / vol
                rows_A.append(w_s * B_S)
                rows_b.append(w_s * np.asarray(entry["stress"], dtype=float64))
        elif include_stress and "stress" in entry and "forces" not in entry:
            # Compute stress columns without forces
            _, A_S = _eval_force_and_stress_columns(pot, entry, n_basis)
            vol = entry.get("volume")
            if vol is None:
                raise ValueError("Entry must include 'volume' when include_stress=True")
            w_s = weight_stress / scale
            B_S = np.zeros((6, n_params))
            B_S[:, :n_basis] = A_S / vol
            rows_A.append(w_s * B_S)
            rows_b.append(w_s * np.asarray(entry["stress"], dtype=float64))

    A = np.vstack(rows_A)
    b = np.concatenate(rows_b)
    return A, b
