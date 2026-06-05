"""Pure-Python implementations of calculate_grade, select_add, and train.

These replace the three launcher.run() calls for the external mlip-3 binary.
Only structure evaluation (eval_structures) still uses launcher.run().

All functions accept and return ASE Atoms objects — no intermediate files.
"""

from __future__ import annotations

import numpy as np

from ._mtp import MTPPotential, MTPCalculator, write_mtp
from .almtp_io import read_active_set, write_active_set
from .maxvol import MaxVol

# ---------------------------------------------------------------------------
# Helper: ASE Atoms → design-matrix entry dict
# ---------------------------------------------------------------------------


def _atoms_to_entry(atoms, calc: MTPCalculator) -> dict:
    """Convert ASE Atoms to the entry dict expected by design_matrix."""
    types = np.asarray(atoms.arrays["_mtp_types"], dtype=np.int32) if "_mtp_types" in atoms.arrays else calc._symbols_to_types(atoms)
    ilist, numneigh, firstneigh, displacements = calc._build_neighbor_list(atoms)

    entry = {
        "types": types,
        "ilist": ilist,
        "numneigh": numneigh,
        "firstneigh": firstneigh,
        "displacements": displacements,
        "energy": float(atoms.get_potential_energy()),
    }
    try:
        entry["forces"] = atoms.get_forces().astype(np.float64)
    except Exception:
        pass
    try:
        stress = atoms.get_stress()  # ASE Voigt (xx,yy,zz,yz,xz,xy)
        entry["stress"] = stress.astype(np.float64)
        entry["volume"] = float(atoms.get_volume())
    except Exception:
        pass
    return entry


# ---------------------------------------------------------------------------
# Helper: neighbor-list-only entry (no DFT data needed)
# ---------------------------------------------------------------------------


def _atoms_to_nl_entry(atoms, calc: MTPCalculator) -> dict:
    """Build minimal entry dict for grade/selection (no DFT data required)."""
    types = np.asarray(atoms.arrays["_mtp_types"], dtype=np.int32) if "_mtp_types" in atoms.arrays else calc._symbols_to_types(atoms)
    ilist, numneigh, firstneigh, displacements = calc._build_neighbor_list(atoms)
    return {
        "types": types,
        "ilist": ilist,
        "numneigh": numneigh,
        "firstneigh": firstneigh,
        "displacements": displacements,
    }


# ---------------------------------------------------------------------------
# Helper: MaxVol information rows for one structure
# ---------------------------------------------------------------------------

_NL_ARGS = ("types", "ilist", "numneigh", "firstneigh", "displacements")


def _build_info_rows(pot: MTPPotential, entry: dict, weights: dict) -> np.ndarray:
    """Construct the MaxVol information matrix rows for one structure.

    Mirrors mlip-3 cfg_selection.cpp::PrepareMatrix():
      - site_en_weight > 0 : n_atoms rows — per-atom ∂E_i/∂c_all
      - energy_weight  > 0 : 1 row — total-energy gradient scaled by w/N^(ws/2)
      - force_weight   > 0 : n_atoms×3 rows — ∂F_{i,d}/∂c_all  (no N-scaling)
      - stress_weight  > 0 : 6 rows — ∂virial_{ab}/∂c_all scaled by w/N^(ws/2)

    c_all = [c_radial | c_species | β_linear] (coeff_count columns).

    Parameters
    ----------
    pot     : MTPPotential — holds current coefficients
    entry   : output of _atoms_to_nl_entry (neighbor list only)
    weights : dict with energy_weight, force_weight, stress_weight,
              site_en_weight, weight_scaling
    """
    nl_args = tuple(entry[k] for k in _NL_ARGS)
    n = len(entry["types"])
    cc = pot.get_coeff_count()

    site_en_w = float(weights.get("site_en_weight", 1.0))
    energy_w = float(weights.get("energy_weight", 0.0))
    force_w = float(weights.get("force_weight", 0.0))
    stress_w = float(weights.get("stress_weight", 0.0))
    ws = float(weights.get("weight_scaling", 1))
    scale = max(n**ws, 1e-30)

    need_force = force_w != 0.0
    need_vg = stress_w != 0.0

    rows = []
    eg_all = fg_all = vg_all = None  # computed lazily

    def _ensure_grad_all():
        nonlocal eg_all, fg_all, vg_all
        if eg_all is None:
            eg_all, fg_all, vg_all = pot.eval_grad_all(*nl_args, need_vg)
            eg_all = np.asarray(eg_all)
            fg_all = np.asarray(fg_all)
            if need_vg:
                vg_all = np.asarray(vg_all)

    # ---- site-energy rows --------------------------------------------------
    if site_en_w:
        _ensure_grad_all()
        rows.append(eg_all * site_en_w)

    # ---- total-energy row --------------------------------------------------
    if energy_w:
        _ensure_grad_all()
        rows.append(eg_all.sum(axis=0, keepdims=True) * (energy_w / scale))

    # ---- force rows --------------------------------------------------------
    if need_force:
        _ensure_grad_all()
        rows.append(fg_all.reshape(n * 3, cc) * force_w)

    # ---- virial/stress rows ------------------------------------------------
    if need_vg:
        _ensure_grad_all()
        # Apply same sign-flip and ASE Voigt reorder as the training loss.
        # vg_all from eval_grad_all uses C++ order (xx,yy,zz,xy,xz,yz);
        # training uses (-xx,-yy,-zz,-yz,-xz,-xy) = ASE Voigt virial convention.
        vg_voigt = np.stack([-vg_all[0], -vg_all[1], -vg_all[2], -vg_all[5], -vg_all[4], -vg_all[3]])
        rows.append(vg_voigt * (stress_w / scale))

    return np.vstack(rows) if rows else np.empty((0, cc))


# ---------------------------------------------------------------------------
# Helper: infer species list from a potential file
# ---------------------------------------------------------------------------


def _infer_species(pot: MTPPotential, structures: list | None = None) -> list[str]:
    """Infer species order for symbol-based fallback paths."""
...

# ---------------------------------------------------------------------------
# calculate_grade
# ---------------------------------------------------------------------------


def calculate_grade(potential_path: str, structures: list) -> list:
    """Compute per-atom extrapolation grades for each structure.

    Reads the MaxVol active set (invA) from the #MVS_v1.1 section of
    *potential_path* and applies the grade formula:
        per_atom_grade = max |v_atom @ invA.T|
    where v_atom is the per-atom information vector (full CoeffCount dim).

    Parameters
    ----------
    potential_path : str
        Path to .almtp / .mtp file.
    structures : list of ase.Atoms
        Input structures (must have the correct calculator / species info
        set so that MTPCalculator can build neighbor lists).

    Returns
    -------
    list of ase.Atoms
        Same structures with .arrays["nbh_grades"] and
        .info["features"]["MV_grade"] populated.
    """
    pot = MTPPotential(potential_path)
    species = _infer_species(pot, structures)
    calc = MTPCalculator(potential_path, species=species)

    weights, A, invA = read_active_set(potential_path)
    if invA is None:
        raise RuntimeError(f"No #MVS_v1.1 active-set section found in {potential_path}. "
                           "Run select_add (or mlp select_add) first to initialise the active set.")
    mv = MaxVol.from_arrays(A, invA)

    if weights is None:
        weights = {"site_en_weight": 1.0, "energy_weight": 0.0, "force_weight": 0.0, "stress_weight": 0.0, "weight_scaling": 2}

    for atoms in structures:
        entry = _atoms_to_nl_entry(atoms, calc)
        rows = _build_info_rows(pot, entry, weights)  # (n_rows, coeff_count)

        scores = np.abs(rows @ mv.invA.T)  # (n_rows, n)
        cfg_grade = float(scores.max())

        # Per-atom grades from site-energy rows (first n_atoms rows when site_en_weight>0)
        site_en_w = float(weights.get("site_en_weight", 1.0))
        n_atoms = len(entry["types"])
        if site_en_w and len(rows) >= n_atoms:
            per_atom = scores[:n_atoms].max(axis=1)
        else:
            # No site-energy rows: assign cfg_grade uniformly
            per_atom = np.full(n_atoms, cfg_grade)

        atoms.arrays["nbh_grades"] = per_atom.astype(np.float64)
        atoms.info.setdefault("features", {})["MV_grade"] = cfg_grade

    return structures


# ---------------------------------------------------------------------------
# select_add
# ---------------------------------------------------------------------------


def select_add(potential_path: str, training_structs: list, candidate_structs: list, threshold: float = 1.001) -> tuple:
    """D-optimality greedy structure selection.

    Rebuilds the MaxVol active set from *training_structs*, then greedily
    selects from *candidate_structs* the structures whose information vectors
    increase the volume of A (grade > threshold).

    Parameters
    ----------
    potential_path : str
    training_structs : list of ase.Atoms  (current training set)
    candidate_structs : list of ase.Atoms  (pre-filtered candidates)
    threshold : float  (mlip-3 default: 1.001)

    Returns
    -------
    tuple of (selected, weights, A, invA) where selected is a list of ase.Atoms
    (selected candidate structures) and weights/A/invA are the updated active-set
    state that the caller may persist via write_active_set if desired.
    """
    pot = MTPPotential(potential_path)
    all_structs = list(training_structs) + list(candidate_structs)
    species = _infer_species(pot, all_structs)
    calc = MTPCalculator(potential_path, species=species)

    # Load existing active set as starting point (mirrors mlip-3 select_add).
    # Starting from the stored A preserves history from previous OTF cycles.
    weights, A, invA = read_active_set(potential_path)
    if weights is None:
        weights = {"energy_weight": 0.0, "force_weight": 0.0, "stress_weight": 0.0, "site_en_weight": 1.0, "weight_scaling": 2}

    n = pot.get_coeff_count()
    mv = MaxVol.from_arrays(A, invA, threshold=threshold) if A is not None else MaxVol(n, threshold=threshold)

    def _rows(atoms):
        return _build_info_rows(pot, _atoms_to_nl_entry(atoms, calc), weights)

    # Update active set from training set, then select from candidates.
    mv.select_candidates([(_rows(a), a) for a in training_structs])
    selected = mv.select_candidates([(_rows(a), a) for a in candidate_structs])

    return selected, weights, mv.A, mv.invA


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


def train(potential_path: str, training_structs: list, save_to: str, iteration_limit: int = 300, energy_weight: float = 1.0, force_weight: float = 0.01, stress_weight: float = 0.001, weight_scaling: int = 1, pre_train=None, comm=None, backend: str = "scipy", optimizer: str = "lbfgs") -> None:
    """Fit MTP coefficients using NonlinearFitter (bi-level L-BFGS-B).

    Mirrors mlip-3's bi-level training approach:
      outer loop: L-BFGS-B on radial basis coefficients (analytical gradient)
      inner loop: weighted least-squares for linear / species coefficients

    Parameters
    ----------
    potential_path : str
    training_structs : list of ase.Atoms (must have energy/forces/stress calculators)
    save_to : str  — output .almtp / .mtp path
    iteration_limit : int  — max outer iterations
    energy_weight / force_weight / stress_weight : float
        Fitting weights (mlip-3 defaults: 1.0 / 0.01 / 0.001).
    weight_scaling : int
        Per-config size exponent (mlip-3 default 1 → divide energy/stress rows by
        sqrt(N); forces unscaled).
    pre_train : bool or None
        When None (default) auto-detects: if the potential has no trained radial
        coefficients (all zero) → True; otherwise → False.  Mirrors mlip-3's
        `!inited` gate: a fresh potential runs 75-iteration warm-up + Rescale
        before the main loop; a previously-trained potential skips this.
    comm : mpi4py communicator or None  — enables MPI-parallel design matrix
    """
    from ._mtp.nonlinear_fit import NonlinearFitter
    from ._mtp.linear_fit import LinearFitter
    from ._mtp.rescale import rescale

    pot = MTPPotential(potential_path)
    species = _infer_species(pot, training_structs)
    calc = MTPCalculator(potential_path, species=species)

    dataset = [_atoms_to_entry(atoms, calc) for atoms in training_structs]

    # --- Update min_dist — mirrors mlip-3 AddSpecies(): min_val = 0.99 * min(distances) ---
    min_dist = np.inf
    for entry in dataset:
        disps = entry["displacements"]
        if len(disps):
            min_dist = min(min_dist, float(np.sqrt((disps**2).sum(axis=1)).min()))
    if np.isfinite(min_dist):
        pot.set_min_cutoff(0.99 * min_dist)

    # Auto-detect whether to run pre-training (mirrors mlip-3's `inited` flag).
    if pre_train is None:
        pre_train = not np.any(pot.get_radial_basis_coeffs())

    lf_kwargs = dict(weight_energy=energy_weight, weight_forces=force_weight, weight_stress=stress_weight, weight_scaling=weight_scaling)

    nl_kwargs = dict(**lf_kwargs, backend=backend, optimizer=optimizer)

    # --- Pre-training (fresh potential only) — mirrors mtpr_trainer.cpp:756-776 ---
    if pre_train:
        LinearFitter(pot, **lf_kwargs).fit(dataset, comm=comm)
        rescale(pot, dataset, lf_kwargs)
        NonlinearFitter(pot, maxiter=75, **nl_kwargs).fit(dataset, comm=comm)
        rescale(pot, dataset, lf_kwargs)

    # --- Main training ---
    NonlinearFitter(pot, maxiter=iteration_limit, **nl_kwargs).fit(dataset, comm=comm)

    # --- Post-training: final linear re-solve + Rescale ---
    LinearFitter(pot, **lf_kwargs).fit(dataset, comm=comm)
    rescale(pot, dataset, lf_kwargs)

    write_mtp(pot, save_to)

    # Rebuild the active set from the training data using the NEW coefficients,
    # then embed it in save_to — mirrors mlip-3's train which calls select.Save().
    # A new MTPCalculator is needed because pot was updated in-place but calc
    # still holds the old coefficients loaded from potential_path.
    calc_new = MTPCalculator(save_to, species=species)
    n = pot.get_coeff_count()
    # Preserve selection weights from the input file; fall back to mlip-3 defaults.
    file_weights, _, _ = read_active_set(potential_path)
    sel_weights = {
        "energy_weight": 0.0,
        "force_weight": 0.0,
        "stress_weight": 0.0,
        "site_en_weight": 1.0,
        "weight_scaling": 2,  # mlip-3 cfg_selection.h default
    }
    if file_weights is not None:
        sel_weights.update(file_weights)
    mv = MaxVol(n)
    train_rows = [(calc_new.eval_grad(atoms), atoms) for atoms in training_structs]
    mv.select_candidates(train_rows)
    write_active_set(save_to, sel_weights, mv.A, mv.invA)
