"""Pure-Python implementations of calculate_grade, select_add, and train.

These replace the three launcher.run() calls for the external mlip-3 binary.
Only structure evaluation (eval_structures) still uses launcher.run().

All functions accept and return ASE Atoms objects — no intermediate files.
"""

from __future__ import annotations

import numpy as np

from ._mtp import MTPPotential, MTPCalculator, write_mtp
from .almtp_io import MVSState, read_mvs_state, write_mvs_state
from .maxvol import MaxVol

# ---------------------------------------------------------------------------
# Helper: ASE Atoms → design-matrix entry dict
# ---------------------------------------------------------------------------


def _atoms_to_entry(atoms, calc: MTPCalculator) -> dict:
    """Convert ASE Atoms to the entry dict expected by design_matrix."""
    types = calc._atoms_to_types(atoms)
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
    types = calc._atoms_to_types(atoms)
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
_DEFAULT_SELECTION_WEIGHTS = {
    "cfg": {"energy_weight": 1.0, "force_weight": 0.0, "stress_weight": 0.0, "site_en_weight": 0.0, "weight_scaling": 2},
    "nbh": {"energy_weight": 0.0, "force_weight": 0.0, "stress_weight": 0.0, "site_en_weight": 1.0, "weight_scaling": 2},
}


def _cfg_key(atoms) -> tuple:
    types = atoms.arrays.get("type_index")
    if types is None:
        types = atoms.numbers
    return (
        tuple(np.asarray(types, dtype=np.int32)),
        tuple(np.round(atoms.cell.array.reshape(-1), 12)),
        tuple(np.round(atoms.positions.reshape(-1), 12)),
    )


def _build_info_rows(pot: MTPPotential, entry: dict, weights: dict) -> np.ndarray:
    """Construct the MaxVol information matrix rows for one structure.

    Mirrors mlip-3 cfg_selection.cpp::PrepareMatrix(), including its row
    ordering and weighting quirks:
      - energy-only mode: 1 scaled total-energy row
      - mixed E/F/S mode: raw total-energy row, weighted force rows,
        weighted 9-component stress rows
      - site_en_weight > 0 : raw per-atom rows appended last

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
    scale = max(n**(ws / 2.0), 1e-30)

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

    # ---- total-energy-only mode --------------------------------------------
    if energy_w and not need_force and not need_vg:
        _ensure_grad_all()
        rows.append(eg_all.sum(axis=0, keepdims=True) * (energy_w / scale))
    elif energy_w or need_force or need_vg:
        _ensure_grad_all()
        if energy_w:
            rows.append(eg_all.sum(axis=0, keepdims=True))
        if need_force:
            rows.append(fg_all.reshape(n * 3, cc) * force_w)
        if need_vg:
            # mlip-3 stores the full 3x3 stress block (9 rows), not Voigt-6.
            vg_full = np.stack([
                vg_all[0],
                vg_all[3],
                vg_all[4],
                vg_all[3],
                vg_all[1],
                vg_all[5],
                vg_all[4],
                vg_all[5],
                vg_all[2],
            ])
            rows.append(vg_full * (stress_w / scale))

    # ---- site-energy rows --------------------------------------------------
    if site_en_w:
        _ensure_grad_all()
        rows.append(eg_all)

    return np.vstack(rows) if rows else np.empty((0, cc))


def _build_info_row_eqn_indices(entry: dict, weights: dict) -> np.ndarray:
    """Return mlip-3 equation indices for the rows built by `_build_info_rows`."""
    n = len(entry["types"])
    site_en_w = float(weights.get("site_en_weight", 1.0))
    energy_w = float(weights.get("energy_weight", 0.0))
    force_w = float(weights.get("force_weight", 0.0))
    stress_w = float(weights.get("stress_weight", 0.0))

    indices = []
    if energy_w and force_w == 0.0 and stress_w == 0.0:
        indices.append(np.array([0], dtype=np.intp))
    elif energy_w or force_w or stress_w:
        if energy_w:
            indices.append(np.array([0], dtype=np.intp))
        if force_w:
            indices.append(np.arange(1, 1 + 3 * n, dtype=np.intp))
        if stress_w:
            indices.append(np.arange(1 + 3 * n, 1 + 3 * n + 9, dtype=np.intp))

    if site_en_w:
        indices.append(np.arange(1 + 3 * n + 9, 1 + 3 * n + 9 + n, dtype=np.intp))

    return np.concatenate(indices) if indices else np.empty(0, dtype=np.intp)


def _resolve_saved_active_rows(state: MVSState, training_structs: list, candidate_structs: list) -> tuple[np.ndarray, np.ndarray]:
    label_by_sig = {}
    for i, atoms in enumerate(training_structs):
        label_by_sig[_cfg_key(atoms)] = i
    for i, atoms in enumerate(candidate_structs):
        label_by_sig[_cfg_key(atoms)] = i + len(training_structs)

    selected_labels = np.array([label_by_sig.get(_cfg_key(atoms), -i - 2) for i, atoms in enumerate(state.selected_cfgs)], dtype=np.intp)
    active_labels = np.array(
        [selected_labels[idx] if 0 <= idx < len(selected_labels) else -1 for idx in state.active_cfg_indices],
        dtype=np.intp,
    )
    return active_labels, np.asarray(state.active_eqn_indices, dtype=np.intp)


def _build_saved_mvs_state(weights: dict, mv: MaxVol, structs: list) -> MVSState:
    selected_labels = sorted({int(label) for label in mv.active_labels if 0 <= int(label) < len(structs)})
    selected_cfgs = []
    for label in selected_labels:
        atoms = structs[label].copy()
        atoms.calc = structs[label].calc
        eqn_indices = sorted({int(eqn_index) for active_label, eqn_index in zip(mv.active_labels, mv.active_eqn_indices, strict=True) if int(active_label) == label and int(eqn_index) >= 0})
        if eqn_indices:
            atoms.info.setdefault("features", {})["selected_eqn_inds"] = ",".join(str(eqn_index) for eqn_index in eqn_indices)
        selected_cfgs.append(atoms)

    cfg_index_of_label = {label: i for i, label in enumerate(selected_labels)}
    active_cfg_indices = np.array([cfg_index_of_label.get(int(label), -1) for label in mv.active_labels], dtype=np.intp)
    return MVSState(weights=weights, A=mv.A, invA=mv.invA, active_cfg_indices=active_cfg_indices, active_eqn_indices=np.asarray(mv.active_eqn_indices, dtype=np.intp), selected_cfgs=selected_cfgs)


# ---------------------------------------------------------------------------
# calculate_grade
# ---------------------------------------------------------------------------


def calculate_grade(potential, structures: list, state: MVSState | None = None) -> list:
    """Compute per-atom extrapolation grades for each structure.

    Reads the MaxVol active set (invA) from the #MVS_v1.1 section of
    *potential_path* and applies the grade formula:
        per_atom_grade = max |v_atom @ invA.T|
    where v_atom is the per-atom information vector (full CoeffCount dim).

    Parameters
    ----------
    potential : str or MTPCalculator
        Path to a potential file, or a preloaded calculator.
    structures : list of ase.Atoms
        Input structures (must have the correct calculator / species info
        set so that MTPCalculator can build neighbor lists).

    Returns
    -------
    list of ase.Atoms
        Same structures with .arrays["nbh_grades"] and
        .info["features"]["MV_grade"] populated.
    """
    if isinstance(potential, str):
        calc = MTPCalculator(potential)
        potential_path = potential
    else:
        calc = potential
        potential_path = None
    pot = calc.potential

    if state is None and potential_path is not None:
        state = read_mvs_state(potential_path)
    weights = state.weights
    mv = MaxVol.from_arrays(state.A, state.invA)

    for atoms in structures:
        entry = _atoms_to_nl_entry(atoms, calc)
        rows = _build_info_rows(pot, entry, weights)  # (n_rows, coeff_count)

        scores = np.abs(rows @ mv.invA.T)  # (n_rows, n)
        cfg_grade = float(scores.max())

        # Per-atom grades come from neighborhood rows, which mlip-3 appends last.
        site_en_w = float(weights.get("site_en_weight", 1.0))
        n_atoms = len(entry["types"])
        if site_en_w and len(rows) >= n_atoms:
            per_atom = scores[-n_atoms:].max(axis=1)
            cfg_grade = float(per_atom.max())
        else:
            # No site-energy rows: assign cfg_grade uniformly
            per_atom = np.full(n_atoms, cfg_grade)

        atoms.arrays["nbh_grades"] = per_atom.astype(np.float64)
        atoms.info.setdefault("features", {})["MV_grade"] = cfg_grade

    return structures


# ---------------------------------------------------------------------------
# select_add
# ---------------------------------------------------------------------------


def select_add(potential, training_structs: list, candidate_structs: list, threshold: float = 1.001, state: MVSState | None = None, weights: dict | None = None, al_mode: str = "cfg") -> tuple:
    """D-optimality greedy structure selection.

    Rebuilds the MaxVol active set from *training_structs*, then greedily
    selects from *candidate_structs* the structures whose information vectors
    increase the volume of A (grade > threshold).

    Parameters
    ----------
    potential : str or MTPCalculator
    training_structs : list of ase.Atoms  (current training set)
    candidate_structs : list of ase.Atoms  (pre-filtered candidates)
    threshold : float  (mlip-3 default: 1.001)

    Returns
    -------
    tuple of (selected, weights, A, invA) where selected is a list of ase.Atoms
    (selected candidate structures) and weights/A/invA are the updated active-set
    state that the caller may persist via write_mvs_state if desired.
    """
    if isinstance(potential, str):
        calc = MTPCalculator(potential)
        potential_path = potential
    else:
        calc = potential
        potential_path = None
    pot = calc.potential

    n = pot.get_coeff_count()
    if state is None and potential_path is not None:
        try:
            state = read_mvs_state(potential_path)
        except RuntimeError:
            state = None

    if state is not None:
        if weights is None:
            weights = state.weights
        active_labels, active_eqn_indices = _resolve_saved_active_rows(state, training_structs, candidate_structs)
        mv = MaxVol.from_arrays(state.A, state.invA, threshold=threshold)
        mv.active_labels = active_labels
        mv.active_eqn_indices = active_eqn_indices
    else:
        if weights is None:
            weights = dict(_DEFAULT_SELECTION_WEIGHTS[al_mode])
        mv = MaxVol(n, threshold=threshold)

    def _rows(atoms):
        return _build_info_rows(pot, _atoms_to_nl_entry(atoms, calc), weights)

    def _eqn_indices(atoms):
        return _build_info_row_eqn_indices(_atoms_to_nl_entry(atoms, calc), weights)

    train_rows = [(_rows(a), a) for a in training_structs]
    cand_rows = [(_rows(a), a) for a in candidate_structs]
    train_eqn_indices = [_eqn_indices(a) for a in training_structs]
    cand_eqn_indices = [_eqn_indices(a) for a in candidate_structs]
    train_labels = np.arange(len(training_structs), dtype=np.intp)
    cand_labels = np.arange(len(candidate_structs), dtype=np.intp) + len(training_structs)

    # This three-pass sequence matches mlip-3 select_add and must stay ordered:
    # training rebuild at 1.001, candidate selection at threshold, training pass again.
    mv.threshold = 1.001
    mv.select_candidates(train_rows, labels=train_labels, eqn_indices_per_struct=train_eqn_indices)
    mv.threshold = threshold
    mv.select_candidates(cand_rows, labels=cand_labels, eqn_indices_per_struct=cand_eqn_indices)
    mv.select_candidates(train_rows, labels=train_labels, eqn_indices_per_struct=train_eqn_indices)

    active = set(int(x) for x in mv.active_labels if x >= len(training_structs))
    selected = [atoms for i, atoms in enumerate(candidate_structs) if (i + len(training_structs)) in active]

    return selected, weights, mv.A, mv.invA


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


def train(potential, training_structs: list, save_to: str, iteration_limit: int = 300, energy_weight: float = 1.0, force_weight: float = 0.01, stress_weight: float = 0.001, weight_scaling: int = 1, pre_train=None, comm=None, backend: str = "scipy", optimizer: str = "lbfgs", al_mode: str = "cfg", selection_state: MVSState | None = None, selection_weights: dict | None = None) -> None:
    """Fit MTP coefficients using NonlinearFitter (bi-level L-BFGS-B).

    Mirrors mlip-3's bi-level training approach:
      outer loop: L-BFGS-B on radial basis coefficients (analytical gradient)
      inner loop: weighted least-squares for linear / species coefficients

    Parameters
    ----------
    potential : str or MTPCalculator
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

    if isinstance(potential, str):
        calc = MTPCalculator(potential)
        potential_path = potential
    else:
        calc = potential
        potential_path = None
    pot = calc.potential

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
    calc_new = MTPCalculator(save_to)
    n = pot.get_coeff_count()
    if selection_weights is not None:
        sel_weights = selection_weights
    elif selection_state is not None:
        sel_weights = selection_state.weights
    elif potential_path is not None:
        try:
            sel_weights = read_mvs_state(potential_path).weights
        except RuntimeError:
            sel_weights = dict(_DEFAULT_SELECTION_WEIGHTS[al_mode])
    else:
        sel_weights = dict(_DEFAULT_SELECTION_WEIGHTS[al_mode])

    mv = MaxVol(n, threshold=1.001)
    train_entries = [_atoms_to_nl_entry(atoms, calc_new) for atoms in training_structs]
    train_rows = [(_build_info_rows(calc_new.potential, entry, sel_weights), atoms) for entry, atoms in zip(train_entries, training_structs, strict=True)]
    train_eqn_indices = [_build_info_row_eqn_indices(entry, sel_weights) for entry in train_entries]
    mv.select_candidates(train_rows, eqn_indices_per_struct=train_eqn_indices)
    write_mvs_state(save_to, _build_saved_mvs_state(sel_weights, mv, training_structs))
