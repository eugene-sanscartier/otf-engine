"""Pure-Python implementations of calculate_grade, select_add, and train.

These replace the three launcher.run() calls for the external mlip-3 binary.
Only structure evaluation (eval_structures) still uses launcher.run().

All functions accept and return ASE Atoms objects — no intermediate files.
"""

from __future__ import annotations

import numpy
from numpy import intp, float64

from ._mtp import MTPPotential, MTPCalculator, write_mtp
from .almtp_io import MVSState, read_mvs_state, write_mvs_state
from .maxvol import MaxVol, Rows

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
        entry["forces"] = atoms.get_forces().astype(float64)
    except Exception:
        pass
    try:
        stress = atoms.get_stress()  # ASE Voigt (xx,yy,zz,yz,xz,xy)
        entry["stress"] = stress.astype(float64)
        entry["volume"] = float(atoms.get_volume())
    except Exception:
        pass
    return entry


# ---------------------------------------------------------------------------
# Helper: MaxVol information rows for one structure
# ---------------------------------------------------------------------------

_NL_ARGS = ("types", "ilist", "numneigh", "firstneigh", "displacements")
_DEFAULT_SELECTION_WEIGHTS = {
    "cfg": {
        "energy_weight": 1.0,
        "force_weight": 0.0,
        "stress_weight": 0.0,
        "site_en_weight": 0.0,
        "weight_scaling": 2
    },
    "nbh": {
        "energy_weight": 0.0,
        "force_weight": 0.0,
        "stress_weight": 0.0,
        "site_en_weight": 1.0,
        "weight_scaling": 2
    },
}
_POOL_SAVED = 0
_POOL_TRAIN = 1
_POOL_CAND = 2


def _info_rows(pot: MTPPotential, atoms, calc: MTPCalculator, weights: dict) -> Rows:
    """Construct the MaxVol information rows and mlip equation indices.

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
    atoms   : ASE atoms object
    calc    : calculator used to build the neighbor list
    weights : dict with energy_weight, force_weight, stress_weight,
              site_en_weight, weight_scaling
    """
    types = calc._atoms_to_types(atoms)
    ilist, numneigh, firstneigh, displacements = calc._build_neighbor_list(atoms)
    entry = {
        "types": types,
        "ilist": ilist,
        "numneigh": numneigh,
        "firstneigh": firstneigh,
        "displacements": displacements,
    }
    nl_args = tuple(entry[k] for k in _NL_ARGS)
    n = len(types)
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
    eqn_indices = []
    eg_all = fg_all = vg_all = None  # computed lazily

    def _ensure_grad_all():
        nonlocal eg_all, fg_all, vg_all
        if eg_all is None:
            eg_all, fg_all, vg_all = pot.eval_grad_all(*nl_args, need_vg)
            eg_all = numpy.asarray(eg_all)
            fg_all = numpy.asarray(fg_all)
            if need_vg:
                vg_all = numpy.asarray(vg_all)

    # ---- total-energy-only mode --------------------------------------------
    if energy_w and not need_force and not need_vg:
        _ensure_grad_all()
        rows += [eg_all.sum(axis=0, keepdims=True) * (energy_w / scale)]
        eqn_indices += [numpy.array([0], dtype=intp)]
    elif energy_w or need_force or need_vg:
        _ensure_grad_all()
        if energy_w:
            rows += [eg_all.sum(axis=0, keepdims=True)]
            eqn_indices += [numpy.array([0], dtype=intp)]
        if need_force:
            rows += [fg_all.reshape(n * 3, cc) * force_w]
            eqn_indices += [numpy.arange(1, 1 + 3 * n, dtype=intp)]
        if need_vg:
            # mlip-3 stores the full 3x3 stress block (9 rows), not Voigt-6.
            vg_full = numpy.stack([
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
            rows += [vg_full * (stress_w / scale)]
            eqn_indices += [numpy.arange(1 + 3 * n, 1 + 3 * n + 9, dtype=intp)]

    # ---- site-energy rows --------------------------------------------------
    if site_en_w:
        _ensure_grad_all()
        rows += [eg_all]
        eqn_indices += [numpy.arange(1 + 3 * n + 9, 1 + 3 * n + 9 + n, dtype=intp)]

    return Rows(rows=numpy.vstack(rows) if rows else numpy.empty((0, cc)), eqn_indices=numpy.concatenate(eqn_indices) if eqn_indices else numpy.empty(0, dtype=intp))


def _build_saved_mvs_state(weights: dict, mv: MaxVol, structs: list, pool_id: int) -> MVSState:
    selected_labels = sorted({int(struct_index) for active_pool_id, struct_index in zip(mv.active_pool_ids, mv.active_struct_indices, strict=True) if int(active_pool_id) == pool_id and 0 <= int(struct_index) < len(structs)})
    selected_cfgs = []
    for label in selected_labels:
        atoms = structs[label].copy()
        atoms.calc = structs[label].calc
        eqn_indices = sorted({int(eqn_index)
                              for active_pool_id, struct_index, eqn_index in zip(
                                  mv.active_pool_ids,
                                  mv.active_struct_indices,
                                  mv.active_eqn_indices,
                                  strict=True,
                              ) if int(active_pool_id) == pool_id and int(struct_index) == label and int(eqn_index) >= 0})
        if eqn_indices:
            atoms.info.setdefault("features", {})["selected_eqn_inds"] = ",".join(str(eqn_index) for eqn_index in eqn_indices)
        selected_cfgs += [atoms]

    cfg_index_of_label = {label: i for i, label in enumerate(selected_labels)}
    active_cfg_indices = numpy.array([cfg_index_of_label.get(int(struct_index), -1) if int(active_pool_id) == pool_id else -1 for active_pool_id, struct_index in zip(mv.active_pool_ids, mv.active_struct_indices, strict=True)], dtype=intp)
    return MVSState(weights=weights, A=mv.A, invA=mv.invA, active_cfg_indices=active_cfg_indices, active_eqn_indices=numpy.asarray(mv.active_eqn_indices, dtype=intp), selected_cfgs=selected_cfgs)


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
        rows = _info_rows(pot, atoms, calc, weights).rows

        scores = numpy.abs(rows @ mv.invA.T)  # (n_rows, n)
        cfg_grade = float(scores.max())

        # Per-atom grades come from neighborhood rows, which mlip-3 appends last.
        site_en_w = float(weights.get("site_en_weight", 1.0))
        n_atoms = len(atoms)
        if site_en_w and len(rows) >= n_atoms:
            per_atom = scores[-n_atoms:].max(axis=1)
            cfg_grade = float(per_atom.max())
        else:
            # No site-energy rows: assign cfg_grade uniformly
            per_atom = numpy.full(n_atoms, cfg_grade)

        atoms.arrays["nbh_grades"] = per_atom.astype(float64)
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
        mv = MaxVol.from_arrays(state.A, state.invA, threshold=threshold)
        mv.restore_active(state.active_cfg_indices, state.active_eqn_indices, _POOL_SAVED)
    else:
        if weights is None:
            weights = dict(_DEFAULT_SELECTION_WEIGHTS[al_mode])
        mv = MaxVol(n, threshold=threshold)

    train_rows = [_info_rows(pot, atoms, calc, weights) for atoms in training_structs]
    cand_rows = [_info_rows(pot, atoms, calc, weights) for atoms in candidate_structs]

    # This three-pass sequence matches mlip-3 select_add and must stay ordered:
    # training rebuild at 1.001, candidate selection at threshold, training pass again.
    mv.threshold = 1.001
    mv.select_candidates(train_rows, pool_id=_POOL_TRAIN)
    mv.threshold = threshold
    mv.select_candidates(cand_rows, pool_id=_POOL_CAND)
    mv.select_candidates(train_rows, pool_id=_POOL_TRAIN)

    active = {int(struct_index) for active_pool_id, struct_index in zip(mv.active_pool_ids, mv.active_struct_indices, strict=True) if int(active_pool_id) == _POOL_CAND and int(struct_index) >= 0}
    selected = [atoms for i, atoms in enumerate(candidate_structs) if i in active]

    return selected, (weights, mv.A, mv.invA)


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
    min_dist = numpy.inf
    for entry in dataset:
        disps = entry["displacements"]
        if len(disps):
            min_dist = min(min_dist, float(numpy.sqrt((disps**2).sum(axis=1)).min()))
    if numpy.isfinite(min_dist):
        pot.set_min_cutoff(0.99 * min_dist)

    # Auto-detect whether to run pre-training (mirrors mlip-3's `inited` flag).
    if pre_train is None:
        pre_train = not numpy.any(pot.get_radial_basis_coeffs())

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
    train_rows = [_info_rows(calc_new.potential, atoms, calc_new, sel_weights) for atoms in training_structs]
    mv.select_candidates(train_rows, pool_id=_POOL_TRAIN)
    write_mvs_state(save_to, _build_saved_mvs_state(sel_weights, mv, training_structs, _POOL_TRAIN))
