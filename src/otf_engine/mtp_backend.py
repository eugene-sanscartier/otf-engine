"""calculate_grade, select_add, update_active_set and train, in place of mlip-3's `mlp` commands.

All functions take ASE Atoms objects — no intermediate files.
"""

from __future__ import annotations

import logging
import os
import numpy
from numpy import intp, float64

logger = logging.getLogger(__name__)

from ._mtp import MTPCalculator, MTPTraining, PairMTP, train_mtp, write_mtp
from ._mtp.neighbors import mtp_types, neighbors
from .almtp_io import MVSState, read_mvs_header, read_mvs_state, write_mvs_state
from .maxvol import Equations, MaxVol

# ---------------------------------------------------------------------------
# Selection equations
# ---------------------------------------------------------------------------

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


def selection_equations(pot: MTPTraining, nl, weights: dict) -> Equations:
    """Construct the MaxVol equations one structure contributes.

    Mirrors mlip-3 cfg_selection.cpp::PrepareMatrix(), including its ordering
    and weighting quirks:
      - energy-only mode: 1 scaled total-energy equation
      - mixed E/F/S mode: raw total-energy equation, weighted force equations,
        weighted 9-component stress equations
      - site_en_weight > 0 : raw per-atom equations appended last

    c_all = [c_radial | c_species | beta_linear] (coeff_count columns).

    A force or stress weight needs the full E/F/S gradient; the energy-only and
    site-energy-only modes need one pass over the neighbor list and no force
    gradient.

    Parameters
    ----------
    pot     : MTPTraining — holds current coefficients
    nl      : NeighList for the structure
    weights : dict with energy_weight, force_weight, stress_weight,
              site_en_weight, weight_scaling
    """
    n = nl.n_atoms
    cc = pot.get_coeff_count()

    site_en_w = float(weights.get("site_en_weight", 1.0))
    energy_w = float(weights.get("energy_weight", 0.0))
    force_w = float(weights.get("force_weight", 0.0))
    stress_w = float(weights.get("stress_weight", 0.0))
    ws = float(weights.get("weight_scaling", 1))
    scale = max(n**(ws / 2.0), 1e-30)

    need_forces = force_w != 0.0
    need_stress = stress_w != 0.0

    grads = []
    indices = []

    if need_forces or need_stress:
        eg_all, fg_all, vg_all = pot.eval_grad(nl, forces=True, virial=need_stress)
        eg_all = numpy.asarray(eg_all)
    else:
        eg_all, fg_all, vg_all = numpy.asarray(pot.eval_grad(nl)), None, None

    # ---- total-energy-only mode --------------------------------------------
    if energy_w and not need_forces and not need_stress:
        grads += [eg_all.sum(axis=0, keepdims=True) * (energy_w / scale)]
        indices += [numpy.array([0], dtype=intp)]
    elif energy_w or need_forces or need_stress:
        if energy_w:
            grads += [eg_all.sum(axis=0, keepdims=True)]
            indices += [numpy.array([0], dtype=intp)]
        if need_forces:
            grads += [numpy.asarray(fg_all).reshape(n * 3, cc) * force_w]
            indices += [numpy.arange(1, 1 + 3 * n, dtype=intp)]
        if need_stress:
            # mlip-3 stores the full 3x3 stress block (9 equations), not Voigt-6.
            vg_all = numpy.asarray(vg_all)
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
            grads += [vg_full * (stress_w / scale)]
            indices += [numpy.arange(1 + 3 * n, 1 + 3 * n + 9, dtype=intp)]

    # ---- site-energy equations --------------------------------------------------
    if site_en_w:
        grads += [eg_all]
        indices += [numpy.arange(1 + 3 * n + 9, 1 + 3 * n + 9 + n, dtype=intp)]

    return Equations(grads=numpy.vstack(grads) if grads else numpy.empty((0, cc)), indices=numpy.concatenate(indices) if indices else numpy.empty(0, dtype=intp))


def _open_calculator(potential) -> tuple[MTPCalculator, str | None]:
    """Accept either a potential path or a preloaded calculator."""
    if isinstance(potential, str):
        return MTPCalculator(potential), potential
    return potential, None


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
    calc, potential_path = _open_calculator(potential)
    pot = calc.potential

    if state is None:
        weights, _, invA = read_mvs_header(potential_path)
    else:
        weights, invA = state.weights, state.invA
    site_en_w = float(weights.get("site_en_weight", 1.0))

    # The contraction is a coeff_count-square GEMM, so numpy outruns
    # PairMTPExtrapolation.compute, which walks invA row by row.
    for i, atoms in enumerate(structures):
        grads = selection_equations(pot, calc.neighbors(atoms), weights).grads

        scores = numpy.abs(grads @ invA.T)  # (n_equations, n)
        cfg_grade = float(scores.max())

        # Per-atom grades come from the site-energy equations, which mlip-3
        # appends last.
        n_atoms = len(atoms)
        if site_en_w and len(grads) >= n_atoms:
            per_atom = scores[-n_atoms:].max(axis=1)
            cfg_grade = float(per_atom.max())
        else:
            # No site-energy equations: assign cfg_grade uniformly
            per_atom = numpy.full(n_atoms, cfg_grade)

        atoms.arrays["nbh_grades"] = per_atom.astype(float64)
        atoms.info.setdefault("features", {})["MV_grade"] = cfg_grade

        logger.info(f"  structure[{i}]: extrapolation grade (gamma) = {cfg_grade:.4f}")

    return structures


# ---------------------------------------------------------------------------
# select_add
# ---------------------------------------------------------------------------


def select_add(potential, training_structs: list, candidate_structs: list, threshold: float = 1.001, state: MVSState | None = None, weights: dict | None = None, al_mode: str = "nbh", train_eqns: list | None = None) -> tuple:
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
    train_eqns : list of Equations or None
        Equations already built for *training_structs* with these coefficients
        and weights, as returned by update_active_set.  Rebuilt when absent.

    Returns
    -------
    tuple of (selected_structs, (weights, A, invA)) — the selected candidate
    structures, and the updated active-set state the caller may persist via
    write_mvs_state.
    """
    calc, potential_path = _open_calculator(potential)
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

    if train_eqns is None:
        train_eqns = [selection_equations(pot, calc.neighbors(atoms), weights) for atoms in training_structs]
    cand_eqns = [selection_equations(pot, calc.neighbors(atoms), weights) for atoms in candidate_structs]

    # This three-pass sequence matches mlip-3 select_add and must stay ordered:
    # training rebuild at 1.001, candidate selection at threshold, training pass again.
    mv.threshold = 1.001
    mv.select_candidates(train_eqns, pool_id=_POOL_TRAIN)
    initial_invA = mv.invA.copy()
    mv.threshold = threshold
    mv.select_candidates(cand_eqns, pool_id=_POOL_CAND)
    mv.select_candidates(train_eqns, pool_id=_POOL_TRAIN)

    active_indices = {int(struct_index) for active_pool_id, struct_index in zip(mv.active_pool_ids, mv.active_struct_indices, strict=True) if int(active_pool_id) == _POOL_CAND and int(struct_index) >= 0}
    selected_structs = [atoms for i, atoms in enumerate(candidate_structs) if i in active_indices]

    for i, eqns in enumerate(e for j, e in enumerate(cand_eqns) if j in active_indices):
        grade = float(numpy.abs(eqns.grads @ initial_invA.T).max())
        logger.info(f"  selected structure[{i}]: extrapolation grade (gamma) = {grade:.4f}")

    return selected_structs, (weights, mv.A, mv.invA)


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


def _references(atoms) -> tuple:
    """Energy, forces and virial (mlip-3's PlusStress, eV, xx yy zz xy xz yz) of *atoms*, None where absent."""
    results = atoms.calc.results
    forces = numpy.asarray(results["forces"], dtype=float64) if "forces" in results else None
    virial = None
    if "stress" in results:
        # read_cfg divides PlusStress by -det(cell); ASE's Voigt order is xx yy zz yz xz xy.
        v = -numpy.asarray(results["stress"], dtype=float64) * numpy.linalg.det(atoms.cell.array)
        virial = v[[0, 1, 2, 5, 4, 3]]
    return float(results["energy"]) if "energy" in results else None, forces, virial


def train(potential: str, training_structs: list, save_to: str, settings: dict | None = None, ranks: int | None = None, al_mode: str = "cfg", comm=None) -> None:
    """Train *potential* on *training_structs* as mlip-3's `mlp train` does, and write it with an active set to *save_to*.

    *potential* may be untrained, and gains the species of *training_structs* it lacks.

    settings : `mlp train` options without the leading "--", e.g. {"iteration_limit": 300, "init_random": True}
    ranks    : threads; defaults to the CPUs this process may run on. The fit depends on it only through rounding.
    al_mode  : selection weights when *potential* has no #MVS_v1.1 block, "cfg" or "nbh"
    comm     : an mpi4py communicator to train over its ranks in place of threads. Every rank passes the
               same *training_structs*, and rank 0 writes *save_to*.
    """
    try:
        weights = read_mvs_header(potential)[0]
    except RuntimeError:
        weights = dict(_DEFAULT_SELECTION_WEIGHTS[al_mode])

    cutoff = PairMTP(potential).get_max_cutoff()
    # Under MPI each rank fits its share, dealt round-robin as mlp and the threads deal them.
    share = training_structs if comm is None else training_structs[comm.Get_rank()::comm.Get_size()]
    structures = ((neighbors(atoms, cutoff), *_references(atoms)) for atoms in share)
    options = {key: str(value) for key, value in (settings or {}).items()}
    if comm is None:
        pot = train_mtp(potential, structures, options, ranks or os.process_cpu_count(), checkpoint=lambda pot: write_mtp(pot, save_to))
    else:
        from ._mtp import _mtp_mpi
        pot = _mtp_mpi.train_mtp(potential, structures, options, comm, checkpoint=lambda pot: write_mtp(pot, save_to))
        if comm.Get_rank() != 0: return

    write_mtp(pot, save_to)

    # with iteration_limit 0 an untrained potential stays untrained, and is written without coefficients to select with
    if pot.is_trained(): update_active_set(save_to, training_structs, weights=weights)


def update_active_set(potential: str, training_structs: list, threshold: float = 1.001, weights: dict | None = None, al_mode: str = "nbh") -> list:
    """Converge the #MVS_v1.1 active set in *potential* over *training_structs*, seeded from the one saved there.

    Returns the selection equations of *training_structs*, for select_add to reuse.
    """
    calc = MTPCalculator(potential)
    pot = calc.potential
    try:
        saved = read_mvs_state(potential)
    except RuntimeError:
        saved = None
    if weights is None:
        weights = saved.weights if saved is not None else dict(_DEFAULT_SELECTION_WEIGHTS[al_mode])

    train_eqns = [selection_equations(pot, calc.neighbors(atoms), weights) for atoms in training_structs]

    # The seed is the saved active equations of structures still in the training set, taken from train_eqns
    # rather than the stored A: stored rows may predate the coefficients, and the search never re-grades an active row.
    seed = [numpy.zeros(len(eqns.indices), dtype=bool) for eqns in train_eqns]
    if saved is not None:
        def key(atoms): return mtp_types(atoms).tobytes(), (numpy.round(atoms.cell[:], 6) + 0.0).tobytes(), (numpy.round(atoms.positions, 6) + 0.0).tobytes()
        index_of = {key(atoms): i for i, atoms in enumerate(training_structs)}
        match = [index_of.get(key(cfg)) for cfg in saved.selected_cfgs]
        for c, e in zip(saved.active_cfg_indices.tolist(), saved.active_eqn_indices.tolist(), strict=True):
            if c >= 0 and match[c] is not None: seed[match[c]] |= train_eqns[match[c]].indices == e

    mv = MaxVol(pot.get_coeff_count(), threshold=threshold)
    mv.select_candidates([Equations(grads=eqns.grads[m], indices=eqns.indices[m]) for eqns, m in zip(train_eqns, seed, strict=True)], pool_id=_POOL_TRAIN)
    mv.select_candidates(train_eqns, pool_id=_POOL_TRAIN)

    state = _build_saved_mvs_state(weights, mv, training_structs, _POOL_TRAIN)
    logger.info(f"Active set: {len(state.selected_cfgs)}/{len(training_structs)} active structures, seeded with {sum(int(m.sum()) for m in seed)} equations.")
    write_mvs_state(potential, state)
    return train_eqns
