"""calculate_grade, select_add and update_active_set, in place of mlp's commands.

All functions take ASE Atoms objects, with no intermediate files, and a potential of either model.
"""

from __future__ import annotations

import logging

import numpy
from numpy import float64

logger = logging.getLogger(__name__)

from ._core import Equations, MaxVol, PotentialCalculator, read_potential, species_indices
from .almtp_io import read_mvs_header, read_mvs_state, write_mvs_state

# ---------------------------------------------------------------------------
# Selection equations
# ---------------------------------------------------------------------------

DEFAULT_SELECTION_WEIGHTS = {
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


def selection_equations(calc: PotentialCalculator, structures: list, weights: dict, comm=None) -> Equations:
    """The MaxVol equations of *structures* under the calculator's coefficients and the selection *weights*.

    comm : an mpi4py communicator; the equations are then those of this rank's share of *structures*,
           dealt round-robin, each structure numbered by its index in *structures*
    """
    rank, size = (0, 1) if comm is None else (comm.Get_rank(), comm.Get_size())
    eqns = Equations(calc.potential.get_coeff_count(), **weights)
    for number in range(rank, len(structures), size):
        eqns.add(calc.potential, calc.neighbors(structures[number]), number)
    return eqns


def maximize_volume(mv: MaxVol, equations: Equations, pool_id: int, comm=None) -> None:
    """mv.maximize_volume over *equations*, or with an mpi4py communicator over its ranks, each holding the equations of its share."""
    if comm is None: return mv.maximize_volume(equations, pool_id=pool_id)
    from ._core import _core_mpi    # only under a communicator, so a process without one never loads the MPI library
    _core_mpi.maximize_volume(mv, equations, pool_id, comm)


def _open_calculator(potential) -> tuple[PotentialCalculator, str | None]:
    """Accept either a potential path or a preloaded calculator."""
    if isinstance(potential, str):
        return PotentialCalculator(read_potential(potential)), potential
    return potential, None


# ---------------------------------------------------------------------------
# calculate_grade
# ---------------------------------------------------------------------------


def calculate_grade(potential, structures: list, weights: dict | None = None, active_set: MaxVol | None = None) -> list:
    """Compute per-atom extrapolation grades for each structure.

    Grades against an active set: per_atom_grade = max |v_atom @ invA.T|, where v_atom is the per-atom
    information vector (full CoeffCount dim).

    Parameters
    ----------
    potential : str or PotentialCalculator
        Path to a potential file, or a preloaded calculator.
    structures : list of ase.Atoms
        Input structures (must have the correct calculator / species info
        set so that the calculator can build neighbor lists).
    weights, active_set : the selection weights and the active set to grade against; by default those stored
        in the #MVS_v1.1 section of *potential*

    Returns
    -------
    list of ase.Atoms
        Same structures with .arrays["nbh_grades"] and
        .info["features"]["MV_grade"] populated.
    """
    calc, potential_path = _open_calculator(potential)

    if active_set is None:
        stored_weights, active_set = read_mvs_header(potential_path)
        weights = weights or stored_weights
    weights = weights or dict(DEFAULT_SELECTION_WEIGHTS["nbh"])
    site_en_w = float(weights.get("site_en_weight", 1.0))

    for i, atoms in enumerate(structures):
        grades = active_set.grade(selection_equations(calc, [atoms], weights))
        cfg_grade = float(grades.max())

        # Per-atom grades come from the site-energy equations, which come last.
        n_atoms = len(atoms)
        if site_en_w and len(grades) >= n_atoms:
            per_atom = grades[-n_atoms:]
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


def select_add(potential, training_structs: list, candidate_structs: list, threshold: float = 1.001, active_set: MaxVol | None = None, weights: dict | None = None, al_mode: str = "nbh", train_eqns: Equations | None = None, comm=None) -> tuple:
    """D-optimality greedy structure selection.

    Rebuilds the MaxVol active set from *training_structs*, then greedily
    selects from *candidate_structs* the structures whose information vectors
    increase the volume of A (grade > threshold).

    Parameters
    ----------
    potential : str or PotentialCalculator
    training_structs : list of ase.Atoms  (current training set)
    candidate_structs : list of ase.Atoms  (pre-filtered candidates)
    threshold : float
    active_set : MaxVol or None
        The active set to start from, by default the one stored in *potential*; select_add
        searches a copy of it.
    weights : the selection weights, by default those stored with the active set, else al_mode's
    train_eqns : Equations or None
        Equations already built for *training_structs* with these coefficients
        and weights, as returned by update_active_set.  Rebuilt when absent.
    comm : mpi4py communicator or None
        Search over its ranks, each holding the equations of its share of the
        training structures and of the candidates, dealt round-robin. Every
        rank passes the same arguments and returns the same selection.

    Returns
    -------
    tuple of (selected_structs, active_set) — the selected candidate structures, and the
    extended active set, which write_mvs_state stores.
    """
    calc, potential_path = _open_calculator(potential)
    pot = calc.potential
    threshold_swap = pot.threshold_swap

    if active_set is None and potential_path is not None:
        try:
            stored_weights, active_set = read_mvs_header(potential_path)
            weights = weights or stored_weights
        except RuntimeError:
            pass
    weights = weights or dict(DEFAULT_SELECTION_WEIGHTS[al_mode])
    if active_set is None:
        mv = MaxVol(pot.get_coeff_count(), threshold=threshold, threshold_swap=threshold_swap)
    else:
        mv = MaxVol(active_set.A, active_set.invA, threshold=threshold, threshold_swap=threshold_swap)

    if train_eqns is None:
        train_eqns = selection_equations(calc, training_structs, weights, comm)
    cand_eqns = selection_equations(calc, candidate_structs, weights, comm)

    # This three-pass sequence matches mlip-3's select_add and must stay ordered:
    # training rebuild at 1.001, candidate selection at threshold, training pass again, if the model makes it.
    mv.threshold = 1.001
    maximize_volume(mv, train_eqns, _POOL_TRAIN, comm)
    initial = MaxVol(mv.A, mv.invA)
    mv.threshold = threshold
    maximize_volume(mv, cand_eqns, _POOL_CAND, comm)
    if pot.retrain_pass: maximize_volume(mv, train_eqns, _POOL_TRAIN, comm)

    active_indices = sorted({int(struct_index) for active_pool_id, struct_index in zip(mv.active_pool_ids, mv.active_struct_indices, strict=True) if int(active_pool_id) == _POOL_CAND and int(struct_index) >= 0})
    selected_structs = [candidate_structs[i] for i in active_indices]

    # Each selected structure's grade against the active set before the candidates entered it
    for i, atoms in enumerate(selected_structs):
        grade = float(initial.grade(selection_equations(calc, [atoms], weights)).max())
        logger.info(f"  selected structure[{i}]: extrapolation grade (gamma) = {grade:.4f}")

    return selected_structs, mv


def update_active_set(potential: str, training_structs: list, threshold: float = 1.001, weights: dict | None = None, al_mode: str = "nbh", comm=None) -> Equations:
    """Converge the #MVS_v1.1 active set in *potential* over *training_structs*, seeded from the one saved there.

    comm : an mpi4py communicator to search over its ranks, each holding the equations of its share of
           *training_structs*, dealt round-robin. Every rank passes the same arguments, and rank 0 writes *potential*.

    Returns the selection equations of *training_structs*, or under *comm* of this rank's share, for select_add to reuse.
    """
    calc, _ = _open_calculator(potential)
    pot = calc.potential
    try:
        stored_weights, saved, stored_cfgs = read_mvs_state(potential, _POOL_SAVED)
    except RuntimeError:
        stored_weights, saved = dict(DEFAULT_SELECTION_WEIGHTS[al_mode]), None
    weights = weights or stored_weights

    train_eqns = selection_equations(calc, training_structs, weights, comm)

    # The seed is the saved active equations of structures still in the training set, taken from train_eqns
    # rather than the stored A: stored rows may predate the coefficients, and the search never re-grades an active row.
    saved_rows = set()
    if saved is not None:
        def key(atoms): return species_indices(atoms).tobytes(), (numpy.round(atoms.cell[:], 6) + 0.0).tobytes(), (numpy.round(atoms.positions, 6) + 0.0).tobytes()
        index_of = {key(atoms): i for i, atoms in enumerate(training_structs)}
        match = [index_of.get(key(cfg)) for cfg in stored_cfgs]
        saved_rows = {(match[c], e) for p, c, e in zip(saved.active_pool_ids.tolist(), saved.active_struct_indices.tolist(), saved.active_eqn_indices.tolist(), strict=True) if p == _POOL_SAVED and match[c] is not None}
    seed = train_eqns.subset(numpy.array([row in saved_rows for row in zip(train_eqns.structure_indices.tolist(), train_eqns.equation_indices.tolist())], dtype=bool))

    mv = MaxVol(pot.get_coeff_count(), threshold=threshold, threshold_swap=pot.threshold_swap)
    maximize_volume(mv, seed, _POOL_TRAIN, comm)
    maximize_volume(mv, train_eqns, _POOL_TRAIN, comm)

    if comm is None or comm.Get_rank() == 0:
        stored = write_mvs_state(potential, weights, mv, training_structs, _POOL_TRAIN)
        logger.info(f"Active set: {len(stored)}/{len(training_structs)} active structures.")
    return train_eqns
