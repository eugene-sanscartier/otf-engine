"""calculate_grade, select_add, update_active_set and train, in place of mlip-3's `mlp` commands.

All functions take ASE Atoms objects — no intermediate files.
"""

from __future__ import annotations

import logging
import os
import numpy
from numpy import intp, float64

logger = logging.getLogger(__name__)

from ._mtp import Equations, MaxVol, MTPCalculator, PairMTP, train_mtp, write_mtp
from ._mtp.neighbors import mtp_types, neighbors
from .almtp_io import MVSState, read_mvs_header, read_mvs_state, write_mvs_state

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


def selection_equations(calc: MTPCalculator, structures: list, weights: dict, numbers=None) -> Equations:
    """The MaxVol equations of *structures* under the calculator's coefficients and the selection *weights*.

    numbers : each structure's number in the pool, by default its index
    """
    eqns = Equations(calc.potential.get_coeff_count(), **weights)
    for number, atoms in zip(numbers or range(len(structures)), structures, strict=True):
        eqns.add(calc.potential, calc.neighbors(atoms), number)
    return eqns


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

    if state is None:
        weights, A, invA = read_mvs_header(potential_path)
    else:
        weights, A, invA = state.weights, state.A, state.invA
    site_en_w = float(weights.get("site_en_weight", 1.0))
    mv = MaxVol(A, invA)

    for i, atoms in enumerate(structures):
        grades = mv.grade(selection_equations(calc, [atoms], weights))
        cfg_grade = float(grades.max())

        # Per-atom grades come from the site-energy equations, which mlip-3
        # appends last.
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


def select_add(potential, training_structs: list, candidate_structs: list, threshold: float = 1.001, state: MVSState | None = None, weights: dict | None = None, al_mode: str = "nbh", train_eqns: Equations | None = None, comm=None) -> tuple:
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
    train_eqns : Equations or None
        Equations already built for *training_structs* with these coefficients
        and weights, as returned by update_active_set.  Rebuilt when absent.
    comm : mpi4py communicator or None
        Search over its ranks, each holding the equations of its share of the
        training structures and of the candidates, dealt round-robin. Every
        rank passes the same arguments and returns the same selection.

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
        mv = MaxVol(state.A, state.invA, threshold=threshold)
        mv.restore_active(state.active_cfg_indices, state.active_eqn_indices, _POOL_SAVED)
    else:
        if weights is None:
            weights = dict(_DEFAULT_SELECTION_WEIGHTS[al_mode])
        mv = MaxVol(n, threshold=threshold)

    rank, size = (0, 1) if comm is None else (comm.Get_rank(), comm.Get_size())
    train_share, cand_share = range(rank, len(training_structs), size), range(rank, len(candidate_structs), size)
    error = None
    try:
        if train_eqns is None:
            train_eqns = selection_equations(calc, [training_structs[i] for i in train_share], weights, train_share)
        cand_eqns = selection_equations(calc, [candidate_structs[i] for i in cand_share], weights, cand_share)
    except Exception as e:
        error = e
    _raise_together(comm, error)

    if comm is None:
        maximize_volume = mv.maximize_volume
    else:
        from ._mtp import _mtp_mpi
        def maximize_volume(pool, pool_id): _mtp_mpi.maximize_volume(mv, pool, pool_id, comm)

    # This three-pass sequence matches mlip-3 select_add and must stay ordered:
    # training rebuild at 1.001, candidate selection at threshold, training pass again.
    mv.threshold = 1.001
    maximize_volume(train_eqns, pool_id=_POOL_TRAIN)
    initial = MaxVol(mv.A, mv.invA)
    mv.threshold = threshold
    maximize_volume(cand_eqns, pool_id=_POOL_CAND)
    maximize_volume(train_eqns, pool_id=_POOL_TRAIN)

    active_indices = sorted({int(struct_index) for active_pool_id, struct_index in zip(mv.active_pool_ids, mv.active_struct_indices, strict=True) if int(active_pool_id) == _POOL_CAND and int(struct_index) >= 0})
    selected_structs = [candidate_structs[i] for i in active_indices]

    # Each selected structure's grade against the active set before the candidates entered it
    grades = numpy.zeros(len(candidate_structs))
    numpy.maximum.at(grades, cand_eqns.structure_indices, initial.grade(cand_eqns))
    if comm is not None: grades = comm.allreduce(grades)    # each candidate's grade comes from one rank, the others' are 0
    for i, j in enumerate(active_indices):
        logger.info(f"  selected structure[{i}]: extrapolation grade (gamma) = {grades[j]:.4f}")

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

    error = None
    if comm is None or comm.Get_rank() == 0:
        try:
            write_mtp(pot, save_to)
        except Exception as e:
            error = e
    _raise_together(comm, error)

    # with iteration_limit 0 an untrained potential stays untrained, and is written without coefficients to select with
    if pot.is_trained(): update_active_set(save_to, training_structs, weights=weights, comm=comm)


def _raise_together(comm, error: Exception | None) -> None:
    """Raise *error* on this rank, and on every other rank of *comm* when any rank has one, so that none is left in a collective."""
    if comm is not None and comm.allreduce(error is not None) and error is None:
        raise RuntimeError("another rank failed; its error is raised there")
    if error is not None: raise error


def update_active_set(potential: str, training_structs: list, threshold: float = 1.001, weights: dict | None = None, al_mode: str = "nbh", comm=None) -> Equations:
    """Converge the #MVS_v1.1 active set in *potential* over *training_structs*, seeded from the one saved there.

    comm : an mpi4py communicator to search over its ranks, each holding the equations of its share of
           *training_structs*, dealt round-robin. Every rank passes the same arguments, and rank 0 writes *potential*.

    Returns the selection equations of *training_structs*, or under *comm* of this rank's share, for select_add to reuse.
    """
    rank, size = (0, 1) if comm is None else (comm.Get_rank(), comm.Get_size())
    calc = MTPCalculator(potential)
    pot = calc.potential
    try:
        saved = read_mvs_state(potential)
    except RuntimeError:
        saved = None
    if weights is None:
        weights = saved.weights if saved is not None else dict(_DEFAULT_SELECTION_WEIGHTS[al_mode])

    share = range(rank, len(training_structs), size)
    error = None
    try:
        train_eqns = selection_equations(calc, [training_structs[i] for i in share], weights, share)
    except Exception as e:
        error = e
    _raise_together(comm, error)

    # The seed is the saved active equations of structures still in the training set, taken from train_eqns
    # rather than the stored A: stored rows may predate the coefficients, and the search never re-grades an active row.
    saved_rows = set()
    if saved is not None:
        def key(atoms): return mtp_types(atoms).tobytes(), (numpy.round(atoms.cell[:], 6) + 0.0).tobytes(), (numpy.round(atoms.positions, 6) + 0.0).tobytes()
        index_of = {key(atoms): i for i, atoms in enumerate(training_structs)}
        match = [index_of.get(key(cfg)) for cfg in saved.selected_cfgs]
        saved_rows = {(match[c], e) for c, e in zip(saved.active_cfg_indices.tolist(), saved.active_eqn_indices.tolist(), strict=True) if c >= 0 and match[c] is not None}
    seed = train_eqns.subset(numpy.array([row in saved_rows for row in zip(train_eqns.structure_indices.tolist(), train_eqns.equation_indices.tolist())], dtype=bool))

    mv = MaxVol(pot.get_coeff_count(), threshold=threshold)
    if comm is None:
        maximize_volume = mv.maximize_volume
    else:
        from ._mtp import _mtp_mpi
        def maximize_volume(pool, pool_id): _mtp_mpi.maximize_volume(mv, pool, pool_id, comm)
    maximize_volume(seed, pool_id=_POOL_TRAIN)
    maximize_volume(train_eqns, pool_id=_POOL_TRAIN)

    seeded = len(seed) if comm is None else comm.allreduce(len(seed))
    if rank == 0:
        state = _build_saved_mvs_state(weights, mv, training_structs, _POOL_TRAIN)
        logger.info(f"Active set: {len(state.selected_cfgs)}/{len(training_structs)} active structures, seeded with {seeded} equations.")
        write_mvs_state(potential, state)
    return train_eqns
