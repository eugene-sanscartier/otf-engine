"""Grade structures, select candidates and converge the active set of a potential.

The functions take ASE Atoms objects and a potential of either model.
"""

from __future__ import annotations

import logging

import numpy
from numpy import float64

logger = logging.getLogger(__name__)

from ._core import Equations, MaxVol, Pair, neighbors, read_potential, species_indices
from .io_mvs import read_mvs_header, read_mvs_state, write_mvs_state

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


def selection_equations(potential: Pair, structures: list, weights: dict, comm=None) -> Equations:
    """Return the MaxVol equations of *structures* under the coefficients of *potential* and the selection *weights*.

    The function sets the type_index array of each structure. Each structure
    keeps its index in *structures* as its number.

    Args:
        comm: mpi4py communicator. The equations then cover the round-robin share of *structures* of this rank.
    """
    species = potential.get_species()
    for structure in structures:
        structure.arrays["type_index"] = species_indices(structure, species)

    rank, size = (0, 1) if comm is None else (comm.Get_rank(), comm.Get_size())
    cutoff = potential.get_max_cutoff()
    eqns = Equations(potential.get_coeff_count(), **weights)
    for index in range(rank, len(structures), size):
        eqns.add(potential, neighbors(structures[index], cutoff), index)
    return eqns


def stored_selection(potential_path: str, al_mode: str, coeff_count: int) -> tuple[dict, MaxVol | None]:
    """Return the selection weights and the active set stored in *potential_path*.

    Without a #MVS_v1.1 section, return the default weights of *al_mode* and None.

    Args:
        coeff_count: Coefficient count of the potential in *potential_path*.
    """
    try:
        return read_mvs_header(potential_path, coeff_count)
    except RuntimeError:
        return dict(DEFAULT_SELECTION_WEIGHTS[al_mode]), None


# ---------------------------------------------------------------------------
# calculate_grade
# ---------------------------------------------------------------------------


def calculate_grade(potential_path: str, structures: list, weights: dict | None = None, active_set: MaxVol | None = None) -> list:
    """Compute the extrapolation grade of each atom of each structure against an active set.

    The grade of an atom is max |v_atom @ invA.T|, where v_atom is the information
    vector of the atom, with one entry per coefficient.

    Args:
        potential_path: File of the potential.
        weights: Selection weights, by default those stored with the active set, else the "nbh" defaults.
        active_set: Active set to grade against, by default the one in the #MVS_v1.1 section of *potential_path*.

    Returns:
        The same structures, with .arrays["nbh_grades"] and .info["features"]["MV_grade"] set.
    """
    potential = read_potential(potential_path)

    if active_set is None:
        stored_weights, active_set = read_mvs_header(potential_path, potential.get_coeff_count())
        weights = weights or stored_weights
    weights = weights or dict(DEFAULT_SELECTION_WEIGHTS["nbh"])
    site_en_w = float(weights["site_en_weight"])

    for i, atoms in enumerate(structures):
        grades = active_set.grade(selection_equations(potential, [atoms], weights))
        structure_grade = float(grades.max())

        # Per-atom grades come from the site-energy equations, which come last.
        n_atoms = len(atoms)
        if site_en_w and len(grades) >= n_atoms:
            per_atom = grades[-n_atoms:]
            structure_grade = float(per_atom.max())
        else:
            per_atom = numpy.full(n_atoms, structure_grade)

        atoms.arrays["nbh_grades"] = per_atom.astype(float64)
        atoms.info.setdefault("features", {})["MV_grade"] = structure_grade

        logger.info(f"  structure[{i}]: extrapolation grade (gamma) = {structure_grade:.4f}")

    return structures


# ---------------------------------------------------------------------------
# select_add
# ---------------------------------------------------------------------------


def select_add(potential_path: str, training_set: list, candidates: list, threshold: float = 1.001, active_set: MaxVol | None = None, weights: dict | None = None, al_mode: str = "nbh", train_eqns: Equations | None = None, comm=None) -> tuple:
    """Select the candidates that extend the active set by D-optimality.

    The function rebuilds the MaxVol active set on *training_set*, then greedily
    selects the candidates whose information vectors increase the volume of A
    (grade > *threshold*).

    Args:
        potential_path: File of the potential.
        training_set: Current training set.
        candidates: Preselected candidates.
        active_set: Active set to start from, by default the one stored in *potential_path*. The search works on a copy.
        weights: Selection weights, by default those stored with the active set, else the defaults of *al_mode*.
        train_eqns: Equations of *training_set* under these coefficients and weights, as update_active_set returns them.
            The function builds them when absent.
        comm: mpi4py communicator to search over its ranks. Each rank holds the equations of its round-robin share
            of the training structures and of the candidates. Every rank passes the same arguments and returns the
            same selection.

    Returns:
        The selected candidates, and the extended active set that write_mvs_state stores.
    """
    potential = read_potential(potential_path)

    if active_set is None:
        stored_weights, active_set = stored_selection(potential_path, al_mode, potential.get_coeff_count())
        weights = weights or stored_weights
    weights = weights or dict(DEFAULT_SELECTION_WEIGHTS[al_mode])
    if active_set is None:
        mv = MaxVol(potential.get_coeff_count(), threshold=threshold)
    else:
        mv = MaxVol(active_set.A, active_set.invA, threshold=threshold)

    if train_eqns is None:
        train_eqns = selection_equations(potential, training_set, weights, comm)
    cand_eqns = selection_equations(potential, candidates, weights, comm)

    # Rebuild on the training set at 1.001, select the candidates at threshold, then pass over the training set again.
    mv.threshold = 1.001
    mv.maximize_volume(train_eqns, pool_id=_POOL_TRAIN, comm=comm)
    initial = MaxVol(mv.A, mv.invA)
    mv.threshold = threshold
    mv.maximize_volume(cand_eqns, pool_id=_POOL_CAND, comm=comm)
    mv.maximize_volume(train_eqns, pool_id=_POOL_TRAIN, comm=comm)

    active_indices = sorted({int(structure_index) for active_pool_id, structure_index in zip(mv.active_pool_ids, mv.active_structure_indices, strict=True) if int(active_pool_id) == _POOL_CAND and int(structure_index) >= 0})
    selected = [candidates[i] for i in active_indices]

    # Log the grade of each selected structure against the active set before the candidates entered it
    for i, atoms in enumerate(selected):
        grade = float(initial.grade(selection_equations(potential, [atoms], weights)).max())
        logger.info(f"  selected structure[{i}]: extrapolation grade (gamma) = {grade:.4f}")

    return selected, mv


def update_active_set(potential_path: str, training_set: list, threshold: float = 1.001, weights: dict | None = None, al_mode: str = "nbh", comm=None) -> Equations:
    """Converge the #MVS_v1.1 active set in *potential_path* on *training_set*, starting from the saved active set.

    Args:
        comm: mpi4py communicator to search over its ranks. Each rank holds the equations of its round-robin share
            of *training_set*. Every rank passes the same arguments, and rank 0 writes *potential_path*.

    Returns:
        The selection equations of *training_set*, or of the share of this rank under *comm*, for select_add to reuse.
    """
    potential = read_potential(potential_path)
    try:
        stored_weights, saved, stored_structures = read_mvs_state(potential_path, _POOL_SAVED, potential.get_coeff_count())
    except RuntimeError:
        stored_weights, saved, stored_structures = dict(DEFAULT_SELECTION_WEIGHTS[al_mode]), None, []
    weights = weights or stored_weights

    active_set, train_eqns = converge_active_set(potential, training_set, weights, threshold, saved, stored_structures, comm)
    if comm is None or comm.Get_rank() == 0: write_active_set(potential_path, weights, active_set, training_set)
    return train_eqns


def converge_active_set(potential: Pair, training_set: list, weights: dict, threshold: float = 1.001, saved: MaxVol | None = None, stored_structures: list = (), comm=None) -> tuple[MaxVol, Equations]:
    """Return the active set of *potential* on *training_set*, seeded from *saved*, and the selection equations of *training_set*.

    Args:
        saved: Active set saved with the potential, as read_mvs_state reads it.
        stored_structures: Structures that the rows of *saved* come from, as read_mvs_state reads them.
        comm: mpi4py communicator to search over its ranks. The equations then cover the share of this rank.
    """
    train_eqns = selection_equations(potential, training_set, weights, comm)

    # Seed: the saved active equations of the structures still in the training set, taken from train_eqns.
    saved_rows = set()
    if saved is not None:
        def key(atoms): return species_indices(atoms).tobytes(), (numpy.round(atoms.cell[:], 6) + 0.0).tobytes(), (numpy.round(atoms.positions, 6) + 0.0).tobytes()
        index_of = {key(atoms): i for i, atoms in enumerate(training_set)}
        match = [index_of.get(key(structure)) for structure in stored_structures]
        saved_rows = {(match[c], e) for p, c, e in zip(saved.active_pool_ids.tolist(), saved.active_structure_indices.tolist(), saved.active_eqn_indices.tolist(), strict=True) if p == _POOL_SAVED and match[c] is not None}
    seed = train_eqns.subset(numpy.array([row in saved_rows for row in zip(train_eqns.structure_indices.tolist(), train_eqns.equation_indices.tolist())], dtype=bool))

    active_set = MaxVol(potential.get_coeff_count(), threshold=threshold)
    active_set.maximize_volume(seed, pool_id=_POOL_TRAIN, comm=comm)
    active_set.maximize_volume(train_eqns, pool_id=_POOL_TRAIN, comm=comm)
    return active_set, train_eqns


def write_active_set(path: str, weights: dict, active_set: MaxVol, training_set: list) -> None:
    """Write *active_set*, converged on *training_set* under the selection *weights*, as the #MVS_v1.1 section of the potential file *path*."""
    stored = write_mvs_state(path, weights, active_set, training_set, _POOL_TRAIN)
    logger.info(f"Active set: {len(stored)}/{len(training_set)} active structures.")
