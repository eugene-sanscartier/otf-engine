"""Operations that the pyKMC workers run for an OTF cycle, registered in the pyKMC extra_ops.

pyKMC calls each operation as fn(comm, **kwargs) on every rank of its communicator,
and the operation returns its result on rank 0. pyKMC steps an otf_cycle.OTFCycle on
its manager and runs these operations for it. grade_dump runs as a local operation per
dump on the sessions. update_active_set, select_add and train_potential run as global
operations over every worker. The workers share the working directory of the manager.
"""

from . import otf_cycle, selection, train
from .training_set import read_training_set


def update_active_set(comm, potential_path, training_set_path) -> None:
    """Converge the active set of *potential_path* on the training set, using the ranks of *comm*.

    Rank 0 writes *potential_path*.
    """
    selection.update_active_set(potential_path, read_training_set(training_set_path), comm=comm)


def grade_dump(comm, dump, potential_path, species=None) -> list | None:
    """Grade one extrapolative dump on rank 0 of *comm*.

    Returns:
        The graded structures on rank 0, None on the other ranks.
    """
    if comm.Get_rank() != 0: return None
    structures = otf_cycle.grade_dump(dump, potential_path, species)
    return structures


def select_add(comm, potential_path, training_set_path, candidates) -> list | None:
    """Select the candidates that extend the active set of *potential_path* on the training set, using the ranks of *comm*.

    Returns:
        The selected candidates on rank 0, None on the other ranks.
    """
    selected, _ = selection.select_add(potential_path, read_training_set(training_set_path), candidates, comm=comm)
    return selected if comm.Get_rank() == 0 else None


def train_potential(comm, potential_path, training_set_path, save_path, options=None) -> None:
    """Train the potential in *potential_path* on the training set, using the ranks of *comm*.

    Rank 0 writes the potential with its active set to *save_path*.
    """
    train.train(potential_path, read_training_set(training_set_path), save_path, options=options, comm=comm)
