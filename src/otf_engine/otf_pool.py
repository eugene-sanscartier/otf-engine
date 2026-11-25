"""The operations pyKMC's workers run for an OTF-MTP cycle, registered in its extra_ops.

Each is called as fn(comm, **kwargs) on every rank of the communicator it runs on, and returns its
result on rank 0 of it. pyKMC steps an otf_cycle.OTFCycle on its manager and runs these for it:
grade_dump as a local operation per dump on the sessions, update_active_set, select_add and
train_potential as global ones over every worker. The workers share the manager's working directory.
"""

from . import otf_cycle, selection, train
from .training_set import read_training_set


def update_active_set(comm, potential_path, training_set_path) -> None:
    """Converge the active set in potential_path over the training set in training_set_path, over the ranks of comm; rank 0 writes potential_path."""
    selection.update_active_set(potential_path, read_training_set(training_set_path), comm=comm)


def grade_dump(comm, dump, potential_path, species=None) -> list | None:
    """Grade one extrapolative dump on rank 0 of comm.

    Returns the graded structures on rank 0, None on the other ranks.
    """
    if comm.Get_rank() != 0: return None
    structures = otf_cycle.grade_dump(dump, potential_path, species)
    return structures


def select_add(comm, potential_path, training_set_path, candidates) -> list | None:
    """Select, over the ranks of comm, the candidates that extend the active set of potential_path over the training set in training_set_path.

    Returns the selected candidates on rank 0, None on the other ranks.
    """
    selected, _ = selection.select_add(potential_path, read_training_set(training_set_path), candidates, comm=comm)
    return selected if comm.Get_rank() == 0 else None


def train_potential(comm, potential_path, training_set_path, save_path, options=None) -> None:
    """Train the potential in potential_path on the training set in training_set_path over the ranks of comm, and write it with an active set to the file save_path on rank 0."""
    train.train(potential_path, read_training_set(training_set_path), save_path, options=options, comm=comm)
