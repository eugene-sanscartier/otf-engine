"""The operations pyKMC's workers run for an OTF-MTP cycle, registered in its extra_ops.

Each is called as fn(comm, **kwargs) on every rank of the communicator it runs on, and returns its
result on rank 0 of it. pyKMC steps an otf_mtp.OTFCycle on its manager and runs these for it:
grade_dump as a local operation per dump on the sessions, train_potential as a global one over every
worker. The workers share the manager's working directory.
"""

from . import otf_mtp
from .mtp_backend import train


def grade_dump(comm, dump, potential, species=None) -> list | None:
    """Grade one extrapolative dump on rank 0 of comm.

    Returns the graded structures on rank 0, None on the other ranks.
    """
    if comm.Get_rank() != 0: return None
    structures = otf_mtp.grade_dump(dump, potential, species)
    return structures


def train_potential(comm, potential, training_set, save_to, species=None, settings=None) -> None:
    """Train potential on training_set over the ranks of comm, and write it with an active set to save_to on rank 0."""
    train(potential, otf_mtp.load_structures(training_set, species), save_to, settings=settings, comm=comm)
