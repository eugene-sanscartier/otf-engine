"""fit, train and errors: training an MTP or an ETN with otf_engine's trainers, on ASE structures.

As a command line:

    python -m otf_engine.train potential.almtp|potential.etn train.cfg [--save_path=FILE] [--ranks=N] [--al_mode=nbh|cfg] [trainer options]

Takes `mlp train`'s options (--iteration_limit=300, --tolerance=1e-3, --skip_preinit, --energy_weight=1,
...). For an MTP, also the trainer's own (--varpro=false, --riemannian=false, ...; MLP_ALGORITHM lists
those that give mlp's algorithm); for an ETN, mlip-4's training settings mlp lacks (--rescale=false, ...). Both take --balance_species (default true), which gives every species of a structure an
equal share of its force weight. --ranks sets the thread count, by default the CPUs this process may run on. The potential may
be untrained, such as an mlip-3 template; the potential gains the species of the training set it lacks. The
trained potential is written to --save_path (or --save_to, mlp's name), by default over the input,
with an active set built on the training set.
"""

from __future__ import annotations

import logging
import os
import sys

from ._core import PotentialCalculator, Sample, error_table, read_potential
from .io_cfg import read_cfg
from .selection import converge_active_set, stored_selection, write_active_set

logger = logging.getLogger(__name__)


def errors(potential: str, structures: list, title: str, first: int = 1, species: list | None = None) -> str:
    """The errors of *potential* on the labelled *structures*, numbered from *first*, as the table training logs, under *title*."""
    pot = read_potential(potential)
    cutoff = pot.get_max_cutoff()
    table = error_table(pot, [Sample(atoms, cutoff, first + k) for k, atoms in enumerate(structures)], species or [], title)
    return table


# The trainer options that give mlp's algorithm in place of the trainer's defaults, for runs compared with mlp.
MLP_ALGORITHM = {"varpro": "false", "riemannian": "false", "reg_once": "false", "reg_at_main": "false", "no_orthogonalize": "false", "init_noise": "5", "balance_species": "false"}


def fit(potential: str, structures, options: dict, checkpoint=None, ranks: int | None = None, comm=None, inverse_hessian=None):
    """train_mtp or train_etn, as the file *potential* holds, on *ranks* threads, or with an mpi4py communicator over its ranks, each passing its share of *structures*.

    ranks           : threads; without comm, defaults to the CPUs this process may run on
    inverse_hessian : (n, n) float64 array BFGS starts from and leaves its final inverse Hessian in, as train_mtp's; an MTP's only
    """
    pot = read_potential(potential)
    extra = {} if inverse_hessian is None else {"inverse_hessian": inverse_hessian}
    if ranks is None: ranks = os.process_cpu_count() if comm is None else 1
    trained = pot.fit(potential, structures, options, ranks, comm, checkpoint=checkpoint, **extra)
    return trained


def train(potential: str, training_structs: list, save_path: str, settings: dict | None = None, ranks: int | None = None, al_mode: str = "nbh", comm=None) -> None:
    """Train *potential* on *training_structs* with otf_engine's trainer, and write it with an active set to the file *save_path*.

    *potential* may be untrained; it gains the species of *training_structs* it lacks.

    settings : the trainer's options without the leading "--", e.g. {"iteration_limit": 300, "skip_preinit": True}:
               `mlp train`'s; for an MTP, also the trainer's own, MLP_ALGORITHM for mlp's algorithm; for an ETN, mlip-4's
               training settings mlp lacks; and "species", the species' names for the log's error tables, comma-separated
    ranks    : threads; defaults to the CPUs this process may run on. The fit depends on it only through rounding.
    al_mode  : selection weights when *potential* has no #MVS_v1.1 block, "cfg" or "nbh"
    comm     : an mpi4py communicator to train over its ranks in place of threads. Every rank passes the
               same *training_structs*, and rank 0 writes *save_path*.
    """
    weights = stored_selection(potential, al_mode)[0]

    cutoff = read_potential(potential).get_max_cutoff()
    # Under MPI each rank fits its share, dealt round-robin as mlp and the threads deal them.
    rank, size = (0, 1) if comm is None else (comm.Get_rank(), comm.Get_size())
    structures = (Sample(training_structs[i], cutoff, i + 1) for i in range(rank, len(training_structs), size))
    options = {key: str(value) for key, value in (settings or {}).items()}
    pot = fit(potential, structures, options, checkpoint=lambda pot: pot.write(save_path), ranks=ranks, comm=comm)

    # with iteration_limit 0 an untrained potential stays untrained, and is written without coefficients to select with
    if pot.is_trained():
        if rank == 0:
            try:
                pot.write(save_path)
            except Exception as e:
                logger.warning(f"cannot save the trained potential to {save_path}: {type(e).__name__}: {e}")
        active_set, _ = converge_active_set(PotentialCalculator(pot), training_structs, weights, comm=comm)

    if rank == 0:
        pot.write(save_path)
        if pot.is_trained(): write_active_set(save_path, weights, active_set, training_structs)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    args = [a for a in argv if not a.startswith("--")]
    options = dict(a[2:].split("=", 1) if "=" in a else (a[2:], "true") for a in argv if a.startswith("--"))
    if len(args) != 2: raise SystemExit(__doc__)
    potential, training_set = args

    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s")
    ranks = int(options.pop("ranks")) if "ranks" in options else None
    al_mode = options.pop("al_mode", "nbh")
    save_path = options.pop("save_path", options.pop("save_to", potential))

    with open(training_set) as f:
        structures = read_cfg(f)
    train(potential, structures, save_path, settings=options, ranks=ranks, al_mode=al_mode)
    logger.info("training complete")


if __name__ == "__main__":
    main()
