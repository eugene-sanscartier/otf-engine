"""fit, train and errors: training an MTP or an ETN with otf_engine's trainers, on ASE structures.

As a command line:

    python -m otf_engine.train potential.almtp|potential.etn train.cfg [--save_to=FILE] [--ranks=N] [--al_mode=nbh|cfg] [trainer options]

Takes `mlp train`'s options (--iteration_limit=300, --tolerance=1e-3, --skip_preinit, --energy_weight=1,
...). For an MTP, also the trainer's own (--varpro=false, --riemannian=false, ...; MLP_ALGORITHM lists
those that give mlp's algorithm); for an ETN, mlip-4's training settings mlp lacks (--rescale=false, ...). Every species of a structure has an equal share of its force weight. --ranks sets the thread count, by default the CPUs this process may run on. The potential may
be untrained, such as an mlip-3 template; the potential gains the species of the training set it lacks. The
trained potential is written to --save_to, by default over the input, with an active set built on
the training set.
"""

from __future__ import annotations

import logging
import os
import sys
from importlib import import_module

from ._core import Sample, error_table, read_potential
from .almtp_io import read_mvs_header
from .io_cfg import read_cfg
from .selection import DEFAULT_SELECTION_WEIGHTS, update_active_set


def errors(potential: str, structures: list, title: str, first: int = 1, species: list | None = None) -> str:
    """The errors of *potential* on the labelled *structures*, numbered from *first*, as the table training logs, under *title*."""
    pot = read_potential(potential)
    cutoff = pot.get_max_cutoff()
    table = error_table(pot, [Sample(atoms, cutoff, first + k) for k, atoms in enumerate(structures)], species or [], title)
    return table


# The trainer options that give mlp's algorithm in place of the trainer's defaults, for runs compared with mlp.
MLP_ALGORITHM = {"varpro": "false", "riemannian": "false", "reg_once": "false", "reg_at_main": "false", "no_orthogonalize": "false", "init_noise": "5"}


def fit(potential: str, structures, options: dict, checkpoint=None, ranks: int | None = None, comm=None, inverse_hessian=None, balance_species: bool = True):
    """train_mtp or train_etn, as the file *potential* holds, on *ranks* threads, or with an mpi4py communicator over its ranks, each passing its share of *structures*.

    inverse_hessian : (n, n) float64 array BFGS starts from and leaves its final inverse Hessian in, as train_mtp's; an MTP's only
    balance_species : give every species of a structure an equal share of its force weight; False weighs every force alike
    """
    pot = read_potential(potential)
    extra = {} if inverse_hessian is None else {"inverse_hessian": inverse_hessian}
    if comm is None: return pot.train_on_threads(potential, structures, options, ranks or os.process_cpu_count(), checkpoint=checkpoint, balance_species=balance_species, **extra)

    module, name = pot.mpi_trainer    # imported here, so a process without a communicator never loads the MPI library
    train_over_mpi = getattr(import_module(module), name)
    return train_over_mpi(potential, structures, options, comm, checkpoint=checkpoint, balance_species=balance_species, **extra)


def train(potential: str, training_structs: list, save_to: str, settings: dict | None = None, ranks: int | None = None, al_mode: str = "nbh", comm=None, balance_species: bool = True) -> None:
    """Train *potential* on *training_structs* with otf_engine's trainer, and write it with an active set to *save_to*.

    *potential* may be untrained; it gains the species of *training_structs* it lacks.

    settings : the trainer's options without the leading "--", e.g. {"iteration_limit": 300, "skip_preinit": True}:
               `mlp train`'s; for an MTP, also the trainer's own, MLP_ALGORITHM for mlp's algorithm; for an ETN, mlip-4's
               training settings mlp lacks; and "species", the species' names for the log's error tables, comma-separated
    ranks    : threads; defaults to the CPUs this process may run on. The fit depends on it only through rounding.
    al_mode  : selection weights when *potential* has no #MVS_v1.1 block, "cfg" or "nbh"
    comm     : an mpi4py communicator to train over its ranks in place of threads. Every rank passes the
               same *training_structs*, and rank 0 writes *save_to*.
    balance_species : give every species of a structure an equal share of its force weight; False weighs every force alike
    """
    try:
        weights = read_mvs_header(potential)[0]
    except RuntimeError:
        weights = dict(DEFAULT_SELECTION_WEIGHTS[al_mode])

    cutoff = read_potential(potential).get_max_cutoff()
    # Under MPI each rank fits its share, dealt round-robin as mlp and the threads deal them.
    rank, size = (0, 1) if comm is None else (comm.Get_rank(), comm.Get_size())
    structures = (Sample(training_structs[i], cutoff, i + 1) for i in range(rank, len(training_structs), size))
    options = {key: str(value) for key, value in (settings or {}).items()}
    pot = fit(potential, structures, options, checkpoint=lambda pot: pot.write(save_to), ranks=ranks, comm=comm, balance_species=balance_species)

    if rank == 0: pot.write(save_to)
    if comm is not None: comm.Barrier()    # every rank reads save_to back

    # with iteration_limit 0 an untrained potential stays untrained, and is written without coefficients to select with
    if pot.is_trained(): update_active_set(save_to, training_structs, weights=weights, comm=comm)


def main(argv=None, balance_species: bool = True):
    argv = sys.argv[1:] if argv is None else argv
    args = [a for a in argv if not a.startswith("--")]
    options = dict(a[2:].split("=", 1) if "=" in a else (a[2:], "true") for a in argv if a.startswith("--"))
    if len(args) != 2: raise SystemExit(__doc__)
    potential, training_set = args

    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s")
    ranks = int(options.pop("ranks")) if "ranks" in options else None
    al_mode = options.pop("al_mode", "nbh")
    save_to = options.pop("save_to", potential)

    with open(training_set) as f:
        structures = read_cfg(f)
    train(potential, structures, save_to, settings=options, ranks=ranks, al_mode=al_mode, balance_species=balance_species)
    logging.getLogger(__name__).info("training complete")


if __name__ == "__main__":
    main()
