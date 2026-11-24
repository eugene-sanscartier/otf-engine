"""fit, train and errors: training an MTP or an ETN with MTPFit or ETNFit, on ASE structures.

As a command line:

    python -m otf_engine.train potential.almtp|potential.etn train.cfg [--save_path=FILE] [--ranks=N] [--al_mode=nbh|cfg] [fit options]

Takes `mlp train`'s options (--iteration_limit=300, --tolerance=1e-3, --skip_preinit, --energy_weight=1,
...). For an MTP, also the fit's own (--varpro=false, --riemannian=false, ...; MLP_ALGORITHM lists
those that give mlp's algorithm); for an ETN, mlip-4's training settings mlp lacks (--rescale=false, ...) and the fit's own
(MLIP4_ALGORITHM lists those that give mlip-4's algorithm). Both take --balance_species (default true), which gives every species of a structure an
equal share of its force weight. --ranks sets the thread count, by default the CPUs this process may run on. The potential may
be untrained, such as an mlip-3 template; the potential gains the species of the training set it lacks. The
trained potential is written to --save_path (or --save_to, mlp's name), by default over the input,
with an active set built on the training set.
"""

from __future__ import annotations

import logging
import os
import sys

from ._core import FitStructure, error_table, potential_class, read_potential
from .io_cfg import read_cfg
from .selection import converge_active_set, stored_selection, write_active_set

logger = logging.getLogger(__name__)


def errors(potential_path: str, structures: list, title: str, first: int = 1, species: list | None = None) -> str:
    """The errors of the potential in *potential_path* on the labelled *structures*, numbered from *first*, as the table training logs, under *title*."""
    potential = read_potential(potential_path)
    cutoff = potential.get_max_cutoff()
    table = error_table(potential, [FitStructure(atoms, cutoff, first + k) for k, atoms in enumerate(structures)], species or [], title)
    return table


# The fit options that give mlp's algorithm in place of the fit's defaults, for runs compared with mlp.
MLP_ALGORITHM = {"varpro": "false", "riemannian": "false", "reg_once": "false", "reg_at_main": "false", "no_orthogonalize": "false", "ldlt": "false", "reg_follows_scaling": "false", "init_noise": "5", "balance_species": "false"}

# The fit options that give mlip-4's algorithm in place of the ETN fit's defaults, for runs compared with mlip-4.
MLIP4_ALGORITHM = {"init_random": "false", "init_species": "false", "varpro": "false", "reg_at_main": "false", "force_weight": "0.01", "balance_species": "false", "no_mindist_update": "true"}


def fit(potential_path: str, structures, options: dict, checkpoint=None, ranks: int | None = None, comm=None, inverse_hessian=None):
    """fit_mtp or fit_etn, as the file *potential_path* holds, on *ranks* threads, or with an mpi4py communicator over its ranks, each passing its share of *structures*.

    ranks           : threads; without comm, defaults to the CPUs this process may run on
    inverse_hessian : (n, n) float64 array BFGS starts from and leaves its final inverse Hessian in, as fit_mtp's; an MTP's only
    """
    extra = {} if inverse_hessian is None else {"inverse_hessian": inverse_hessian}
    if ranks is None: ranks = os.process_cpu_count() if comm is None else 1
    trained = potential_class(potential_path).fit(potential_path, structures, options, ranks, comm, checkpoint=checkpoint, **extra)
    return trained


def train(potential_path: str, training_set: list, save_path: str, options: dict | None = None, ranks: int | None = None, al_mode: str = "nbh", comm=None) -> None:
    """Train the potential in *potential_path* on *training_set* with MTPFit or ETNFit, and write it with an active set to the file *save_path*.

    The potential may be untrained; it gains the species of *training_set* it lacks.

    options  : the fit's options without the leading "--", e.g. {"iteration_limit": 300, "skip_preinit": True}:
               `mlp train`'s; for an MTP, also the fit's own, MLP_ALGORITHM for mlp's algorithm; for an ETN, mlip-4's
               training settings mlp lacks and the fit's own, MLIP4_ALGORITHM for mlip-4's algorithm; and "species", the species' names for the log's error tables, comma-separated
    ranks    : threads; defaults to the CPUs this process may run on. The fit depends on it only through rounding.
    al_mode  : selection weights when *potential_path* has no #MVS_v1.1 block, "cfg" or "nbh"
    comm     : an mpi4py communicator to train over its ranks in place of threads. Every rank passes the
               same *training_set*, and rank 0 writes *save_path*.
    """
    template = read_potential(potential_path)
    weights = stored_selection(potential_path, al_mode, template.get_coeff_count())[0]

    cutoff = template.get_max_cutoff()
    # Under MPI each rank fits its share, dealt round-robin as mlp and the threads deal them.
    rank, size = (0, 1) if comm is None else (comm.Get_rank(), comm.Get_size())
    structures = (FitStructure(training_set[i], cutoff, i + 1) for i in range(rank, len(training_set), size))
    options = {key: str(value) for key, value in (options or {}).items()}
    potential = fit(potential_path, structures, options, checkpoint=lambda potential: potential.write(save_path), ranks=ranks, comm=comm)

    # with iteration_limit 0 an untrained potential stays untrained, and is written without coefficients to select with
    if potential.is_trained():
        if rank == 0:
            try:
                potential.write(save_path)
            except Exception as e:
                logger.warning(f"cannot save the trained potential to {save_path}: {type(e).__name__}: {e}")
        active_set, _ = converge_active_set(potential, training_set, weights, comm=comm)

    if rank == 0:
        potential.write(save_path)
        if potential.is_trained(): write_active_set(save_path, weights, active_set, training_set)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    args = [a for a in argv if not a.startswith("--")]
    options = dict(a[2:].split("=", 1) if "=" in a else (a[2:], "true") for a in argv if a.startswith("--"))
    if len(args) != 2: raise SystemExit(__doc__)
    potential_path, training_set_path = args

    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s")
    ranks = int(options.pop("ranks")) if "ranks" in options else None
    al_mode = options.pop("al_mode", "nbh")
    save_path = options.pop("save_path", options.pop("save_to", potential_path))

    with open(training_set_path) as f:
        training_set = read_cfg(f)
    train(potential_path, training_set, save_path, options=options, ranks=ranks, al_mode=al_mode)
    logger.info("training complete")


if __name__ == "__main__":
    main()
