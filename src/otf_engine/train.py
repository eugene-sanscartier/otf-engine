"""Train an MTP or an ETN on ASE structures with MTPFit or ETNFit.

Usage:

    python -m otf_engine.train potential.almtp|potential.etn train.xyz|train.cfg [--save_path=FILE] [--ranks=N] [--al_mode=nbh|cfg] [fit options]

Options:
    fit options          --iteration_limit=300, --tolerance=1e-3, --skip_preinit, --energy_weight=1, ...
    MTP fit options      --varpro=false, --riemannian=false, ...; MLP_ALGORITHM lists those that give the mlp algorithm
    ETN fit options      --rescale=false, ...; MLIP4_ALGORITHM lists those that give the mlip-4 algorithm
    --balance_species    give every species of a structure an equal share of its force weight (default true)
    --ranks              thread count, by default the CPUs this process may run on
    --save_path          file of the trained potential with an active set built on the training set,
                         by default the input; --save_to is an alias
    --al_mode            selection weights when the potential has no #MVS_v1.1 block

The potential may be untrained. It gains the species of the training set that it lacks.
"""

from __future__ import annotations

import logging
import os
import sys

from ._core import FitStructure, error_table, potential_class, read_potential
from .selection import converge_active_set, stored_selection, write_active_set
from .training_set import read_training_set

logger = logging.getLogger(__name__)


def errors(potential_path: str, structures: list, title: str, first: int = 1) -> str:
    """Return the error table of the potential in *potential_path* on the labelled *structures*, under *title*.

    Args:
        first: Number of the first structure in the table.
    """
    potential = read_potential(potential_path)
    cutoff = potential.get_max_cutoff()
    species = potential.get_species()
    table = error_table(potential, [FitStructure(atoms, cutoff, first + k, species) for k, atoms in enumerate(structures)], species, title)
    return table


# Fit options that give the mlp algorithm in place of the MTP fit defaults.
MLP_ALGORITHM = {"varpro": "false", "riemannian": "false", "reg_once": "false", "reg_at_main": "false", "no_orthogonalize": "false", "ldlt": "false", "reg_follows_scaling": "false", "init_noise": "5", "balance_species": "false"}

# Fit options that give the mlip-4 algorithm in place of the ETN fit defaults.
MLIP4_ALGORITHM = {"init_random": "false", "init_species": "false", "varpro": "false", "reg_at_main": "false", "force_weight": "0.01", "balance_species": "false", "no_mindist_update": "true"}


def fit(potential_path: str, structures, options: dict, checkpoint=None, ranks: int | None = None, comm=None, inverse_hessian=None):
    """Fit the potential in *potential_path* with fit_mtp or fit_etn, by the model of the file.

    The fit runs on *ranks* threads, or over the ranks of the mpi4py
    communicator *comm*. Each rank passes its share of *structures*.

    Args:
        ranks: Thread count, by default the CPUs this process may run on without *comm*, and 1 with it.
        inverse_hessian: (n, n) float64 array that BFGS starts from and overwrites with its final inverse Hessian. MTP only.

    Returns:
        The trained potential.
    """
    extra = {} if inverse_hessian is None else {"inverse_hessian": inverse_hessian}
    if ranks is None: ranks = os.process_cpu_count() if comm is None else 1
    trained = potential_class(potential_path).fit(potential_path, structures, options, ranks, comm, checkpoint=checkpoint, **extra)
    return trained


def train(potential_path: str, training_set: list, save_path: str, options: dict | None = None, ranks: int | None = None, al_mode: str = "nbh", comm=None) -> None:
    """Train the potential in *potential_path* on *training_set* and write it with an active set to *save_path*.

    The potential may be untrained. It gains the species of *training_set* that it lacks.

    Args:
        options: Fit options without the leading "--", e.g. {"iteration_limit": 300, "skip_preinit": True}.
            MLP_ALGORITHM gives the mlp algorithm for an MTP, and MLIP4_ALGORITHM the mlip-4 algorithm for an ETN.
            The key "species" names the species in the error tables of the log, comma-separated.
        ranks: Thread count, by default the CPUs this process may run on. The fit depends on it only through rounding.
        al_mode: Selection weights, "cfg" or "nbh", when *potential_path* has no #MVS_v1.1 block.
        comm: mpi4py communicator to train over its ranks in place of threads. Every rank passes the same
            *training_set*, and rank 0 writes *save_path*.
    """
    template = read_potential(potential_path)
    weights = stored_selection(potential_path, al_mode, template.get_coeff_count())[0]
    rank, size = (0, 1) if comm is None else (comm.Get_rank(), comm.Get_size())

    species = template.get_species()
    for atoms in training_set:
        if "type_index" not in atoms.arrays: species += [element for element in dict.fromkeys(atoms.get_chemical_symbols()) if element not in species]
    if species and rank == 0: logger.info("species indices: " + ", ".join(f"{index}={element}" for index, element in enumerate(species)))

    cutoff = template.get_max_cutoff()
    structures = (FitStructure(training_set[i], cutoff, i + 1, species) for i in range(rank, len(training_set), size))
    options = {key: str(value) for key, value in (options or {}).items()}
    if species: options.setdefault("species", ",".join(species))
    potential = fit(potential_path, structures, options, checkpoint=lambda potential: potential.write(save_path), ranks=ranks, comm=comm)
    potential.set_species(species)

    # With iteration_limit 0 an untrained potential stays untrained, and its file gets no active set.
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

    training_set = read_training_set(training_set_path)
    train(potential_path, training_set, save_path, options=options, ranks=ranks, al_mode=al_mode)
    logger.info("training complete")


if __name__ == "__main__":
    main()
