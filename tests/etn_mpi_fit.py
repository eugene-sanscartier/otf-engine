"""A case of test_etn_trainer fitted through train.fit over MPI.COMM_WORLD; rank 0 saves the coefficients.

    mpirun -n 2 python tests/etn_mpi_fit.py CASE WORKDIR
"""

import sys
from pathlib import Path

import numpy
from mpi4py import MPI

sys.path.insert(0, str(Path(__file__).resolve().parent))

from conftest import read_structures    # noqa: E402
from test_etn_trainer import case_inputs, trainer_options    # noqa: E402
from otf_engine.train import fit    # noqa: E402

name, workdir = sys.argv[1], Path(sys.argv[2])
comm = MPI.COMM_WORLD
path, settings, labelled = case_inputs(name, read_structures("structures.cfg"), workdir)
trained = fit(str(path), [s for _, s in labelled[comm.Get_rank() :: comm.Get_size()]], trainer_options(settings, str(workdir / f"{name}.log")), comm=comm, balance_species=False)
if comm.Get_rank() == 0: numpy.save(workdir / f"{name}-coeffs.npy", trained.get_coeffs())
