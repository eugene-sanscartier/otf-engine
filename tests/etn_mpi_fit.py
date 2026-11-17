"""A case of test_etn_fit fitted over MPI.COMM_WORLD; rank 0 saves the coefficients and the losses of each step.

    mpirun -n 2 python tests/etn_mpi_fit.py CASE WORKDIR
"""

import sys
from pathlib import Path

import numpy
from mpi4py import MPI

sys.path.insert(0, str(Path(__file__).resolve().parent))

from conftest import read_structures    # noqa: E402
from test_etn_fit import case_inputs, fit_options    # noqa: E402
from otf_engine._etn._etn_testing import fit_etn_steps    # noqa: E402

name, workdir = sys.argv[1], Path(sys.argv[2])
comm = MPI.COMM_WORLD
path, settings, labelled = case_inputs(name, read_structures("structures.cfg"), workdir)
trained, steps = fit_etn_steps(str(path), [s for _, s in labelled[comm.Get_rank() :: comm.Get_size()]], fit_options(settings), comm=comm)
if comm.Get_rank() == 0: numpy.savez(workdir / f"{name}-fit.npz", coeffs=trained.get_coeffs(), losses=numpy.concatenate(steps), counts=[len(s) for s in steps])
