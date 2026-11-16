"""The MPI side of test_otf_pool, run in its work directory: pyKMC's manager on rank 0, its workers on the others.

    mpirun -n 5 python tests/otf_pool_worker.py SESSION_SIZE ITERATIONS CANDIDATES SPECIES...

The work directory holds potential.almtp, train.cfg and dump.lammps, which the test wrote before mpirun.
"""

import json
import shutil
import sys
from pathlib import Path

import numpy
from ase.calculators.singlepoint import SinglePointCalculator
from mpi4py import MPI
from pykmc.manager import ManagerFactory

from otf_engine import NestedLauncher, otf_pool
from otf_engine._core import PotentialCalculator
from otf_engine._mtp import PairMTPGrad
from otf_engine.otf_cycle import OTFCycle

session_size, iterations, candidate_count, species = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4:]

extra_ops = {name: getattr(otf_pool, name) for name in ("update_active_set", "grade_dump", "select_add", "train_potential")}
manager = ManagerFactory(obj_factory=lambda comm, mode: None, session_size=session_size, comm=MPI.COMM_WORLD, extra_ops=extra_ops).launch()
if manager is None: sys.exit(0)    # a worker, once the manager has shut the pool down

shutil.copy("potential.almtp", "mpi_updated.almtp")
manager.global_update_active_set(potential="mpi_updated.almtp", training_set="train.cfg", species=species)
graded = manager.grade_dump(dump="dump.lammps", potential="potential.almtp", species=species).result()
grades = [atoms.info["features"]["MV_grade"] for atoms in graded]
numpy.save("mpi_grades.npy", grades)
candidates = [graded[i] for i in numpy.argsort(grades)[::-1][:candidate_count]]
selected = manager.global_select_add(potential="mpi_updated.almtp", training_set="train.cfg", candidates=candidates, species=species)
numpy.save("mpi_selected.npy", [next(i for i, c in enumerate(candidates) if numpy.array_equal(c.positions, s.positions)) for s in selected])
manager.global_train_potential(potential="potential.almtp", training_set="train.cfg", save_to="mpi.almtp", species=species, settings={"iteration_limit": iterations, "log": "mpi_train.log"})


def evaluator(structure):
    atoms = structure.copy()
    atoms.calc = PotentialCalculator(PairMTPGrad("potential.almtp"))
    results = dict(energy=atoms.get_potential_energy(), forces=atoms.get_forces(), stress=atoms.get_stress())
    atoms.calc = SinglePointCalculator(atoms, **results)
    return atoms


# cycle_0 is ok; cycle_1 is given a missing dump. Each is stepped through as pyKMC's controller steps it.
outcomes = {}
for name, dumps in [("ok", ["dump.lammps"]), ("fail", ["dump.lammps", "missing.lammps"])]:
    try:
        with OTFCycle(dumps, NestedLauncher(), potential="potential.almtp", training_set="train.cfg", species=species, iteration_limit=iterations, evaluator_fn=evaluator) as cycle:
            manager.global_update_active_set(**cycle.active_set_update)
            futures = {dump: manager.grade_dump(dump=dump, potential=cycle.potential, species=cycle.species) for dump in dumps}
            chosen = cycle.preselect({dump: future.result() for dump, future in futures.items()})
            cycle.evaluate(manager.global_select_add(**cycle.selection(chosen)))
            manager.global_train_potential(**cycle.training)
            cycle.replace_potential()
        outcomes[name] = "returned"
    except Exception as e:
        outcomes[name] = f"raised {type(e).__name__}" + (" on missing.lammps" if "missing.lammps" in str(e) else "")

manager.shutdown()
Path("outcomes.json").write_text(json.dumps(outcomes))
