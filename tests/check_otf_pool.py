"""Gate for otf_pool: its operations on pyKMC's worker pool against the same steps in one process, and the cycle through them.

Under `mpirun -n R+1`, pyKMC's ManagerFactory puts the manager on rank 0 and R workers in sessions
of --session-size ranks, with otf_pool's operations as extra_ops. On one dump of MD output and the
first N structures of a cfg:
  grade_dump       submitted per dump through the sessions, must give the grades of
                   otf_mtp.grade_extrapolative_dumps on its process pool
  train_potential  as a global operation, must log the losses mtp_backend.train logs on R threads,
                   to --tol relatively
  run_cycle        with both, must archive a cycle as ok. Given a missing dump, it must raise on the
                   manager and archive the cycle as failed, and no rank may hang.
The references run in this process. The cycle's evaluator labels structures with the input potential.

    python tests/check_otf_pool.py --potential tmp/potential-16.almtp --cfg tmp/set.cfg --dump tmp/extrapolating_dump.6.lammps --species Ni Si H
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy

from bench_training import BFGS_LINE, extract_cfgs

TIMEOUT = 300    # s, so that a hung rank fails the gate instead of stalling it


def worker(args):
    """The MPI side, run in the work directory: pyKMC's manager on rank 0, its workers on the others."""
    from ase.calculators.singlepoint import SinglePointCalculator
    from mpi4py import MPI
    from pykmc.manager import ManagerFactory
    from otf_engine import NestedLauncher
    from otf_engine._mtp import MTPCalculator
    from otf_engine.otf_mtp import grade_extrapolative_dumps, run_cycle
    from otf_engine.otf_pool import grade_dump, train_potential

    manager = ManagerFactory(obj_factory=lambda comm, mode: None, session_size=args.session_size, comm=MPI.COMM_WORLD, extra_ops={"grade_dump": grade_dump, "train_potential": train_potential}).launch()
    if manager is None: return 0    # a worker, once the manager has shut the pool down

    graded = grade_extrapolative_dumps("potential.almtp", ["dump.lammps"], species=args.species, submit_grade=manager.grade_dump)
    numpy.save("mpi_grades.npy", [atoms.info["features"]["MV_grade"] for atoms in graded])
    manager.global_train_potential(potential="potential.almtp", training_set="train.cfg", save_to="mpi.almtp", species=args.species, settings={"iteration_limit": args.iterations, "log": "mpi_train.log"})

    def evaluator(structure):
        atoms = structure.copy()
        atoms.calc = MTPCalculator(args.potential)
        results = dict(energy=atoms.get_potential_energy(), forces=atoms.get_forces(), stress=atoms.get_stress())
        atoms.calc = SinglePointCalculator(atoms, **results)
        return atoms

    # cycle_0 is ok; cycle_1 is given a missing dump
    outcomes = {}
    for name, dumps in [("ok", ["dump.lammps"]), ("fail", ["dump.lammps", "missing.lammps"])]:
        try:
            run_cycle(dumps, NestedLauncher(), potential="potential.almtp", training_set="train.cfg", species=args.species, iteration_limit=args.iterations, evaluator_fn=evaluator, submit_grade=manager.grade_dump, train_potential=manager.global_train_potential)
            outcomes[name] = "returned"
        except Exception as e:
            outcomes[name] = f"raised {type(e).__name__}" + (" on missing.lammps" if "missing.lammps" in str(e) else "")

    manager.shutdown()
    Path("outcomes.json").write_text(json.dumps(outcomes))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--potential", required=True)
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--dump", required=True)
    parser.add_argument("--species", nargs="+", required=True)
    parser.add_argument("-n", "--n-structures", type=int, default=8, help="training structures (default 8)")
    parser.add_argument("--ranks", type=int, default=4, help="worker ranks, and threads for the reference (default 4)")
    parser.add_argument("--session-size", type=int, default=2, help="ranks per pyKMC session (default 2)")
    parser.add_argument("--iterations", type=int, default=2, help="training iterations (default 2)")
    parser.add_argument("--tol", type=float, default=1e-8, help="relative tolerance on logged losses (default 1e-8)")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker: return worker(args)

    from otf_engine.io_cfg import read_cfg
    from otf_engine.mtp_backend import train
    from otf_engine.otf_mtp import grade_extrapolative_dumps

    start = time.perf_counter()
    failures = 0
    with TemporaryDirectory() as tmp:
        work = Path(tmp)
        shutil.copy(args.potential, work / "potential.almtp")
        extract_cfgs(args.cfg, args.n_structures, work / "train.cfg")
        shutil.copy(args.dump, work / "dump.lammps")

        # The references, in this process
        pool_grades = numpy.array([atoms.info["features"]["MV_grade"] for atoms in grade_extrapolative_dumps(str(work / "potential.almtp"), [str(work / "dump.lammps")], species=args.species)])
        with open(work / "train.cfg") as f:
            structs = read_cfg(f, args.species)
        train(str(work / "potential.almtp"), structs, str(work / "threads.almtp"), settings={"iteration_limit": args.iterations, "log": str(work / "threads_train.log")}, ranks=args.ranks)

        cmd = ["mpirun", "-n", str(args.ranks + 1), sys.executable, str(Path(__file__).resolve()), "--worker", "--potential", str(Path(args.potential).resolve()), "--cfg", args.cfg, "--dump", args.dump,
               "--species", *args.species, "--iterations", str(args.iterations), "--session-size", str(args.session_size)]
        try:
            proc = subprocess.run(cmd, cwd=work, capture_output=True, text=True, timeout=TIMEOUT)
            code = proc.returncode
        except subprocess.TimeoutExpired:
            proc, code = None, "timed out"
        failures += code != 0
        print(f"mpirun -n {args.ranks + 1}: exit {code} — {'as expected' if code == 0 else 'UNEXPECTED'}")
        if code != 0 and proc is not None: print(proc.stdout[-3000:], proc.stderr[-3000:])

        if code == 0:
            mpi_grades = numpy.load(work / "mpi_grades.npy")
            same = len(mpi_grades) == len(pool_grades) and numpy.array_equal(mpi_grades, pool_grades)
            failures += not same
            print(f"grade_dump       {len(mpi_grades):4d} grades against the pool's {len(pool_grades):4d} — {'identical' if same else 'DIFFER'}")

            losses = {arm: numpy.array([float(m.group(2)) for m in BFGS_LINE.finditer((work / f"{arm}_train.log").read_text())]) for arm in ("threads", "mpi")}
            n = min(len(losses["threads"]), len(losses["mpi"]))
            rel = numpy.abs(losses["mpi"][:n] - losses["threads"][:n]) / numpy.abs(losses["threads"][:n])
            ok = n > 0 and len(losses["threads"]) == len(losses["mpi"]) and rel.max() <= args.tol
            failures += not ok
            print(f"train_potential  {n:4d} logged losses against {args.ranks} threads', max relative difference {rel.max() if n else float('nan'):.1e} — {'match' if ok else 'MISMATCH'}")

            outcomes = json.loads((work / "outcomes.json").read_text())
            for index, (name, expected, status) in enumerate([("ok", "returned", "ok"), ("fail", "raised RuntimeError on missing.lammps", "failed")]):
                status_file = work / f"otf_cycles/cycle_{index}/status"
                archived = status_file.read_text().strip() if status_file.exists() else "nothing"
                ok = outcomes.get(name) == expected and archived == status
                failures += not ok
                print(f"run_cycle {name:<5}  {outcomes.get(name)}, archived as {archived} — {'as expected' if ok else 'UNEXPECTED'}")

    print(f"{time.perf_counter() - start:.1f} s")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
