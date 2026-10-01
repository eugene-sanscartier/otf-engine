"""Gate for otf_pool: its operations on pyKMC's worker pool against the same steps in one process, and the cycle through them.

Under `mpirun -n R+1`, pyKMC's ManagerFactory puts the manager on rank 0 and R workers in sessions
of --session-size ranks, with otf_pool's operations as extra_ops. On one dump of MD output and the
first N structures of a cfg:
  update_active_set as a global operation, must leave the active set of mtp_backend.update_active_set
                   in this process: the same active equations, and A to 1e-12 relatively
  grade_dump       submitted to a session, must give the grades of otf_mtp.grade_dump
  select_add       as a global operation, offered the dump's 24 highest-graded structures against
                   the active set update_active_set left, must select those mtp_backend.select_add
                   selects in this process
  train_potential  as a global operation, must log the losses mtp_backend.train logs on R threads,
                   to --tol relatively, and leave the active set update_active_set finds on one
                   rank from the same fit
  OTFCycle         stepped through as pyKMC steps it, with all four, must archive a cycle as ok.
                   Given a missing dump, it must raise on the manager and archive the cycle as
                   failed, and no rank may hang.
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
CANDIDATES = 24  # the dump's highest-graded structures, offered to select_add


def worker(args):
    """The MPI side, run in the work directory: pyKMC's manager on rank 0, its workers on the others."""
    from ase.calculators.singlepoint import SinglePointCalculator
    from mpi4py import MPI
    from pykmc.manager import ManagerFactory
    from otf_engine import NestedLauncher
    from otf_engine._mtp import MTPCalculator
    from otf_engine.otf_mtp import OTFCycle
    from otf_engine import otf_pool

    extra_ops = {name: getattr(otf_pool, name) for name in ("update_active_set", "grade_dump", "select_add", "train_potential")}
    manager = ManagerFactory(obj_factory=lambda comm, mode: None, session_size=args.session_size, comm=MPI.COMM_WORLD, extra_ops=extra_ops).launch()
    if manager is None: return 0    # a worker, once the manager has shut the pool down

    shutil.copy("potential.almtp", "mpi_updated.almtp")
    manager.global_update_active_set(potential="mpi_updated.almtp", training_set="train.cfg", species=args.species)
    graded = manager.grade_dump(dump="dump.lammps", potential="potential.almtp", species=args.species).result()
    grades = [atoms.info["features"]["MV_grade"] for atoms in graded]
    numpy.save("mpi_grades.npy", grades)
    candidates = [graded[i] for i in numpy.argsort(grades)[::-1][:CANDIDATES]]
    selected = manager.global_select_add(potential="mpi_updated.almtp", training_set="train.cfg", candidates=candidates, species=args.species)
    numpy.save("mpi_selected.npy", [next(i for i, c in enumerate(candidates) if numpy.array_equal(c.positions, s.positions)) for s in selected])
    manager.global_train_potential(potential="potential.almtp", training_set="train.cfg", save_to="mpi.almtp", species=args.species, settings={"iteration_limit": args.iterations, "log": "mpi_train.log"})

    def evaluator(structure):
        atoms = structure.copy()
        atoms.calc = MTPCalculator(args.potential)
        results = dict(energy=atoms.get_potential_energy(), forces=atoms.get_forces(), stress=atoms.get_stress())
        atoms.calc = SinglePointCalculator(atoms, **results)
        return atoms

    # cycle_0 is ok; cycle_1 is given a missing dump. Each is stepped through as pyKMC's controller steps it.
    outcomes = {}
    for name, dumps in [("ok", ["dump.lammps"]), ("fail", ["dump.lammps", "missing.lammps"])]:
        try:
            with OTFCycle(dumps, NestedLauncher(), potential="potential.almtp", training_set="train.cfg", species=args.species, iteration_limit=args.iterations, evaluator_fn=evaluator) as cycle:
                manager.global_update_active_set(**cycle.active_set_update)
                futures = {dump: manager.grade_dump(dump=dump, potential=cycle.potential, species=cycle.species) for dump in dumps}
                candidates = cycle.preselect({dump: future.result() for dump, future in futures.items()})
                cycle.evaluate(manager.global_select_add(**cycle.selection(candidates)))
                manager.global_train_potential(**cycle.training)
                cycle.replace_potential()
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

    from otf_engine._mtp import MTPTraining, write_mtp
    from otf_engine.almtp_io import read_mvs_header, read_mvs_state
    from otf_engine.io_cfg import read_cfg
    from otf_engine.mtp_backend import select_add, train, update_active_set
    from otf_engine.otf_mtp import grade_dump

    def same_active_set(mpi_set, serial_set):
        """Whether the workers' active set is the one-rank search's: the same active equations, and A to 1e-12 relatively."""
        dA = float(numpy.abs(mpi_set.A - serial_set.A).max() / numpy.abs(serial_set.A).max())
        same = numpy.array_equal(mpi_set.active_cfg_indices, serial_set.active_cfg_indices) and numpy.array_equal(mpi_set.active_eqn_indices, serial_set.active_eqn_indices)
        return same and dA <= 1e-12, f"{len(mpi_set.selected_cfgs)} active structures against {len(serial_set.selected_cfgs)} on one rank, equations {'the same' if same else 'DIFFER'}, A {dA:.1e} apart"

    start = time.perf_counter()
    failures = 0
    with TemporaryDirectory() as tmp:
        work = Path(tmp)
        shutil.copy(args.potential, work / "potential.almtp")
        extract_cfgs(args.cfg, args.n_structures, work / "train.cfg")
        shutil.copy(args.dump, work / "dump.lammps")

        # The references, in this process
        with open(work / "train.cfg") as f:
            structs = read_cfg(f, args.species)
        shutil.copy(work / "potential.almtp", work / "serial_updated.almtp")
        update_active_set(str(work / "serial_updated.almtp"), structs)
        graded = grade_dump(str(work / "dump.lammps"), str(work / "potential.almtp"), args.species)
        grades = numpy.array([atoms.info["features"]["MV_grade"] for atoms in graded])
        candidates = [graded[i] for i in numpy.argsort(grades)[::-1][:CANDIDATES]]
        selected = [next(i for i, c in enumerate(candidates) if c is s) for s in select_add(str(work / "serial_updated.almtp"), structs, candidates)[0]]
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
            same = len(mpi_grades) == len(grades) and numpy.array_equal(mpi_grades, grades)
            failures += not same
            print(f"grade_dump       {len(mpi_grades):4d} grades against this process's {len(grades):4d} — {'identical' if same else 'DIFFER'}")

            ok, detail = same_active_set(read_mvs_state(str(work / "mpi_updated.almtp")), read_mvs_state(str(work / "serial_updated.almtp")))
            failures += not ok
            print(f"update_active_set {detail} — {'match' if ok else 'MISMATCH'}")

            mpi_selected = numpy.load(work / "mpi_selected.npy").tolist()
            ok = mpi_selected == selected
            failures += not ok
            print(f"select_add       candidates {mpi_selected} of {CANDIDATES} against {selected} in this process — {'match' if ok else 'MISMATCH'}")

            losses = {arm: numpy.array([float(m.group(2)) for m in BFGS_LINE.finditer((work / f"{arm}_train.log").read_text())]) for arm in ("threads", "mpi")}
            n = min(len(losses["threads"]), len(losses["mpi"]))
            rel = numpy.abs(losses["mpi"][:n] - losses["threads"][:n]) / numpy.abs(losses["threads"][:n])
            ok = n > 0 and len(losses["threads"]) == len(losses["mpi"]) and rel.max() <= args.tol
            failures += not ok
            print(f"train_potential  {n:4d} logged losses against {args.ranks} threads', max relative difference {rel.max() if n else float('nan'):.1e} — {'match' if ok else 'MISMATCH'}")

            # The fits differ in their linear coefficients (6e-6 relatively), so the reference searches the MPI fit on one rank.
            write_mtp(MTPTraining(str(work / "mpi.almtp")), str(work / "serial.almtp"))
            update_active_set(str(work / "serial.almtp"), structs, weights=read_mvs_header(str(work / "potential.almtp"))[0])
            ok, detail = same_active_set(read_mvs_state(str(work / "mpi.almtp")), read_mvs_state(str(work / "serial.almtp")))
            failures += not ok
            print(f"  its active set {detail} — {'match' if ok else 'MISMATCH'}")

            outcomes = json.loads((work / "outcomes.json").read_text())
            for index, (name, expected, status) in enumerate([("ok", "returned", "ok"), ("fail", "raised RuntimeError on missing.lammps", "failed")]):
                status_file = work / f"otf_cycles/cycle_{index}/status"
                archived = status_file.read_text().strip() if status_file.exists() else "nothing"
                ok = outcomes.get(name) == expected and archived == status
                failures += not ok
                print(f"OTFCycle {name:<5}   {outcomes.get(name)}, archived as {archived} — {'as expected' if ok else 'UNEXPECTED'}")

    print(f"{time.perf_counter() - start:.1f} s")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
