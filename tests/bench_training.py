"""Training gate and benchmark: otf_engine's trainer against mlip-3's `mlp train`.

Trains the same starting potential on the first N structures of a cfg file with each arm and
reports, per run: wall time per BFGS iteration, time to the end of the nonlinear fit, total time
(the trainer's includes building neighbor lists and the active set), and the energy and force
RMSE of the trained potential under one shared evaluator.

Arms:
  mlp    `mlp train` under `mpirun -np R` (plain `mlp` for R = 1)
  port   `python -m otf_engine.train --ranks=R`

Exits non-zero when a port run's logged losses (`BFGS iter N: f=...`) differ from mlp's on one
rank by more than --tol, or stop at another line. mlp's result depends on R wherever R does not
divide N; the port's does not, so mlp at other R is timed but not compared. Keep -i at 30 or
less: rounding differences grow through the BFGS, and past about 45 steps they separate even
mlp on 1 rank from mlp on 4 ranks, which fit the same loss.

An untrained potential (an mlip-3 template) logs 75 pre-training steps first, over which
rounding separates even mlp's runs; tests/check_training.py gates the untrained paths.

    python tests/bench_training.py --potential tmp/potential-16.almtp --cfg tmp/set.cfg -n 20 -i 30 --ranks 1 8
"""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy

from otf_engine._mtp import MTPCalculator, PairMTP
from otf_engine.io_cfg import read_cfg

MLP = Path.home() / "Doctorat/code_library/mlip-3/bin/mlp"
BFGS_LINE = re.compile(r"BFGS iter (\d+): f=(\S+)")


def extract_cfgs(cfg_path, n_structures, out_path):
    """Copy the first n_structures BEGIN_CFG blocks of cfg_path to out_path."""
    count = 0
    with open(cfg_path) as src, open(out_path, "w") as dst:
        for line in src:
            dst.write(line)
            if line.strip() == "END_CFG":
                count += 1
                if count == n_structures:
                    break


def rmse(potential_path, structs):
    """Energy RMSE per atom and force RMSE per component of a potential on structs."""
    calc = MTPCalculator(str(potential_path))
    e_err, f_err = [], []
    for atoms in structs:
        result = calc.potential.compute(calc.neighbors(atoms))
        e_err += [(float(result["energy"]) - atoms.get_potential_energy()) / len(atoms)]
        f_err += [(numpy.asarray(result["forces"]) - atoms.get_forces()).ravel()]
    errors = (float(numpy.sqrt(numpy.mean(numpy.square(e_err)))), float(numpy.sqrt(numpy.mean(numpy.square(numpy.concatenate(f_err))))))
    return errors


def run_logged(cmd, workdir):
    """Run a trainer, timestamping each BFGS iteration line as it is flushed."""
    iters, train_end = [], None
    start = time.perf_counter()
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=workdir) as proc:
        for line in proc.stdout:
            now = time.perf_counter() - start
            match = BFGS_LINE.search(line)
            if match:
                iters += [(now, float(match.group(2)))]
            elif "MTPR training ended" in line:
                train_end = now
    if proc.returncode != 0: raise RuntimeError(f"{cmd[0]} exited with {proc.returncode}")

    result = dict(times=[t for t, _ in iters], losses=[f for _, f in iters], train_end=train_end, total=time.perf_counter() - start)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--potential", required=True, type=Path)
    parser.add_argument("--cfg", required=True, type=Path)
    parser.add_argument("-n", "--n-structures", type=int, default=20)
    parser.add_argument("-i", "--iterations", type=int, default=30)
    parser.add_argument("--ranks", type=int, nargs="+", default=[1])
    parser.add_argument("--arms", nargs="+", default=["mlp", "port"], choices=["mlp", "port"])
    parser.add_argument("--tol", type=float, default=1e-5, help="relative tolerance on logged losses (default 1e-5)")
    parser.add_argument("--options", nargs="+", default=[], help="further mlp train options for both arms, without the leading --, e.g. skip_preinit=true")
    args = parser.parse_args()
    potential = args.potential.resolve()

    with TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        train_cfg = workdir / "train.cfg"
        extract_cfgs(args.cfg, args.n_structures, train_cfg)
        with open(train_cfg) as f:
            structs = read_cfg(f)
        start = "untrained" if not PairMTP(str(potential)).is_trained() else "starting RMSE E {:.6e} eV/atom, F {:.6e} eV/A".format(*rmse(potential, structs))
        print(f"{len(structs)} structures, {sum(len(a) for a in structs)} atoms, {args.iterations} iterations; {start}")

        runs = {}
        for arm in args.arms:
            for ranks in args.ranks:
                out = workdir / f"{arm}_{ranks}.almtp"
                options = [f"--save_to={out}", f"--iteration_limit={args.iterations}"] + [f"--{option}" for option in args.options]
                if arm == "mlp":
                    cmd = ([] if ranks == 1 else ["mpirun", "-np", str(ranks)]) + [str(MLP), "train", str(potential), str(train_cfg)] + options
                else:
                    cmd = [sys.executable, "-m", "otf_engine.train", str(potential), str(train_cfg), f"--ranks={ranks}"] + options
                r = runs[arm, ranks] = run_logged(cmd, workdir)
                steps = len(r["times"]) - 1
                per_iter = (r["times"][-1] - r["times"][0]) / steps if steps > 0 else float("nan")
                e, f = rmse(out, structs)
                print(f"{arm:<4} ranks {ranks:<3d} iters {len(r['times']):4d}  f_end {r['losses'][-1]:.6e}  {per_iter:8.3f} s/iter  fit {r['train_end']:8.2f} s  total {r['total']:8.2f} s  RMSE E {e:.6e}  F {f:.6e}", flush=True)

        failures = 0
        if ("mlp", 1) in runs:
            mlp_f = numpy.array(runs["mlp", 1]["losses"])
            for ranks in args.ranks:
                if ("port", ranks) not in runs: continue
                port_f = numpy.array(runs["port", ranks]["losses"])
                n = min(len(mlp_f), len(port_f))
                rel = numpy.abs(port_f[:n] - mlp_f[:n]) / numpy.abs(mlp_f[:n])
                first = int(numpy.argmax(rel > args.tol)) if (rel > args.tol).any() else None
                ok = len(mlp_f) == len(port_f) and first is None
                failures += not ok
                where = "" if first is None else f", first beyond tolerance at iteration {first}"
                print(f"port ranks {ranks} vs mlp ranks 1: {n} logged losses, {int((rel == 0).sum())} identical, max relative difference {rel.max():.2e}{where} — {'match' if ok else 'MISMATCH'}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
