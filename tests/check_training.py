"""Fast training gate: otf_engine's trainer against mlip-3's `mlp train` on each way into a fit.

Fits mlip-3's level-8 template to the first structure and to the first 8 structures of a cfg,
with mlp on one rank and with the trainer on one rank and on one rank per structure:

  skip_preinit  the untrained template, without pre-training
  preinit       the untrained template, with its 75-step pre-training
  add_species   a two-species potential, fitted by mlp to the structures with species 2 renamed 1
  trained       mlp's pre-trained potential, fitted further
  init_random   the untrained template with random radial coefficients, run by the trainer only

The starting potentials come from mlp, so that they do not change with the trainer under test.

A run fails when its log, numbers aside, or its chosen scalings differ from mlp's, or when a logged
loss (`BFGS iter N: f=...`) differs from mlp's by more than --tol relatively. Pre-training starts
so ill-conditioned that rounding separates mlp on 2, 4 or 8 ranks from mlp on 1 rank by more than
1e-5 from step 30 to 38 on 8 structures, so its log is compared up to its --preinit-steps-th
logged loss only; over the first 15 the trainer stays within 1.4e-6 of mlp.

    python tests/check_training.py --cfg tmp/set.cfg
"""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy

from bench_training import BFGS_LINE, MLP, extract_cfgs
from otf_engine.io_cfg import read_cfg, write_cfg
from otf_engine.mtp_backend import train

TEMPLATE = MLP.parents[1] / "MTP_templates/08.almtp"
NUMBER = re.compile(r"[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?")
ITERATIONS = 30
PREINIT_ITERATIONS = 5


def fit_log(text):
    """The trainer's lines of a log, from the training-set count to the last rescaling."""
    lines = [line.strip() for line in text.splitlines()]
    start = next(i for i, line in enumerate(lines) if "configurations found" in line)
    end = max(i for i, line in enumerate(lines) if line.startswith("Rescaling to"))
    return lines[start:end + 1]


def compare(mlp_text, port_text, tol, steps=None):
    """Whether a trainer log matches mlp's up to its steps-th logged loss (all by default), and a one-line account of how closely."""
    a, b = fit_log(mlp_text), fit_log(port_text)
    if steps:
        a = a[:[i for i, line in enumerate(a) if BFGS_LINE.search(line)][steps - 1] + 1]
        b = b[:[i for i, line in enumerate(b) if BFGS_LINE.search(line)][steps - 1] + 1]
    same_shape = [NUMBER.sub("#", line) for line in a] == [NUMBER.sub("#", line) for line in b]
    scalings_a, scalings_b = [line for line in a if line.startswith("Rescaling to")], [line for line in b if line.startswith("Rescaling to")]
    same_scalings = sum(x == y for x, y in zip(scalings_a, scalings_b))

    fa = numpy.array([float(m.group(2)) for line in a if (m := BFGS_LINE.search(line))])
    fb = numpy.array([float(m.group(2)) for line in b if (m := BFGS_LINE.search(line))])
    n = min(len(fa), len(fb))
    rel = numpy.abs(fb[:n] - fa[:n]) / numpy.abs(fa[:n])
    first = int(numpy.argmax(rel > tol)) if (rel > tol).any() else None

    ok = same_shape and same_scalings == len(scalings_a) == len(scalings_b) and first is None
    account = f"{n:3d} losses compared, {int((rel == 0).sum()):3d} identical, max rel {rel.max():.1e}, first beyond tol {'-' if first is None else first:>3}, scalings {same_scalings:2d}/{len(scalings_a):2d}, shape {'same' if same_shape else 'DIFFERS'}"
    return ok, account


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cfg", required=True, type=Path)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 8], help="structure counts (default 1 8)")
    parser.add_argument("--tol", type=float, default=1e-5, help="relative tolerance on logged losses (default 1e-5)")
    parser.add_argument("--preinit-steps", type=int, default=15, help="pre-training losses compared (default 15)")
    args = parser.parse_args()

    failures = 0
    start_time = time.perf_counter()
    with TemporaryDirectory() as tmp:
        for size in args.sizes:
            workdir = Path(tmp) / f"n{size}"
            workdir.mkdir()
            cfg = workdir / "train.cfg"
            extract_cfgs(args.cfg, size, cfg)
            with open(cfg) as f:
                structs = read_cfg(f)
            ranks = sorted({1, size})

            def fit(arm, path, potential, options, r=1, data=cfg):
                """Start one fit; returns a function that waits for it and returns its log."""
                out = workdir / f"{path}_{arm}_{r}.almtp"
                log = out.with_suffix(".log")
                options = {"iteration_limit": ITERATIONS} | options
                if arm == "mlp":
                    with open(log, "w") as stream:
                        proc = subprocess.Popen([str(MLP), "train", str(potential), str(data), f"--save_to={out}"] + [f"--{k}={v}" for k, v in options.items()], stdout=stream, stderr=subprocess.STDOUT, cwd=workdir)
                    return lambda: (proc.wait(), log.read_text())[1]
                train(str(potential), structs, str(out), settings=options | {"log": str(log)}, ranks=r)
                return lambda: log.read_text()

            renamed = [atoms.copy() for atoms in structs]
            for atoms, original in zip(renamed, structs, strict=True):
                atoms.arrays["type_index"][atoms.arrays["type_index"] == 2] = 1
                atoms.calc = original.calc
            renamed_cfg = workdir / "renamed.cfg"
            with open(renamed_cfg, "w") as f:
                write_cfg(f, renamed)

            # mlp's starting potentials and untrained fits first, then each dependent fit once its start exists
            logs = {}
            two_species = fit("mlp", "two_species", TEMPLATE, {"skip_preinit": "true"}, data=renamed_cfg)
            paths = {"skip_preinit": (TEMPLATE, {"skip_preinit": "true"}), "preinit": (TEMPLATE, {"iteration_limit": PREINIT_ITERATIONS})}
            for path, (potential, options) in paths.items():
                logs[path, "mlp", 1] = fit("mlp", path, potential, options)
            for path, (potential, options) in paths.items():
                for r in ranks:
                    logs[path, "port", r] = fit("port", path, potential, options, r)

            for path, start, wait in [("add_species", workdir / "two_species_mlp_1.almtp", two_species), ("trained", workdir / "preinit_mlp_1.almtp", logs["preinit", "mlp", 1])]:
                wait()
                logs[path, "mlp", 1] = fit("mlp", path, start, {})
                for r in ranks:
                    logs[path, "port", r] = fit("port", path, start, {}, r)

            for path in [*paths, "add_species", "trained"]:
                mlp_log = logs[path, "mlp", 1]()
                for r in ranks:
                    ok, account = compare(mlp_log, logs[path, "port", r](), args.tol, args.preinit_steps if path == "preinit" else None)
                    failures += not ok
                    print(f"{size:2d} structures  {path:<12}  ranks {r:2d}: {account} — {'match' if ok else 'MISMATCH'}", flush=True)

            random_log = fit("port", "init_random", TEMPLATE, {"iteration_limit": PREINIT_ITERATIONS, "init_random": "true"})()
            ok = "Random initialization of radial coefficients" in random_log and "Pre-training ended" in random_log
            failures += not ok
            print(f"{size:2d} structures  {'init_random':<12}  ranks  1: {len(BFGS_LINE.findall(random_log)):3d} losses logged — {'ran' if ok else 'FAILED'}", flush=True)

    print(f"{time.perf_counter() - start_time:.1f} s")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
