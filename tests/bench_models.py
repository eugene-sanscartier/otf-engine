"""Benchmark of the MTP against the ETN: test errors, evaluation time and training time, over a family of each.

`run NAME` trains one potential of the families from its untrained shape on the training part of a
random split of a cfg, on --ranks threads, then writes into OUT/NAME:

  trained.mtp|etn, fit.log   the fitted potential and the fit's log
  fit.json                   the fit's wall time, its step count (pre-training included) and their ratio
  errors.json                energy RMSE per atom and force RMSE per component, on the test and training parts
  timing-JOB.json            per structure, on one thread, over the first --timed test structures, in ms:
                             compute on a built neighbor list, compute with the list's build, and
                             compute + eval_loss_grad (one structure's share of a loss gradient)

A rerun skips the fit and the errors already written and adds a timing file, so reruns on other nodes
give the timings a spread. Every potential has the same radial cutoffs: min_dist 0.99 of the
shortest distance in the training part, max_dist 5 and 8 radial basis functions. The MTP keeps
its min_dist (no_mindist_update), as the ETN does. Each fit runs with its defaults.

`report` gathers OUT into a table, pairs each MTP with the ETN closest in coefficient count, and plots
the times against the test force RMSE and against the coefficient count into OUT/bench_models.svg.

    python tests/bench_models.py list
    python tests/bench_models.py run mtp-16 --cfg tmp/set.cfg --out tmp/bench_models --ranks 32
    python tests/bench_models.py report --out tmp/bench_models
"""

import argparse
import json
import os
import re
import socket
import sys
import time
from pathlib import Path

import numpy
from ase.neighborlist import neighbor_list
from tqdm import tqdm

from bench_mtp import rmse
from otf_engine._core import Sample, neighbors, read_potential
from otf_engine._etn import ETN
from otf_engine._mtp import MTP
from otf_engine.io_cfg import read_cfg
from otf_engine.train import fit

MAX_DIST, RADIAL_BASIS_SIZE = 5.0, 8
LEVELS = range(8, 30, 2)
# (lmax, cores, rank): chosen to span 200 to 3000 coefficients, with one shape of each lmax near each MTP level's count
ETN_SHAPES = [(1, 3, 3), (1, 3, 4), (1, 4, 4), (1, 4, 5), (1, 5, 6), (2, 3, 3), (2, 4, 3), (2, 3, 4), (2, 4, 4), (2, 5, 4), (2, 5, 5), (3, 3, 3), (3, 4, 3), (3, 3, 4), (3, 4, 4), (3, 3, 5), (3, 4, 5)]
# The fits' default loss weights, for the timed loss gradient
LOSS_WEIGHTS = {"energy_weight": 1.0, "force_weight": 0.01, "stress_weight": 0.001}
# A logged step: mlp's and the MTP fit's "BFGS iter N:" or "Riemann iter N:", or mlip-4's step number and its losses
STEP_LINE = re.compile(r"^(?:(?:BFGS|Riemann) iter +\d+:|\d+\t)")
BAR = dict(ascii=True, ncols=60, bar_format="{desc:<14}|{bar}| {n_fmt:>2}/{total_fmt:>2}")


def etn_shape(lmax, cores, rank):
    """The ETN shape of input degrees 0 to lmax, `cores` cores and channel count `rank`, falling by one per degree."""
    ranks = [max(1, rank - l) for l in range(lmax + 1)]
    shape = dict(species_ranks=[max(1, (rank + 1) // 2 - l // 2) for l in range(lmax + 1)], radial_funcs_counts=[min(RADIAL_BASIS_SIZE, r + 2) for r in ranks], radial_ranks=ranks, bond_ranks=[[1]] + [ranks] * (cores - 1) + [[1]])
    return shape


FAMILIES = {f"mtp-{level:02d}": ("mtp", dict(level=level)) for level in LEVELS} | {f"etn-l{l}-c{c}-r{r}": ("etn", etn_shape(l, c, r)) for l, c, r in ETN_SHAPES}


def untrained(name, min_dist, species_count=3):
    """The untrained potential *name* of the families, with the shared radial cutoffs."""
    model, shape = FAMILIES[name]
    radial = dict(species_count=species_count, min_dist=min_dist, max_dist=MAX_DIST, radial_basis_size=RADIAL_BASIS_SIZE)
    potential = MTP(**shape, **radial) if model == "mtp" else ETN(**shape, **radial)
    return potential


def split(cfg, n_train, n_test, seed):
    """The training and test parts of the structures of *cfg*, drawn at random without overlap."""
    with open(cfg) as f:
        structures = read_cfg(f)
    order = numpy.random.default_rng(seed).permutation(len(structures))
    parts = [structures[i] for i in order[:n_train]], [structures[i] for i in order[n_train : n_train + n_test]]
    return parts


def timings(potential, labelled, repeats, weights):
    """Per structure, in ms: compute on a built neighbor list, compute with its build, and compute + eval_loss_grad."""
    lists = [s.neighbors for _, s in labelled]
    start = time.perf_counter()
    for _ in range(repeats):
        for nl in lists:
            potential.compute(nl)
    eval_ms = (time.perf_counter() - start) * 1e3 / (repeats * len(lists))

    start = time.perf_counter()
    for _ in range(repeats):
        for atoms, _ in labelled:
            potential.compute(neighbors(atoms, potential.get_max_cutoff()))
    eval_build_ms = (time.perf_counter() - start) * 1e3 / (repeats * len(lists))

    start = time.perf_counter()
    for _ in range(repeats):
        for _, s in labelled:
            result = potential.compute(s.neighbors)
            dforces = 2 * weights["force_weight"] * (result["forces"] - s.forces)
            dvirial = None
            if s.virial is not None:
                dvirial = 2 * weights["stress_weight"] * (result["virial"] - s.virial) * numpy.array([1, 1, 1, 2, 2, 2])
            potential.eval_loss_grad(s.neighbors, 2 * weights["energy_weight"] * (result["energy"] - s.energy), dforces, dvirial)
    loss_grad_ms = (time.perf_counter() - start) * 1e3 / (repeats * len(lists))
    return eval_ms, eval_build_ms, loss_grad_ms


def run(name, args):
    """Fit *name* unless OUT/NAME holds its fit, write its errors unless written, and add a timing file."""
    workdir = args.out / name
    workdir.mkdir(parents=True, exist_ok=True)
    model = FAMILIES[name][0]
    trained = workdir / f"trained.{model}"
    train_set, test_set = split(args.cfg, args.train, args.test, args.seed)

    with tqdm(total=3, desc=name, **BAR) as bar:
        if not (workdir / "fit.json").exists():
            min_dist = 0.99 * min(neighbor_list("d", atoms, MAX_DIST).min() for atoms in train_set)
            template = workdir / f"untrained.{model}"
            untrained(name, min_dist).write(str(template))
            samples = [Sample(atoms, MAX_DIST, k + 1) for k, atoms in enumerate(train_set)]
            options = {"log": str(workdir / "fit.log")} | ({"no_mindist_update": "true"} if model == "mtp" else {})

            start = time.perf_counter()
            potential = fit(str(template), samples, options, ranks=args.ranks)
            seconds = time.perf_counter() - start

            potential.write(str(trained))
            with open(workdir / "fit.log") as f:
                steps = sum(1 for line in f if STEP_LINE.match(line))
            record = dict(fit_s=seconds, steps=steps, s_per_step=seconds / steps, ranks=args.ranks, min_dist=min_dist, coefficients=potential.get_coeff_count(), host=socket.gethostname())
            (workdir / "fit.json").write_text(json.dumps(record, indent=1))
        bar.update()

        if not (workdir / "errors.json").exists():
            test_e, test_f = rmse(trained, test_set)
            train_e, train_f = rmse(trained, train_set)
            record = dict(test_energy=test_e, test_force=test_f, train_energy=train_e, train_force=train_f, n_train=len(train_set), n_test=len(test_set), seed=args.seed)
            (workdir / "errors.json").write_text(json.dumps(record, indent=1))
        bar.update()

        potential = read_potential(str(trained))
        labelled = [(atoms, Sample(atoms, potential.get_max_cutoff(), k + 1)) for k, atoms in enumerate(test_set[: args.timed])]
        eval_ms, eval_build_ms, loss_grad_ms = timings(potential, labelled, args.repeats, LOSS_WEIGHTS)
        record = dict(eval_ms=eval_ms, eval_build_ms=eval_build_ms, loss_grad_ms=loss_grad_ms, atoms=float(numpy.mean([len(a) for a, _ in labelled])), structures=len(labelled), host=socket.gethostname())
        (workdir / f"timing-{os.environ.get('SLURM_JOB_ID', os.getpid())}.json").write_text(json.dumps(record, indent=1))
        bar.update()


def gather(out):
    """One row per potential of OUT with a fit and errors: its fit, errors, and the median, min and max of its timings."""
    rows = []
    for name in FAMILIES:
        workdir = out / name
        if not (workdir / "errors.json").exists(): continue
        row = dict(name=name, model=FAMILIES[name][0]) | json.loads((workdir / "fit.json").read_text()) | json.loads((workdir / "errors.json").read_text())
        times = [json.loads(path.read_text()) for path in sorted(workdir.glob("timing-*.json"))]
        for key in ("eval_ms", "eval_build_ms", "loss_grad_ms"):
            values = [t[key] for t in times]
            row |= {key: float(numpy.median(values)), key + "_min": min(values), key + "_max": max(values)}
        row |= dict(atoms=times[0]["atoms"], timings=len(times))
        rows += [row]
    return rows


def report(out):
    """Print the table and the MTP–ETN pairs at equal coefficient count, and plot into OUT/bench_models.svg."""
    import matplotlib.pyplot as plt

    rows = gather(out)
    header = f"{'potential':<16}{'coeffs':>7}{'E test':>9}{'F test':>9}{'F train':>9}{'eval ms':>10}{'µs/atom':>9}{'+ neigh':>9}{'grad ms':>10}{'steps':>7}{'s/step':>9}{'ranks':>6}{'timings':>8}"
    print("errors in meV/atom and meV/Å; times per structure on one thread, median of the timings; s/step of the fit")
    print(header)
    for r in rows:
        print(f"{r['name']:<16}{r['coefficients']:>7d}{r['test_energy'] * 1e3:>9.2f}{r['test_force'] * 1e3:>9.1f}{r['train_force'] * 1e3:>9.1f}{r['eval_ms']:>10.2f}{r['eval_ms'] * 1e3 / r['atoms']:>9.2f}{r['eval_build_ms']:>9.2f}{r['loss_grad_ms']:>10.2f}{r['steps']:>7d}{r['s_per_step']:>9.3f}{r['ranks']:>6d}{r['timings']:>8d}")

    print("\nequal coefficient count: each MTP and the ETN closest to it")
    etns = [r for r in rows if r["model"] == "etn"]
    for m in (r for r in rows if r["model"] == "mtp"):
        if not etns: break
        e = min(etns, key=lambda r: abs(numpy.log(r["coefficients"] / m["coefficients"])))
        print(f"{m['name']:<8} {m['coefficients']:5d}  {e['name']:<14} {e['coefficients']:5d}   F test {m['test_force'] * 1e3:6.1f} / {e['test_force'] * 1e3:6.1f} meV/Å   eval {m['eval_ms']:7.2f} / {e['eval_ms']:7.2f} ms   grad {m['loss_grad_ms']:7.2f} / {e['loss_grad_ms']:7.2f} ms   s/step {m['s_per_step']:7.3f} / {e['s_per_step']:7.3f}")

    figure, axes = plt.subplots(2, 3, figsize=(15, 9))
    columns = [("eval_ms", "evaluation, ms per structure"), ("loss_grad_ms", "loss gradient, ms per structure"), ("s_per_step", "fit, s per step")]
    for model in ("mtp", "etn"):
        part = [r for r in rows if r["model"] == model]
        force = numpy.array([r["test_force"] * 1e3 for r in part])
        coefficients = numpy.array([r["coefficients"] for r in part])
        for k, (key, label) in enumerate(columns):
            value = numpy.array([r[key] for r in part])
            errorbar = None
            if key + "_min" in part[0]:
                errorbar = numpy.array([[r[key] - r[key + "_min"] for r in part], [r[key + "_max"] - r[key] for r in part]])
            axes[0, k].errorbar(force, value, yerr=errorbar, fmt="o", label=model.upper())
            axes[1, k].errorbar(coefficients, value, yerr=errorbar, fmt="o", label=model.upper())
            for r, x, y in zip(part, force, value):
                axes[0, k].annotate(r["name"][4:], (x, y), fontsize=7)
            axes[0, k].set(xlabel="test force RMSE, meV/Å", ylabel=label, yscale="log")
            axes[1, k].set(xlabel="coefficients", ylabel=label, xscale="log", yscale="log")
    for ax in axes.flat:
        ax.legend()
    figure.tight_layout()
    figure.savefig(out / "bench_models.svg")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["list", "run", "report"])
    parser.add_argument("names", nargs="*", help="potentials of the families, for run")
    parser.add_argument("--cfg", type=Path, default=Path("tmp/set.cfg"))
    parser.add_argument("--out", type=Path, default=Path("tmp/bench_models"))
    parser.add_argument("--train", type=int, default=1000, help="training structures (default 1000)")
    parser.add_argument("--test", type=int, default=1000, help="test structures (default 1000)")
    parser.add_argument("--seed", type=int, default=0, help="seed of the split (default 0)")
    parser.add_argument("--ranks", type=int, default=os.process_cpu_count(), help="training threads (default the CPUs this process may run on)")
    parser.add_argument("--timed", type=int, default=50, help="test structures timed (default 50)")
    parser.add_argument("--repeats", type=int, default=3, help="passes averaged in a timing (default 3)")
    args = parser.parse_args()

    if args.command == "list":
        for name in FAMILIES:
            print(name, untrained(name, 1.0).get_coeff_count())
    elif args.command == "run":
        for name in args.names:
            run(name, args)
    else:
        report(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
