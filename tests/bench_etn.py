"""Benchmark of the ETN against mlip-4: evaluation, a loss gradient, and a training run.

On the first N structures of a cfg, for the 3-core and the 5-core potentials of test_etn_kernels
(etn_reference/etn-3.etn and etn-5.etn), on one thread, it reports in ms:

  eval       energy, forces and virial of one structure: PairETN.compute on a built neighbor list,
             and with the list's build; mlip-4's CalcCfg::Run, which builds its own neighbors
  loss grad  the loss and its gradient over the N structures: PairETNGrad's compute and
             eval_loss_grad per structure; mlip-4's LossFunction::AccumulateGrads
  train      train_etn from the untrained potential, without pre-training, for -i iterations;
             mlip-4's Train through etn_reference/mlip4_train.cpp, its input's reading included

mlip-4 runs in its VM and in its JIT-compiled code. The JIT's compilation is left out of eval and
loss grad, and counted in train.

    python tests/bench_etn.py --cfg tmp/set.cfg --mlip4-build DIR
"""

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy

import etn_potentials as kernels
import test_etn_trainer as training
from conftest import ETN_REFERENCE
from otf_engine._etn import PairETNGrad, train_etn
from otf_engine._core import neighbors
from otf_engine.io_cfg import read_cfg
from otf_engine._core import Sample

BENCH_SOURCE = ETN_REFERENCE / "mlip4_bench.cpp"


def ours(pot, labelled, repeats):
    """Our eval (without and with the neighbor build) and loss-gradient times, in ms."""
    lists = [s.neighbors for _, s in labelled]
    start = time.perf_counter()
    for _ in range(repeats):
        for nl in lists:
            pot.compute(nl)
    eval_ms = (time.perf_counter() - start) * 1e3 / (repeats * len(lists))

    start = time.perf_counter()
    for _ in range(repeats):
        for atoms, _ in labelled:
            pot.compute(neighbors(atoms, pot.get_max_cutoff()))
    eval_build_ms = (time.perf_counter() - start) * 1e3 / (repeats * len(lists))

    weights = training.WEIGHTS
    start = time.perf_counter()
    for _ in range(repeats):
        for _, s in labelled:
            nl, energy, forces, virial = s.neighbors, s.energy, s.forces, s.virial
            result = pot.compute(nl)
            dforces = 2 * weights["force_weight"] * (result["forces"] - forces)
            dvirial = None
            if virial is not None:
                dvirial = 2 * weights["stress_weight"] * (result["virials"] - virial) * numpy.array([1, 1, 1, 2, 2, 2])
            pot.eval_loss_grad(nl, 2 * weights["energy_weight"] * (result["energy"] - energy), dforces, dvirial)
    loss_grad_ms = (time.perf_counter() - start) * 1e3 / repeats
    return eval_ms, eval_build_ms, loss_grad_ms


def mlip4(bench, training_file, jit, repeats):
    """mlip-4's eval and loss-gradient times, in ms."""
    printed = subprocess.run([str(bench), str(training_file), str(int(jit)), str(repeats)], check=True, capture_output=True, text=True).stdout
    values = dict(line[2:].split() for line in printed.splitlines() if line.startswith("@ "))
    return float(values["eval_ms"]), float(values["loss_grad_ms"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cfg", required=True, type=Path)
    parser.add_argument("--mlip4-build", required=True, type=Path, help="mlip-4 build directory, holding lib_mlip_4_interface.a")
    parser.add_argument("-n", "--structures", type=int, default=4, help="the first structures of the cfg (default 4)")
    parser.add_argument("-i", "--iterations", type=int, default=20, help="training iterations (default 20)")
    parser.add_argument("--repeats", type=int, default=5, help="runs averaged for eval and loss grad (default 5)")
    parser.add_argument("--potentials", nargs="+", default=["etn-3", "etn-5"])
    args = parser.parse_args()

    with open(args.cfg) as f:
        configs = read_cfg(f)[: args.structures]
    print(f"{len(configs)} structures of {len(configs[0])} atoms, one thread")

    with tempfile.TemporaryDirectory() as workdir:
        bench = kernels.compile_probe(args.mlip4_build, workdir, BENCH_SOURCE)
        trainer = kernels.compile_probe(args.mlip4_build, workdir, ETN_REFERENCE / "mlip4_train.cpp")
        for name in args.potentials:
            pot = PairETNGrad(str(ETN_REFERENCE / f"{name}.etn"))
            labelled = [(atoms, Sample(atoms, pot.get_max_cutoff(), k + 1)) for k, atoms in enumerate(configs)]
            settings = dict(training.SETTINGS, pretrain=False, max_iter_count=args.iterations)
            training_file = Path(workdir) / f"{name}.txt"
            training.write_training_file(training_file, pot, labelled, settings)

            eval_ms, eval_build_ms, loss_grad_ms = ours(pot, labelled, args.repeats)
            vm_eval, vm_loss = mlip4(bench, training_file, False, args.repeats)
            jit_eval, jit_loss = mlip4(bench, training_file, True, args.repeats)
            print(f"\n{name} ({pot.get_coeff_count()} coefficients)")
            print(f"  eval        ours {eval_ms:9.2f} ({eval_build_ms:.2f} with neighbors)   mlip-4 VM {vm_eval:9.2f}   JIT {jit_eval:9.2f}")
            print(f"  loss grad   ours {loss_grad_ms:9.2f}                       mlip-4 VM {vm_loss:9.2f}   JIT {jit_loss:9.2f}")

            template = Path(workdir) / f"{name}-template.etn"
            template.write_text(kernels.template(name))
            untrained = PairETNGrad(str(template))
            training.write_training_file(training_file, untrained, labelled, settings)
            options = training.trainer_options(settings, "none")
            start = time.perf_counter()
            train_etn(str(template), [s for _, s in labelled], options, 1, balance_species=False)
            ours_train = time.perf_counter() - start
            times = []
            for jit in (False, True):
                start = time.perf_counter()
                subprocess.run([str(trainer), str(training_file), str(int(jit))], check=True, capture_output=True)
                times += [time.perf_counter() - start]
            print(f"  train {args.iterations:3d}   ours {ours_train * 1e3:9.0f}                       mlip-4 VM {times[0] * 1e3:9.0f}   JIT {times[1] * 1e3:9.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
