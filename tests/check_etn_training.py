"""Training gate: otf_engine's ETN trainer against mlip-4's Train, in seconds.

Fits the untrained potentials of check_etn_kernels, whose coefficients start at mlip-4's
sin(10 + k^2), to structures 0 (no stress) and 1 of tmp/set.cfg with train_etn, and compares
with the fits of mlip-4's Train stored in etn_reference/training.npz:

  skip_pretrain  the 3-core potential on both structures, without pre-training
  pretrain       the 3-core potential on structure 1, with its 75-step pre-training, compared over
                 its first 7 steps: from losses near 1e87, rounding moves its line searches, and
                 mlip-4's JIT-compiled code already takes another path than its VM at step 7
  imaginary      the 5-core potential, whose middle cores have imaginary parts, on both structures
  threads        skip_pretrain on one thread per structure, against mlip-4 on one process
  mpi            skip_pretrain through train.fit under mpirun, one rank per structure

A run fails when its log, numbers aside, differs from mlip-4's; when a logged loss differs from
mlip-4's by more than --tol relatively, mlip-4's log holding 6 significant digits; or when the
trained potential's energy, forces, virial, input vectors and bond vectors on structure 1 differ
by more than --tensor-tol relatively from those of mlip-4's trained coefficients, both evaluated
by PairETN, which check_etn_kernels holds to mlip-4's own evaluation.

    python tests/check_etn_training.py
        check this build against the stored fits

    python tests/check_etn_training.py --capture --cfg tmp/set.cfg --mlip4-build DIR
        rerun mlip-4's fits through etn_reference/mlip4_train.cpp, compiled against the build
        directory DIR that holds lib_mlip_4_interface.a, and store them
"""

import argparse
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

import check_etn_kernels as kernels
from otf_engine._etn import ETNTraining, train_etn
from otf_engine._core import neighbors
from otf_engine.train import training_structure

PROBE_SOURCE = kernels.REFERENCE / "mlip4_train.cpp"
GOLDEN = kernels.REFERENCE / "training.npz"
STRUCTURES = [0, 1]
SETTINGS = {"max_iter_count": 1000, "max_func_eval": 10000, "linstop": 5e-4, "rescale": True, "pretrain": True, "wolfe_c1": 0.1, "wolfe_c2": 0.5, "bfgs_max_iters_no_improve": 10}
WEIGHTS = {"energy_weight": 1.0, "force_weight": 0.01, "stress_weight": 0.001}
# name: potential, settings, structures, ranks (one, a thread per structure, or an MPI rank per structure), the
# mlip-4 fit it is compared with, and the steps compared (None for all, and the trained potential with them)
CASES = {
    "skip_pretrain": ("etn-3", {"pretrain": False, "max_iter_count": 10}, [0, 1], "one", "skip_pretrain", None),
    "pretrain": ("etn-3", {"pretrain": True, "max_iter_count": 3}, [1], "one", "pretrain", 7),
    "imaginary": ("etn-5", {"pretrain": False, "max_iter_count": 10}, [0, 1], "one", "imaginary", None),
    "threads": ("etn-3", {"pretrain": False, "max_iter_count": 10}, [0, 1], "threads", "skip_pretrain", None),
    "mpi": ("etn-3", {"pretrain": False, "max_iter_count": 10}, [0, 1], "mpi", "skip_pretrain", None),
}
STEP = re.compile(r"^(\d+)((\t[-+]?(nan|inf)|\t[-+0-9.eE]+)+)")
# the trainer's lines, not mlip-4's: its error tables and timing; and the probe's params
TRAINER_LINE = re.compile(r"^Training set errors|^ +n +MAE |\d\.\d\de[-+]\d\d +\d\.\d\de[-+]\d\d|^max (energy/atom|force) error: |^Trained in |^@ params ")


def parse_log(text):
    """The losses of each logged step, and the other lines of the log mlip-4 writes too."""
    steps, lines = [], []
    for line in text.splitlines():
        line = line.removeprefix("[MLIP] ")
        if TRAINER_LINE.search(line):
            continue
        match = STEP.match(line)
        if match:
            steps += [[float(x) for x in match.group(2).split("\t")[1:]]]
            line = line[match.end():]
        if line.strip():
            lines += [line.strip()]
    return steps, lines


def structures(golden):
    """The stored structures, labelled with their reference energy, forces and virial."""
    built = []
    for s in range(len(STRUCTURES)):
        atoms = Atoms(numbers=numpy.full(len(golden[f"structure/{s}/types"]), 1), positions=golden[f"structure/{s}/positions"], cell=golden[f"structure/{s}/cell"], pbc=True)
        atoms.arrays["type_index"] = numpy.asarray(golden[f"structure/{s}/types"], dtype=numpy.int32)
        results = {"energy": float(golden[f"structure/{s}/energy"]), "forces": golden[f"structure/{s}/forces"]}
        if f"structure/{s}/stress" in golden:
            results["stress"] = golden[f"structure/{s}/stress"]
        atoms.calc = SinglePointCalculator(atoms, **results)
        built += [atoms]
    return built


def write_training_file(path, pot, labelled, settings):
    """mlip4_train's input: shape and params of pot, the weights and settings, and the labelled structures."""
    ints = lambda values: " ".join(map(str, values))
    floats = lambda values: " ".join(repr(float(x)) for x in values)
    lines = [f"{pot.get_species_count()} {len(pot.get_species_ranks())} {pot.get_core_count()} {pot.get_radial_basis_size()} {pot.get_min_cutoff()!r} {pot.get_max_cutoff()!r}"]
    lines += [ints(pot.get_species_ranks()), ints(pot.get_radial_funcs_counts()), ints(pot.get_radial_ranks())]
    lines += [f"{len(ranks)} {ints(ranks)}" for ranks in pot.get_bond_ranks()]
    coeffs = pot.get_coeffs()
    lines += [str(len(coeffs)), floats(coeffs), floats(WEIGHTS.values())]
    lines += [" ".join(str(int(v)) if isinstance(v, bool) else repr(v) for v in settings.values())]
    lines += [str(len(labelled))]
    for atoms, (_, energy, forces, virial, _) in labelled:
        lines += [str(len(atoms))] + [floats(row) for row in atoms.cell.array]
        lines += [f"{t} {floats(p)}" for t, p in zip(atoms.arrays["type_index"], atoms.positions)]
        lines += [floats([energy]), "0" if virial is None else f"1 {floats(virial)}"] + [floats(row) for row in forces]
    Path(path).write_text("\n".join(lines) + "\n")


def case_inputs(name, cells, workdir):
    """The untrained potential's file, the case's settings, and its labelled structures."""
    potential, overrides, chosen, _, _, _ = CASES[name]
    template = Path(workdir) / f"{potential}.etn"
    template.write_text(kernels.template(potential))
    settings = dict(SETTINGS, **overrides)
    labelled = [(cells[k], training_structure(cells[k], ETNTraining(str(template)).get_max_cutoff(), k + 1)) for k in chosen]
    return template, settings, labelled


def tensors(pot, coeffs, atoms):
    """Energy, forces, virial, input and bond vectors of pot at coeffs on atoms."""
    pot.set_coeffs(coeffs)
    result = pot.compute(neighbors(atoms, pot.get_max_cutoff()))
    out = {key: numpy.asarray(result[key]) for key in ("energy", "forces", "virials", "inputs")}
    for b, (re_part, im_part) in enumerate(zip(result["bonds"], result["bonds_im"])):
        out[f"bonds/{b}"], out[f"bonds_im/{b}"] = numpy.asarray(re_part), numpy.asarray(im_part)
    return out


def run_case(name, golden, cells, workdir, tol, tensor_tol):
    """Fits one case; returns its failures and a one-line account."""
    _, _, _, ranks, reference, steps = CASES[name]
    template, settings, labelled = case_inputs(name, cells, workdir)
    log_file = Path(workdir) / f"{name}.log"
    options = {key: str(value) for key, value in dict(settings, **WEIGHTS).items()} | {"log": str(log_file)}
    start = time.perf_counter()
    if ranks == "mpi":
        # train.fit over mpirun's ranks, each passing its share; rank 0 saves the coefficients
        subprocess.run(["mpirun", "-n", str(len(labelled)), sys.executable, str(Path(__file__).resolve()), "--mpi-worker", name, workdir], check=True)
        trained = ETNTraining(str(template))
        trained.set_coeffs(numpy.load(Path(workdir) / f"{name}-coeffs.npy"))
    else:
        trained = train_etn(str(template), [s for _, s in labelled], options, len(labelled) if ranks == "threads" else 1, balance_species=False)
    elapsed = time.perf_counter() - start

    failures = []
    mlip4_lines = list(golden[f"{reference}/lines"])
    mlip4_counts = list(golden[f"{reference}/counts"][:steps])
    b = golden[f"{reference}/losses"][: sum(mlip4_counts)]
    ours_steps, ours_lines = parse_log(log_file.read_text())
    ours_steps = ours_steps[:steps]
    if ours_lines != mlip4_lines:
        failures += [f"{name}: log lines differ: {[l for l in ours_lines if l not in mlip4_lines][:3]} vs {[l for l in mlip4_lines if l not in ours_lines][:3]}"]
    if [len(s) for s in ours_steps] != mlip4_counts:
        failures += [f"{name}: evaluations per step differ: {[len(s) for s in ours_steps]} vs {mlip4_counts}"]
    a = numpy.concatenate(ours_steps)
    count = min(len(a), len(b))
    a, b = a[:count], b[:count]
    if not numpy.array_equal(numpy.isfinite(a), numpy.isfinite(b)):
        failures += [f"{name}: non-finite losses at {numpy.flatnonzero(~numpy.isfinite(a))[:5]} vs mlip-4's at {numpy.flatnonzero(~numpy.isfinite(b))[:5]}"]
    finite = numpy.isfinite(a) & numpy.isfinite(b)
    rel = numpy.abs(a[finite] - b[finite]) / numpy.abs(b[finite])
    worst = float(rel.max())
    if worst > tol:
        failures += [f"{name}: loss {int(numpy.flatnonzero(finite)[rel.argmax()])} of {count} differs by {worst:.2e}"]

    tensor_account = "not compared"
    if steps is None:
        ours_tensors = tensors(trained, trained.get_coeffs(), cells[1])
        mlip4_tensors = tensors(trained, golden[f"{reference}/params"], cells[1])
        tensor_worst = max(float(numpy.abs(ours_tensors[k] - mlip4_tensors[k]).max()) / max(float(numpy.abs(mlip4_tensors[k]).max()), 1e-300) for k in mlip4_tensors)
        tensor_account = f"{tensor_worst:.2e}"
        if tensor_worst > tensor_tol:
            failures += [f"{name}: trained tensors differ by {tensor_worst:.2e}"]

    account = f"{name:14s} {len(mlip4_counts):3d} steps, {count:4d} losses, worst {worst:.2e}, trained tensors {tensor_account}, last loss {b[-1]:.6e}  ({elapsed:.1f} s)"
    return failures, account


def mpi_worker(name, workdir):
    """A case's fit through train.fit over MPI.COMM_WORLD, the coefficients saved by rank 0."""
    from mpi4py import MPI

    from otf_engine.train import fit

    comm = MPI.COMM_WORLD
    template, settings, labelled = case_inputs(name, structures(numpy.load(GOLDEN)), workdir)
    options = {key: str(value) for key, value in dict(settings, **WEIGHTS).items()} | {"log": str(Path(workdir) / f"{name}.log")}
    trained = fit(str(template), [s for _, s in labelled[comm.Get_rank() :: comm.Get_size()]], options, comm=comm, balance_species=False)
    if comm.Get_rank() == 0: numpy.save(Path(workdir) / f"{name}-coeffs.npy", trained.get_coeffs())


def capture(args):
    from otf_engine.io_cfg import read_cfg

    with open(args.cfg) as f:
        configs = read_cfg(f)
    golden = {}
    for s, k in enumerate(STRUCTURES):
        atoms = configs[k]
        golden[f"structure/{s}/positions"] = atoms.positions
        golden[f"structure/{s}/cell"] = atoms.cell.array
        golden[f"structure/{s}/types"] = atoms.arrays["type_index"]
        golden[f"structure/{s}/energy"] = atoms.calc.results["energy"]
        golden[f"structure/{s}/forces"] = atoms.calc.results["forces"]
        if "stress" in atoms.calc.results:
            golden[f"structure/{s}/stress"] = atoms.calc.results["stress"]
    cells = structures(golden)

    with tempfile.TemporaryDirectory() as workdir:
        probe = kernels.compile_probe(args.mlip4_build, workdir, PROBE_SOURCE)
        for name in {case[4] for case in CASES.values()}:
            template, settings, labelled = case_inputs(name, cells, workdir)
            training_file = Path(workdir) / f"{name}.txt"
            write_training_file(training_file, ETNTraining(str(template)), labelled, settings)
            start = time.perf_counter()
            printed = subprocess.run([str(probe), str(training_file)], check=True, capture_output=True, text=True).stdout
            steps, lines = parse_log(printed)
            golden[f"{name}/losses"] = numpy.concatenate(steps)
            golden[f"{name}/counts"] = numpy.array([len(s) for s in steps])
            golden[f"{name}/lines"] = numpy.array(lines)
            golden[f"{name}/params"] = numpy.array([float(x) for x in next(line for line in printed.splitlines() if line.startswith("@ params")).split(" :")[1].split()])
            print(f"{name}: {len(steps)} steps in {time.perf_counter() - start:.1f} s")

    numpy.savez_compressed(GOLDEN, **golden)
    print(f"captured {len(golden)} arrays to {GOLDEN} ({GOLDEN.stat().st_size / 1024:.0f} KB)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--capture", action="store_true", help="rerun mlip-4's fits and store them")
    parser.add_argument("--cfg", type=Path, help="cfg holding the structures, for the capture")
    parser.add_argument("--mlip4-build", type=Path, help="mlip-4 build directory, for the capture")
    parser.add_argument("--cases", nargs="+", default=list(CASES), choices=list(CASES))
    parser.add_argument("--tol", type=float, default=1e-5, help="relative tolerance on logged losses, which mlip-4 writes to 6 digits (default 1e-5)")
    parser.add_argument("--tensor-tol", type=float, default=1e-6, help="relative tolerance on the trained tensors (default 1e-6)")
    parser.add_argument("--mpi-worker", nargs=2, metavar=("CASE", "WORKDIR"), help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.mpi_worker:
        mpi_worker(*args.mpi_worker)
        return 0
    if args.capture:
        if not args.cfg or not args.mlip4_build:
            parser.error("--capture needs --cfg and --mlip4-build")
        capture(args)
        return 0

    golden = numpy.load(GOLDEN)
    cells = structures(golden)
    failures = []
    with tempfile.TemporaryDirectory() as workdir:
        for name in args.cases:
            case_failures, account = run_case(name, golden, cells, workdir, args.tol, args.tensor_tol)
            print(account, flush=True)
            failures += case_failures

    for line in failures:
        print(f"  FAIL {line}")
    print("every fit matches mlip-4" if not failures else f"{len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
