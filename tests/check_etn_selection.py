"""Selection gate: the ETN's active learning in otf_engine.selection against mlip-4's CfgCalc, in seconds.

On the 3-core potential of check_etn_kernels (etn_reference/etn-3.etn), in nbh and cfg modes,
compares with mlip-4's CfgCalc, run once and stored in etn_reference/selection.npz:

  active set   update_active_set from no active set on the first 30 structures of tmp/set.cfg:
               the structure and equation of each row of A, or an equal equation of another
               structure, which rounding may pick in its place, and log |det A|
  grades       calculate_grade on the candidates, structures 30, 31, 2410 and 17763, training
               structure 5, and structure 0 rattled by 0.1 A: each candidate's grade and, in nbh mode, its
               per-atom grades
  selection    select_add of the candidates: the selected ones, exactly, and log |det A| after

then trains the untrained 3-core potential with train.train, which writes its active set
into the .etn, and grades the training structures with the written file, below mlip-4's
threshold_swap. A run fails when a number differs from mlip-4's by more than --tol relatively.

    python tests/check_etn_selection.py
        check this build against the stored selections

    python tests/check_etn_selection.py --capture --cfg tmp/set.cfg --mlip4-build DIR
        rerun mlip-4's CfgCalc through etn_reference/mlip4_selection.cpp, compiled against the
        build directory DIR that holds lib_mlip_4_interface.a, and store it
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

import check_etn_kernels as kernels
from otf_engine._etn import ETNTraining, PairETN
from otf_engine._core import PotentialCalculator
from otf_engine._core import neighbors
from otf_engine.almtp_io import read_mvs_state
from otf_engine.selection import calculate_grade, select_add, update_active_set
from otf_engine.train import train

PROBE_SOURCE = kernels.REFERENCE / "mlip4_selection.cpp"
GOLDEN = kernels.REFERENCE / "selection.npz"
POTENTIAL = "etn-3"
TRAINING = list(range(30))
CANDIDATES = [30, 31, 2410, 17763, 5]    # 5 is a training structure, which grades below threshold
RATTLED = 0       # a candidate rattled from this structure
RATTLE = 0.1      # A, seeded
MODES = ["nbh", "cfg"]
THRESHOLD_SWAP = 1.00001
EQUAL_EQUATIONS = 1e-12    # relative difference under which two equations are one


def structures(golden):
    """The stored training structures and candidates, with their species as type_index."""
    built = []
    for s in range(len(TRAINING) + len(CANDIDATES) + 1):
        atoms = Atoms(numbers=numpy.full(len(golden[f"structure/{s}/types"]), 1), positions=golden[f"structure/{s}/positions"], cell=golden[f"structure/{s}/cell"], pbc=True)
        atoms.arrays["type_index"] = numpy.asarray(golden[f"structure/{s}/types"], dtype=numpy.int32)
        built += [atoms]
    return built[: len(TRAINING)], built[len(TRAINING) :]


def write_selection_file(path, pot, mode, training, candidates):
    """mlip4_selection's input: shape and params of pot, the mode, the training structures and the candidates."""
    ints = lambda values: " ".join(map(str, values))
    floats = lambda values: " ".join(repr(float(x)) for x in values)
    lines = [f"{pot.get_species_count()} {len(pot.get_species_ranks())} {pot.get_core_count()} {pot.get_radial_basis_size()} {pot.get_min_cutoff()!r} {pot.get_max_cutoff()!r}"]
    lines += [ints(pot.get_species_ranks()), ints(pot.get_radial_func_counts()), ints(pot.get_radial_ranks())]
    lines += [f"{len(ranks)} {ints(ranks)}" for ranks in pot.get_bond_ranks()]
    coeffs = kernels.params(pot)
    lines += [str(len(coeffs)), floats(coeffs), str(int(mode == "nbh"))]
    for group in (training, candidates):
        lines += [str(len(group))]
        for atoms in group:
            lines += [str(len(atoms))] + [floats(row) for row in atoms.cell.array]
            lines += [f"{t} {floats(p)}" for t, p in zip(atoms.arrays["type_index"], atoms.positions)]
    Path(path).write_text("\n".join(lines) + "\n")


def parse_probe(printed):
    """The probe's "@ name ... : values" lines, as name -> list of floats; grades keyed "grade/c"."""
    out = {}
    for line in printed.splitlines():
        if not line.startswith("@ "): continue
        head, values = line[2:].split(" :") if " :" in line else (line[2:].rsplit(" ", 1)[0], line.rsplit(" ", 1)[1])
        words = head.split()
        key = f"grade/{words[1]}" if words[0] == "grade" else words[0]
        out[key] = [float(x) for x in values.split()]
    return out


def active_rows(path, training):
    """(structure, equation) of each row of A in the active set of the .etn at path, -1 -1 for an unset row."""
    state = read_mvs_state(path)
    key = lambda atoms: numpy.round(atoms.positions, 6).tobytes()
    index_of = {key(atoms): s for s, atoms in enumerate(training)}
    labels = [index_of[key(cfg)] for cfg in state.selected_cfgs]
    rows = numpy.array([[labels[c], e] if c >= 0 else [-1, -1] for c, e in zip(state.active_cfg_indices, state.active_eqn_indices)])
    return rows, state.A


def equation(pot, training, structure, eqn):
    """Equation eqn of a training structure: the energy over N (0) or a site energy's gradient (1 + 3 n + 9 + i)."""
    if structure < 0: return numpy.zeros(pot.get_coeff_count())
    atoms = training[structure]
    grads = pot.eval_grad(neighbors(atoms, pot.get_max_cutoff()))
    row = grads.sum(axis=0) / len(atoms) if eqn == 0 else grads[eqn - (1 + 3 * len(atoms) + 9)]
    return row


def log_det(A):
    """log |det A| in base n, as mlip-4 keeps it."""
    log_abs = numpy.linalg.slogdet(A)[1]
    return log_abs / numpy.log(len(A))


def relative(a, b):
    a, b = numpy.atleast_1d(numpy.asarray(a, dtype=float)), numpy.atleast_1d(numpy.asarray(b, dtype=float))
    return float(numpy.max(numpy.abs(a - b)) / max(numpy.max(numpy.abs(b)), 1e-300))


def run_mode(mode, golden, training, candidates, workdir, tol):
    """One mode's comparisons; returns its failures and a one-line account."""
    start = time.perf_counter()
    failures, worst = [], 0.0

    def compare(what, ours, theirs):
        nonlocal worst
        error = relative(ours, theirs)
        worst = max(worst, error)
        if error > tol: failures.append(f"{mode} {what}: {error:.2e} relative")

    path = str(Path(workdir) / f"{POTENTIAL}-{mode}.etn")
    shutil.copy(kernels.REFERENCE / f"{POTENTIAL}.etn", path)
    update_active_set(path, training, al_mode=mode)
    rows, A = active_rows(path, training)
    theirs = numpy.asarray(golden[f"{mode}/active_set"], dtype=int).reshape(-1, 2)
    # Atoms of the same environment in two structures give one equation twice, and rounding picks either.
    pot = ETNTraining(path)
    mismatched = [k for k in numpy.nonzero(numpy.any(rows != theirs, axis=1))[0] if relative(equation(pot, training, *rows[k]), equation(pot, training, *theirs[k])) > EQUAL_EQUATIONS]
    if mismatched: failures.append(f"{mode} active set: {len(mismatched)} of {len(rows)} rows hold another equation than mlip-4's")
    compare("log det", log_det(A), golden[f"{mode}/log_det"])

    graded = calculate_grade(path, [atoms.copy() for atoms in candidates])
    for c, atoms in enumerate(graded):
        theirs = golden[f"{mode}/grade/{c}"]
        compare(f"grade of candidate {c}", atoms.info["features"]["MV_grade"], theirs[0])
        if mode == "nbh": compare(f"per-atom grades of candidate {c}", atoms.arrays["nbh_grades"], theirs[1:])

    selected, (_, A_after, _) = select_add(path, training, candidates)
    ours = sorted(next(c for c, atoms in enumerate(candidates) if atoms is s) for s in selected)
    theirs = sorted(int(c) for c in golden[f"{mode}/selected"])
    if ours != theirs: failures.append(f"{mode} selection: {ours}, mlip-4 {theirs}")
    compare("log det after selection", log_det(A_after), golden[f"{mode}/log_det_after"])

    account = f"{mode}  {int(numpy.sum(rows[:, 0] >= 0))} active rows, selected {ours}, worst {worst:.2e}  ({time.perf_counter() - start:.1f} s)"
    return failures, account


def run_training(training, workdir):
    """train on an untrained .etn, then the grades of the training structures with the written file."""
    start = time.perf_counter()
    template = Path(workdir) / f"{POTENTIAL}-template.etn"
    template.write_text(kernels.template(POTENTIAL))
    trained = str(Path(workdir) / f"{POTENTIAL}-trained.etn")
    # references to fit: the stored potential's energies and forces
    reference = PotentialCalculator(PairETN(str(kernels.REFERENCE / f"{POTENTIAL}.etn")))
    labelled = []
    for atoms in training[:4]:
        reference.calculate(atoms, ["energy", "forces"])
        labelled += [atoms.copy()]
        labelled[-1].calc = SinglePointCalculator(labelled[-1], energy=reference.results["energy"], forces=reference.results["forces"])
    train(str(template), labelled, trained, settings={"iteration_limit": 3, "skip_preinit": True, "log": "none"}, ranks=2)

    failures = []
    if not PairETN(trained).is_trained(): failures.append("train: the written potential is untrained")
    grades = [atoms.info["features"]["MV_grade"] for atoms in calculate_grade(trained, [atoms.copy() for atoms in labelled])]
    if max(grades) > THRESHOLD_SWAP: failures.append(f"train: a training structure grades {max(grades):.6f} against the written active set")
    return failures, f"train   wrote an active set, training grades at most {max(grades):.6f}  ({time.perf_counter() - start:.1f} s)"


def capture(args):
    from otf_engine.io_cfg import read_cfg

    with open(args.cfg) as f:
        configs = read_cfg(f)
    rattled = configs[RATTLED].copy()
    rattled.positions += numpy.random.default_rng(0).uniform(-RATTLE, RATTLE, rattled.positions.shape)
    chosen = [configs[k] for k in TRAINING + CANDIDATES] + [rattled]
    golden = {}
    for s, atoms in enumerate(chosen):
        golden[f"structure/{s}/positions"] = atoms.positions
        golden[f"structure/{s}/cell"] = atoms.cell.array
        golden[f"structure/{s}/types"] = atoms.arrays["type_index"]
    training, candidates = structures(golden)

    pot = PairETN(str(kernels.REFERENCE / f"{POTENTIAL}.etn"))
    with tempfile.TemporaryDirectory() as workdir:
        probe = kernels.compile_probe(args.mlip4_build, workdir, PROBE_SOURCE)
        for mode in MODES:
            selection_file = Path(workdir) / f"{mode}.txt"
            write_selection_file(selection_file, pot, mode, training, candidates)
            start = time.perf_counter()
            printed = subprocess.run([str(probe), str(selection_file)], check=True, capture_output=True, text=True).stdout
            for key, values in parse_probe(printed).items():
                golden[f"{mode}/{key}"] = numpy.array(values)
            print(f"{mode}: selected {golden[f'{mode}/selected'].astype(int).tolist()} in {time.perf_counter() - start:.1f} s")

    numpy.savez_compressed(GOLDEN, **golden)
    print(f"captured {len(golden)} arrays to {GOLDEN} ({GOLDEN.stat().st_size / 1024:.0f} KB)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--capture", action="store_true", help="rerun mlip-4's CfgCalc and store it")
    parser.add_argument("--cfg", type=Path, help="cfg holding the structures, for the capture")
    parser.add_argument("--mlip4-build", type=Path, help="mlip-4 build directory, for the capture")
    parser.add_argument("--tol", type=float, default=1e-8, help="relative tolerance on grades and log det, which A's 1e-6 unset rows amplify rounding into: 3e-9 at most (default 1e-8)")
    args = parser.parse_args()

    if args.capture:
        if not args.cfg or not args.mlip4_build:
            parser.error("--capture needs --cfg and --mlip4-build")
        capture(args)
        return 0

    golden = numpy.load(GOLDEN)
    training, candidates = structures(golden)
    failures = []
    with tempfile.TemporaryDirectory() as workdir:
        for mode in MODES:
            mode_failures, account = run_mode(mode, golden, training, candidates, workdir, args.tol)
            print(account, flush=True)
            failures += mode_failures
        training_failures, account = run_training(training, workdir)
        print(account, flush=True)
        failures += training_failures

    for line in failures:
        print(f"  FAIL {line}")
    print("every selection matches mlip-4" if not failures else f"{len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
