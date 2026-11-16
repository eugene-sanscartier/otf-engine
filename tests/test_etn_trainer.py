"""ETNTrainer against mlip-4's Train over short fits.

Fits the untrained potentials of test_etn_kernels, whose coefficients start at mlip-4's sin(10 + k^2), to structures 0
(no stress) and 1 with train_etn, and compares with mlip-4's fits, stored in etn_reference/training.npz:

  skip_pretrain  the 3-core potential on both structures, without pre-training
  pretrain       the 3-core potential on structure 1, with its 75-step pre-training, compared over its first 7 steps:
                 from losses near 1e87, rounding moves its line searches
  imaginary      the 5-core potential, whose middle cores have imaginary parts, on both structures
  threads        skip_pretrain on one thread per structure, against mlip-4 on one process
  mpi            skip_pretrain through train.fit under mpirun, one rank per structure

A fit fails when its log, numbers aside, differs from mlip-4's; when a logged loss differs from mlip-4's by more than
1e-5 relatively, mlip-4's log holding 6 significant digits; or when the trained potential's energy, forces, virial,
input and bond vectors on structure 1 differ by more than 1e-6 relatively from those of mlip-4's trained coefficients.
"""

import re
import subprocess
import sys
from pathlib import Path

import numpy
import pytest

from conftest import ETN_REFERENCE, StoredOutputs, TESTS
from etn_potentials import compile_probe, untrained_etn
from otf_engine._core import neighbors
from otf_engine._core import Sample

SETTINGS = {"max_iter_count": 1000, "max_func_eval": 10000, "linstop": 5e-4, "rescale": True, "pretrain": True, "wolfe_c1": 0.1, "wolfe_c2": 0.5, "bfgs_max_iters_no_improve": 10}
WEIGHTS = {"energy_weight": 1.0, "force_weight": 0.01, "stress_weight": 0.001}
MLP_NAMES = {"max_iter_count": "iteration_limit", "linstop": "tolerance"}    # train_etn's names of mlip-4's settings
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
NUMBER = re.compile(r"[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?")
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


def trainer_options(settings, log):
    """mlip-4's *settings* and the weights as train_etn takes them: strings, under mlp's names where mlp has the
    setting, pretrain inverted as skip_preinit, and the log target, without balance_species, which mlip-4 lacks."""
    options = {MLP_NAMES.get(key, key): str(value) for key, value in dict(settings, **WEIGHTS).items() if key != "pretrain"}
    options["skip_preinit"] = str(not settings["pretrain"])
    return options | {"log": log, "balance_species": "false"}


def case_inputs(name, structures, workdir):
    """The untrained potential's file, the case's settings, and its labelled structures.

    The file is written unless it exists: the MPI ranks of etn_mpi_fit.py read the one the test wrote, since a rank
    rewriting it under another's reading fails that rank and leaves the other waiting in a collective.
    """
    from otf_engine._etn import ETN

    potential, overrides, chosen, _, _, _ = CASES[name]
    path = Path(workdir) / f"{potential}.etn"
    if not path.exists(): untrained_etn(potential).write(path)
    settings = dict(SETTINGS, **overrides)
    labelled = [(structures[k], Sample(structures[k], ETN(str(path)).get_max_cutoff(), k + 1)) for k in chosen]
    return path, settings, labelled


def write_training_file(path, pot, labelled, settings):
    """mlip4_train's input: shape and params of pot, the weights and settings, and the labelled structures."""
    ints = lambda values: " ".join(map(str, values))
    floats = lambda values: " ".join(repr(float(x)) for x in values)
    lines = [f"{pot.get_species_count()} {len(pot.get_species_ranks())} {pot.get_core_count()} {pot.get_radial_basis_size()} {pot.get_min_cutoff()!r} {pot.get_max_cutoff()!r}"]
    lines += [ints(pot.get_species_ranks()), ints(pot.get_radial_func_counts()), ints(pot.get_radial_ranks())]
    lines += [f"{len(ranks)} {ints(ranks)}" for ranks in pot.get_bond_ranks()]
    coeffs = pot.get_coeffs()
    lines += [str(len(coeffs)), floats(coeffs), floats(WEIGHTS.values())]
    lines += [" ".join(str(int(v)) if isinstance(v, bool) else repr(v) for v in settings.values())]
    lines += [str(len(labelled))]
    for atoms, s in labelled:
        lines += [str(len(atoms))] + [floats(row) for row in atoms.cell.array]
        lines += [f"{t} {floats(p)}" for t, p in zip(atoms.arrays["type_index"], atoms.positions)]
        lines += [floats([s.energy]), "0" if s.virial is None else f"1 {floats(s.virial)}"] + [floats(row) for row in s.forces]
    Path(path).write_text("\n".join(lines) + "\n")


def tensors(pot, coeffs, atoms):
    """Energy, forces, virial, input and bond vectors of pot at coeffs on atoms."""
    pot.set_coeffs(coeffs)
    result = pot.compute(neighbors(atoms, pot.get_max_cutoff()))
    out = {key: numpy.asarray(result[key]) for key in ("energy", "forces", "virials", "inputs")}
    for b, (re_part, im_part) in enumerate(zip(result["bonds"], result["bonds_im"])):
        out[f"bonds/{b}"], out[f"bonds_im/{b}"] = numpy.asarray(re_part), numpy.asarray(im_part)
    return out


@pytest.fixture(scope="module")
def stored(recapture, mlip4_build, structures, tmp_path_factory):
    """mlip-4's fits, rerun first under --recapture."""
    from otf_engine._etn import ETN

    outputs = StoredOutputs(ETN_REFERENCE / "training.npz", recapture)
    if recapture:
        workdir = tmp_path_factory.mktemp("etn_training")
        probe = compile_probe(mlip4_build, workdir, ETN_REFERENCE / "mlip4_train.cpp")
        for name in sorted({case[4] for case in CASES.values()}):
            path, settings, labelled = case_inputs(name, structures, workdir)
            training_file = workdir / f"{name}.txt"
            write_training_file(training_file, ETN(str(path)), labelled, settings)
            printed = subprocess.run([str(probe), str(training_file)], check=True, capture_output=True, text=True).stdout
            steps, lines = parse_log(printed)
            outputs[f"{name}/losses"] = numpy.concatenate(steps)
            outputs[f"{name}/counts"] = numpy.array([len(s) for s in steps])
            outputs[f"{name}/lines"] = numpy.array(lines)
            outputs[f"{name}/params"] = numpy.array([float(x) for x in next(line for line in printed.splitlines() if line.startswith("@ params")).split(" :")[1].split()])
        outputs.save()
    return outputs


@pytest.mark.parametrize("name", [pytest.param(name, marks=pytest.mark.mpi) if name == "mpi" else name for name in CASES])
def test_fit(name, structures, stored, tmp_path):
    from otf_engine._etn import ETN, train_etn

    _, _, _, ranks, reference, steps = CASES[name]
    path, settings, labelled = case_inputs(name, structures, tmp_path)
    log_file = tmp_path / f"{name}.log"
    if ranks == "mpi":
        subprocess.run(["mpirun", "-n", str(len(labelled)), sys.executable, str(TESTS / "etn_mpi_fit.py"), name, str(tmp_path)], check=True, timeout=120)
        trained = ETN(str(path))
        trained.set_coeffs(numpy.load(tmp_path / f"{name}-coeffs.npy"))
    else:
        trained = train_etn(str(path), [s for _, s in labelled], trainer_options(settings, str(log_file)), len(labelled) if ranks == "threads" else 1)

    counts = list(stored[f"{reference}/counts"][:steps])
    b = stored[f"{reference}/losses"][: sum(counts)]
    ours_steps, ours_lines = parse_log(log_file.read_text())
    ours_steps = ours_steps[:steps]
    # the trainer writes the numbers of its warnings to 17 digits, mlip-4 to 6
    assert [NUMBER.sub("#", line) for line in ours_lines] == [NUMBER.sub("#", line) for line in stored[f"{reference}/lines"]]
    assert [len(s) for s in ours_steps] == counts
    a = numpy.concatenate(ours_steps)[: len(b)]
    numpy.testing.assert_array_equal(numpy.isfinite(a), numpy.isfinite(b))
    finite = numpy.isfinite(b)
    numpy.testing.assert_allclose(a[finite], b[finite], rtol=1e-5)

    if steps is None:
        ours, theirs = tensors(trained, trained.get_coeffs(), structures[1]), tensors(trained, stored[f"{reference}/params"], structures[1])
        for key in theirs:
            assert numpy.max(numpy.abs(ours[key] - theirs[key])) <= 1e-6 * max(numpy.max(numpy.abs(theirs[key])), 1e-300), key
