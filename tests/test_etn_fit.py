"""ETNFit against mlip-4's Train over short fits.

Fits the untrained potentials of test_etn_kernels, whose coefficients start at mlip-4's sin(10 + k^2), to structures 0
(no stress) and 1 with fit_etn, and compares with mlip-4's fits, stored in etn_reference/training.npz:

  skip_pretrain  the 3-core potential on both structures, without pre-training
  pretrain       the 3-core potential on structure 1, with its 75-step pre-training, compared over its first 7 steps:
                 from losses near 1e87, rounding moves its line searches
  imaginary      the 5-core potential, whose middle cores have imaginary parts, on both structures
  threads        skip_pretrain on one thread per structure, against mlip-4 on one process
  mpi            skip_pretrain under mpirun, one rank per structure

A fit fails when its steps take other numbers of evaluations than mlip-4's; when the loss at an evaluation differs from
the one mlip-4 logs by more than 1e-5 relatively, mlip-4's log holding 6 significant digits; or when the trained
potential's energy, forces, virial, input and bond vectors on structure 1 differ by more than 1e-6 relatively from those
of mlip-4's trained coefficients.
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
from otf_engine._core import FitStructure

SETTINGS = {"max_iter_count": 1000, "max_func_eval": 10000, "linstop": 5e-4, "rescale": True, "pretrain": True, "wolfe_c1": 0.1, "wolfe_c2": 0.5, "bfgs_max_iters_no_improve": 10}
WEIGHTS = {"energy_weight": 1.0, "force_weight": 0.01, "stress_weight": 0.001}
MLP_NAMES = {"max_iter_count": "iteration_limit", "linstop": "tolerance"}    # fit_etn's names of mlip-4's settings
# name: potential, settings, structures, ranks (one, a thread per structure, or an MPI rank per structure), the
# mlip-4 fit it is compared with, and the steps compared (None for all, and the trained potential with them)
CASES = {
    "skip_pretrain": ("etn-3", {"pretrain": False, "max_iter_count": 10}, [0, 1], "one", "skip_pretrain", None),
    "pretrain": ("etn-3", {"pretrain": True, "max_iter_count": 3}, [1], "one", "pretrain", 7),
    "imaginary": ("etn-5", {"pretrain": False, "max_iter_count": 10}, [0, 1], "one", "imaginary", None),
    "threads": ("etn-3", {"pretrain": False, "max_iter_count": 10}, [0, 1], "threads", "skip_pretrain", None),
    "mpi": ("etn-3", {"pretrain": False, "max_iter_count": 10}, [0, 1], "mpi", "skip_pretrain", None),
}
STEP = re.compile(r"^(\d+)((\t[-+]?(nan|inf)|\t[-+0-9.eE]+)+)")    # mlip-4's step: its number, then the loss of each evaluation


def mlip4_steps(text):
    """The losses of each step mlip-4's log prints."""
    steps = [[float(x) for x in match.group(2).split("\t")[1:]] for line in text.splitlines() if (match := STEP.match(line.removeprefix("[MLIP] ")))]
    return steps


def fit_options(settings):
    """mlip-4's *settings* and the weights as fit_etn takes them: strings, under mlp's names where mlp has the
    setting, pretrain inverted as skip_preinit, over MLIP4_ALGORITHM, without a log."""
    from otf_engine.train import MLIP4_ALGORITHM

    options = {MLP_NAMES.get(key, key): str(value) for key, value in dict(settings, **WEIGHTS).items() if key != "pretrain"}
    options["skip_preinit"] = str(not settings["pretrain"])
    return MLIP4_ALGORITHM | options | {"log": "none"}


def case_inputs(name, structures, workdir):
    """The untrained potential's file, the case's settings, and its labelled structures.

    The file is written unless it exists: the MPI ranks of etn_mpi_fit.py read the one the test wrote, since a rank
    rewriting it under another's reading fails that rank and leaves the other waiting in a collective.
    """
    from otf_engine._etn import ETN

    potential_name, overrides, chosen, _, _, _ = CASES[name]
    path = Path(workdir) / f"{potential_name}.etn"
    if not path.exists(): untrained_etn(potential_name).write(path)
    settings = dict(SETTINGS, **overrides)
    labelled = [(structures[k], FitStructure(structures[k], ETN(str(path)).get_max_cutoff(), k + 1)) for k in chosen]
    return path, settings, labelled


def write_training_file(path, potential, labelled, settings):
    """mlip4_train's input: shape and params of potential, the weights and settings, and the labelled structures."""
    ints = lambda values: " ".join(map(str, values))
    floats = lambda values: " ".join(repr(float(x)) for x in values)
    lines = [f"{potential.get_species_count()} {len(potential.get_species_ranks())} {potential.get_core_count()} {potential.get_radial_basis_size()} {potential.get_min_cutoff()!r} {potential.get_max_cutoff()!r}"]
    lines += [ints(potential.get_species_ranks()), ints(potential.get_radial_func_counts()), ints(potential.get_radial_ranks())]
    lines += [f"{len(ranks)} {ints(ranks)}" for ranks in potential.get_bond_ranks()]
    coeffs = potential.get_coeffs()
    lines += [str(len(coeffs)), floats(coeffs), floats(WEIGHTS.values())]
    lines += [" ".join(str(int(v)) if isinstance(v, bool) else repr(v) for v in settings.values())]
    lines += [str(len(labelled))]
    for atoms, s in labelled:
        lines += [str(len(atoms))] + [floats(row) for row in atoms.cell.array]
        lines += [f"{t} {floats(p)}" for t, p in zip(atoms.arrays["type_index"], atoms.positions)]
        lines += [floats([s.energy]), "0" if s.virial is None else f"1 {floats(s.virial)}"] + [floats(row) for row in s.forces]
    Path(path).write_text("\n".join(lines) + "\n")


def tensors(potential, coeffs, atoms):
    """Energy, forces, virial, input and bond vectors of potential at coeffs on atoms."""
    potential.set_coeffs(coeffs)
    nl = neighbors(atoms, potential.get_max_cutoff())
    result, bonds = potential.compute(nl), potential.eval_bonds(nl)
    out = {key: numpy.asarray(result[key]) for key in ("energy", "forces", "virial")}
    out["inputs"] = numpy.asarray(bonds["inputs"])
    for b, (re_part, im_part) in enumerate(zip(bonds["bonds"], bonds["bonds_im"])):
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
            steps = mlip4_steps(printed)
            outputs[f"{name}/losses"] = numpy.concatenate(steps)
            outputs[f"{name}/counts"] = numpy.array([len(s) for s in steps])
            outputs[f"{name}/params"] = numpy.array([float(x) for x in next(line for line in printed.splitlines() if line.startswith("@ params")).split(" :")[1].split()])
        outputs.save()
    return outputs


@pytest.mark.parametrize("name", [pytest.param(name, marks=pytest.mark.mpi) if name == "mpi" else name for name in CASES])
def test_fit(name, structures, stored, tmp_path):
    from otf_engine._etn import ETN
    from otf_engine._etn._etn_testing import fit_etn_steps

    _, _, _, ranks, reference, steps = CASES[name]
    path, settings, labelled = case_inputs(name, structures, tmp_path)
    if ranks == "mpi":
        subprocess.run(["mpirun", "-n", str(len(labelled)), sys.executable, str(TESTS / "etn_mpi_fit.py"), name, str(tmp_path)], check=True, timeout=120)
        saved = numpy.load(tmp_path / f"{name}-fit.npz")
        trained = ETN(str(path))
        trained.set_coeffs(saved["coeffs"])
        ours_steps = numpy.split(saved["losses"], numpy.cumsum(saved["counts"])[:-1])
    else:
        trained, ours_steps = fit_etn_steps(str(path), [s for _, s in labelled], fit_options(settings), len(labelled) if ranks == "threads" else 1)

    counts = list(stored[f"{reference}/counts"][:steps])
    b = stored[f"{reference}/losses"][: sum(counts)]
    ours_steps = ours_steps[:steps]
    assert [len(s) for s in ours_steps] == counts
    a = numpy.concatenate(ours_steps)[: len(b)]
    numpy.testing.assert_array_equal(numpy.isfinite(a), numpy.isfinite(b))
    finite = numpy.isfinite(b)
    numpy.testing.assert_allclose(a[finite], b[finite], rtol=1e-5)

    if steps is None:
        ours, theirs = tensors(trained, trained.get_coeffs(), structures[1]), tensors(trained, stored[f"{reference}/params"], structures[1])
        for key in theirs:
            assert numpy.max(numpy.abs(ours[key] - theirs[key])) <= 1e-6 * max(numpy.max(numpy.abs(theirs[key])), 1e-300), key


RECIPE = {"energy_weight": "1.0", "force_weight": "0.1", "stress_weight": "0.001", "iteration_limit": "10", "skip_preinit": "true", "init_random": "true", "random_seed": "1", "init_scale": "0.3", "init_species": "true", "varpro": "true", "reg_at_main": "true", "log": "none"}


# The 6 structures 11 times over, so that varpro_fraction 0.5 draws the 32 structures a fraction needs on average
FRACTION_CHOSEN = list(range(6)) * 11


def varpro_fit(structures, workdir, chosen=(0, 1), **options):
    """The recipe's short fit of the 3-core potential on the chosen structures, numbered by position, from a seeded start:
    the trained potential, its steps' losses and the labelled structures."""
    from otf_engine._etn import ETN
    from otf_engine._etn._etn_testing import fit_etn_steps

    path = Path(workdir) / "etn-3.etn"
    untrained_etn("etn-3").write(path)
    labelled = [FitStructure(structures[k], ETN(str(path)).get_max_cutoff(), n + 1) for n, k in enumerate(chosen)]
    trained, steps = fit_etn_steps(str(path), labelled, RECIPE | options, 1)
    return trained, steps, labelled


def test_varpro_fit(structures, tmp_path):
    """The recipe's VarPro fit, whose loss takes the outputs from the columns of the last solve, gives the losses of a build
    that computed them, stored in etn_reference/varpro_fit.npz, to rounding; that build kept min_dist."""
    stored = numpy.load(ETN_REFERENCE / "varpro_fit.npz")
    _, steps, _ = varpro_fit(structures, tmp_path, no_mindist_update="true")
    assert [len(s) for s in steps] == list(stored["counts"])
    numpy.testing.assert_allclose(numpy.concatenate(steps), stored["losses"], rtol=1e-9)


@pytest.mark.parametrize("fraction", ["1", "0.5"])
def test_varpro_gradient(fraction, structures, tmp_path):
    """Under varpro, its system built on every structure or on half of them, the objective's gradient matches
    fourth-order finite differences along 4 directions of the coefficients BFGS steps, after the recipe's short fit on
    the 6 structures 11 times over."""
    from otf_engine._etn._etn_testing import ETNObjective

    trained, _, labelled = varpro_fit(structures, tmp_path, chosen=FRACTION_CHOSEN)
    path = tmp_path / "trained.etn"
    trained.write(path)
    objective = ETNObjective(str(path), labelled, RECIPE | {"varpro_fraction": fraction})
    x = numpy.asarray(trained.get_coeffs())
    grad = objective(x)[1]

    rng = numpy.random.default_rng(0)
    for _ in range(4):
        xi = rng.standard_normal(len(x)) * (grad != 0.0)
        xi /= numpy.linalg.norm(xi)
        analytic = grad @ xi
        errors = []
        for h in (1e-2, 1e-3, 1e-4):
            f = {m: objective(x + m * h * xi)[0] for m in (-2, -1, 1, 2)}
            fd = (8 * (f[1] - f[-1]) - (f[2] - f[-2])) / (12 * h)
            errors += [abs(fd - analytic) / abs(analytic)]
        assert min(errors) < 1e-4


def test_varpro_fraction_small_set(structures, tmp_path):
    """On the 6 structures, varpro_fraction 0.5 would draw fewer than 32 structures on average, so the objective takes them
    all, as varpro_fraction 1 does."""
    from otf_engine._etn._etn_testing import ETNObjective

    trained, _, labelled = varpro_fit(structures, tmp_path, chosen=range(6))
    path = tmp_path / "trained.etn"
    trained.write(path)
    x = numpy.asarray(trained.get_coeffs())
    halved = ETNObjective(str(path), labelled, RECIPE | {"varpro_fraction": "0.5"})(x)[0]
    full = ETNObjective(str(path), labelled, RECIPE | {"varpro_fraction": "1"})(x)[0]

    assert halved == full


def test_varpro_redraw(structures, tmp_path):
    """Under varpro_fraction 0.5 on the 6 structures 11 times over, varpro_redraw_every 1 draws anew at each step: its fit
    leaves the fixed draw's losses after the first step, and lowers the loss."""
    _, fixed, _ = varpro_fit(structures, tmp_path, chosen=FRACTION_CHOSEN, varpro_fraction="0.5", varpro_redraw_every="0")
    _, redrawn, _ = varpro_fit(structures, tmp_path, chosen=FRACTION_CHOSEN, varpro_fraction="0.5", varpro_redraw_every="1")
    assert redrawn[0] == fixed[0]
    assert redrawn[2] != fixed[2]
    assert redrawn[-1][-1] < redrawn[0][0]


def test_set_min_cutoff(structures):
    """set_min_cutoff rebuilds RBCinf at the new min_dist: the ETN computes as one built with it."""
    from otf_engine._etn import ETN
    from etn_potentials import POTENTIALS, MAX_DIST, RADIAL_BASIS_SIZE

    moved = untrained_etn("etn-3")
    moved.set_min_cutoff(1.5)
    _, species_ranks, radial_funcs_counts, radial_ranks, bond_ranks = POTENTIALS["etn-3"]
    built = ETN(species_ranks=species_ranks, radial_funcs_counts=radial_funcs_counts, radial_ranks=radial_ranks, bond_ranks=bond_ranks, species_count=moved.get_species_count(), min_dist=1.5, max_dist=MAX_DIST, radial_basis_size=RADIAL_BASIS_SIZE)
    nl = neighbors(structures[1], MAX_DIST)

    moved_result, built_result = moved.compute(nl), built.compute(nl)

    assert moved.get_min_cutoff() == 1.5
    assert moved_result["energy"] == pytest.approx(built_result["energy"], rel=1e-12)
    numpy.testing.assert_allclose(moved_result["forces"], built_result["forces"], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("keep", [False, True], ids=["update", "no_mindist_update"])
def test_min_dist_update(keep, structures, tmp_path):
    """The ETN fit sets min_dist to 0.99 of the shortest training distance, as the MTP fit does, unless no_mindist_update."""
    from ase.neighborlist import neighbor_list
    from etn_potentials import MIN_DIST, MAX_DIST

    options = {"no_mindist_update": "true"} if keep else {}
    trained, _, _ = varpro_fit(structures, tmp_path, **options)
    shortest = min(neighbor_list("d", structures[k], MAX_DIST).min() for k in (0, 1))

    assert trained.get_min_cutoff() == pytest.approx(MIN_DIST if keep else 0.99 * shortest, rel=1e-12)
