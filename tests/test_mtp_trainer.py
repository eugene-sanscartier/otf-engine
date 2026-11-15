"""MTPTrainer's stages against mlip-3's, one at a time, and BFGS against mlip-3's on the Rosenbrock function.

Each script runs in mlip3_probe.cpp under --recapture, which stores its outputs in mtp_reference/trainer.npz,
and in _mtp_testing.MTPTrainerStages, whose outputs must match them.
"""

import re
import subprocess

import numpy
import pytest

from conftest import DATA, MTP_REFERENCE, StoredOutputs, parse_probe
from otf_engine.train import MLP_ALGORITHM, training_structure

TEACHER = MTP_REFERENCE / "potential-16.mtp"


def perturbed_coeffs(path, scale, seed):
    """The coefficients of the potential in path, the radial ones multiplied by 1 + scale * N(0, 1)."""
    from otf_engine._mtp import MTPTraining

    potential = MTPTraining(str(path))
    coeffs = numpy.array(potential.get_coeffs())
    n_radial = potential.get_radial_coeff_count()
    coeffs[:n_radial] *= 1 + scale * numpy.random.default_rng(seed).standard_normal(n_radial)
    return coeffs


def coeffs_line(coeffs):
    return f"coeffs {len(coeffs)} " + " ".join(repr(float(c)) for c in coeffs)


# name: (potential, settings, script); the settings are mlp's, which the trainer shares
SCRIPTS = {
    "untrained": (MTP_REFERENCE / "template-08.almtp", {}, ["add_species", "print", "lin_optimize", "print", "objective", "rescale", "print", "objective"]),
    "trained": (TEACHER, {}, ["add_species", coeffs_line(perturbed_coeffs(TEACHER, 0.01, 1)), "objective", "lin_optimize", "print", "objective", "rescale", "print", "objective"]),
    "add_species": (DATA / "two-species-08.mtp", {}, ["add_species", "print", "objective", "lin_optimize", "print"]),
    "radial_above_basis": (MTP_REFERENCE / "template-16-r3.almtp", {}, ["add_species", "print", "lin_optimize", "print", "objective"]),
    "weights": (DATA / "two-species-08.mtp", {"weight_scaling": "2", "weight_scaling_forces": "1", "select_factor": "2", "penalty_weight": "0.001"}, ["add_species", "objective", "lin_optimize", "print", "objective"]),
    "scale_by_force": (DATA / "two-species-08.mtp", {"scale_by_force": "0.1"}, ["add_species", "objective", "lin_optimize", "print", "objective"]),
    "energy_only": (DATA / "two-species-08.mtp", {"force_weight": "0", "stress_weight": "0"}, ["add_species", "objective", "lin_optimize", "print", "objective"]),
    "no_mindist_update": (MTP_REFERENCE / "template-08.almtp", {"no_mindist_update": "true"}, ["add_species", "print"]),
}


def run_ours(potential, settings, script, structures):
    """The script's outputs from MTPTrainerStages, as mlip3_probe prints them: (name, values, whether a linear solve came before, radial coefficient count)."""
    from otf_engine._mtp._mtp_testing import MTPTrainerStages

    stages = MTPTrainerStages(str(potential), [training_structure(atoms, 5.0, k) for k, atoms in enumerate(structures)], {**MLP_ALGORITHM, **settings, "log": "none"})
    outputs = []
    solved = False
    for line in script:
        command, *words = line.split()
        if command == "add_species":
            stages.add_species()
            stages.update_min_dist()
        elif command == "orthogonalize":
            stages.orthogonalize()
        elif command == "lin_optimize":
            stages.fit_linear()
        elif command == "rescale":
            stages.rescale()
        elif command == "coeffs":
            stages.coeffs = numpy.array(words[1:], dtype=float)
        solved = solved or command in ("lin_optimize", "rescale")

        n_radial = stages.potential.get_radial_coeff_count()
        if command == "objective":
            loss, grad = stages.objective(stages.coeffs)
            outputs += [("loss", numpy.array([loss]), solved, n_radial), ("grad", grad, solved, n_radial)]
        elif command == "print":
            outputs += [(label, values, solved, n_radial) for label, values in [("coeffs", stages.coeffs), ("scaling", numpy.array([stages.scaling])), ("min_dist", numpy.array([stages.min_dist])), ("reg_vector", stages.reg_vector)]]
    return outputs


@pytest.fixture(scope="module")
def stored(recapture):
    outputs = StoredOutputs(MTP_REFERENCE / "trainer.npz", recapture)
    yield outputs
    outputs.save()


def reference(stored, name, run_probe):
    """The stored outputs of name, rerun through the probe first under --recapture."""
    if stored.recapture:
        stored.replace_under(name, run_probe())
    keys = stored.keys_under(name)
    return [(key.rsplit("/", 1)[1].split("_", 1)[1], stored[key]) for key in keys]


@pytest.mark.parametrize("name", [pytest.param(name, marks=pytest.mark.xfail(strict=True, reason="mlp weighs a force of its loss by the predicted force under scale_by_force, the trainer by the reference one")) if name == "scale_by_force" else name for name in SCRIPTS])
def test_stage(name, structures, stored, mlip3_probe, tmp_path):
    potential, settings, script = SCRIPTS[name]

    def run_probe():
        script_file = tmp_path / "script.txt"
        script_file.write_text("".join(f"set {k} {v}\n" for k, v in settings.items()) + "trainer\n" + "\n".join(script) + "\n")
        printed = subprocess.run([str(mlip3_probe), str(potential), str(DATA / "structures.cfg"), str(script_file)], check=True, capture_output=True, text=True).stdout
        return parse_probe(printed)

    theirs = reference(stored, name, run_probe)
    ours = run_ours(potential, settings, script, structures)
    assert [label for label, *_ in ours] == [label for label, _ in theirs]

    # A linear solve is ill-conditioned on so few structures: rounding moves its coefficients along flat directions by up to
    # 1e-5, and the loss by 1e-11. After the first solve the linear block and the gradient are left out, and the loss
    # compared to 1e-10.
    for k, ((label, a, solved, n_radial), (_, b)) in enumerate(zip(ours, theirs)):
        assert a.shape == b.shape, f"output {k} ({label}): shape {a.shape} vs mlp's {b.shape}"
        if solved and label == "grad":
            continue
        if solved and label == "coeffs":
            a, b = a[:n_radial], b[:n_radial]
        tol = 1e-10 if solved and label == "loss" else 0.0 if label in ("scaling", "min_dist") else 1e-12
        scale = numpy.max(numpy.abs(b)) or 1.0
        worst = numpy.max(numpy.abs(a - b)) / scale
        assert worst <= tol, f"output {k} ({label}): {worst:.1e} relative to its largest entry"


@pytest.mark.parametrize("n", [2, 6])
def test_bfgs_rosenbrock(n, stored, mlip3_probe):
    """BFGS takes mlp's steps exactly over 300 iterations, line searches included."""
    from otf_engine._mtp._mtp_testing import bfgs_rosenbrock

    steps = 300
    if stored.recapture:
        printed = subprocess.run([str(mlip3_probe), "--rosenbrock", str(n), str(steps)], check=True, capture_output=True, text=True).stdout
        outputs = parse_probe(printed)
        for label in ("x", "state"):
            stored[f"rosenbrock_{n}/{label}"] = numpy.array([v for name, v in outputs if name == label])

    xs, states = bfgs_rosenbrock(n, steps)
    numpy.testing.assert_array_equal(xs, stored[f"rosenbrock_{n}/x"])
    numpy.testing.assert_array_equal(states, stored[f"rosenbrock_{n}/state"])


NUMBER = re.compile(r"[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?")
BFGS_LINE = re.compile(r"BFGS iter +\d+: +(?:f=)?(\S+)")    # mlp's, and the trainer's without f=
# the trainer's lines that mlp does not write: its own, and its error tables
TRAINER_LINE = re.compile(r"^\d+ coefficients \(|^ +loss +energy |^\d+ iterations in |^Trained in |^Training set errors|^ +n +MAE | \d\.\d\de[-+]\d+ +\d\.\d\de[-+]\d+ |^max (energy/atom|force) error: |^min_dist = |^Minimal interatomic distance ")
LOSS_TERMS = re.compile(r"( +\S+ \( *\d+%\)){4}$")    # the trainer's terms of the loss, after mlp's loss

# name: (potential, options, the log's last line compared); the options are mlp's. After the 200 steps to the tolerance
# stop, rounding has moved the coefficients enough to flip the rescaling's choice between close condition numbers.
FITS = {
    "preinit": (MTP_REFERENCE / "template-08.almtp", {"iteration_limit": "30"}, "Rescaling to"),
    "tolerance": (DATA / "two-species-08.mtp", {"iteration_limit": "400", "tolerance": "1"}, "MTPR training ended"),
}


def fit_log(text, last):
    """The lines of a log that mlp writes too, from the training-set count to the last one starting with last, with single spaces."""
    lines = [" ".join(LOSS_TERMS.sub("", line).replace(": f=", ": ").split()) for line in text.splitlines() if not TRAINER_LINE.search(line)]
    start = next(i for i, line in enumerate(lines) if "configurations found" in line)
    end = max(i for i, line in enumerate(lines) if line.startswith(last))
    return lines[start:end + 1]


@pytest.mark.parametrize("name", FITS)
def test_fit_log(name, structures, stored, recapture, mlip3, tmp_path):
    """A whole fit logs mlp's stages in mlp's order, and its first 5 losses to 1e-4.

    mlp logs a loss after a line search, along a gradient that the linear solve's conditioning moves by 1e-5 relatively.
    """
    from otf_engine._mtp._mtp_testing import MTPTrainerStages

    potential, options, last = FITS[name]
    if recapture:
        cmd = [str(mlip3 / "bin" / "mlp"), "train", str(potential), str(DATA / "structures.cfg"), "--log=stdout", f"--save_to={tmp_path / 'out.mtp'}"] + [f"--{k}={v}" for k, v in options.items()]
        printed = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=tmp_path).stdout
        stored[f"fit_{name}/log"] = numpy.array(fit_log(printed, last))

    stages = MTPTrainerStages(str(potential), [training_structure(atoms, 5.0, k) for k, atoms in enumerate(structures)], {**MLP_ALGORITHM, **options, "log": "none"})
    stages.train()
    ours, theirs = fit_log(stages.log, last), list(stored[f"fit_{name}/log"])

    assert [NUMBER.sub("#", line) for line in ours] == [NUMBER.sub("#", line) for line in theirs]
    a = numpy.array([float(m.group(1)) for line in ours if (m := BFGS_LINE.search(line))][:5])
    b = numpy.array([float(m.group(1)) for line in theirs if (m := BFGS_LINE.search(line))][:5])
    numpy.testing.assert_allclose(a, b, rtol=1e-4)
