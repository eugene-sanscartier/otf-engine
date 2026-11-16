"""MTPFit's properties that hold without a reference: its gradient, a teacher's fit, its invariance under the ranks, and its options."""

import numpy
import pytest

from conftest import DATA, MTP_REFERENCE
from otf_engine._core.calculator import PairCalculator
from otf_engine._core import FitStructure
from otf_engine.train import MLP_ALGORITHM

TEACHER = MTP_REFERENCE / "potential-16.mtp"
TEMPLATE = MTP_REFERENCE / "template-08.almtp"


def labelled(structures):
    return [FitStructure(atoms, 5.0, k) for k, atoms in enumerate(structures)]


def stages(potential_path, structures, **options):
    from otf_engine._mtp._mtp_testing import MTPFitStages

    return MTPFitStages(str(potential_path), labelled(structures), {"log": "none", **options})


def radial_rows(stage):
    """The radial functions as the rows of W, K x (species pairs * basis size)."""
    radial = numpy.asarray(stage.potential.get_radial_basis_coeffs())
    P, K, R = radial.shape
    return stage.coeffs[: radial.size].reshape(P, K, R).transpose(1, 0, 2).reshape(K, P * R)


@pytest.fixture(scope="module")
def teacher_labelled(structures):
    """The structures labelled by the teacher to full precision."""
    from otf_engine._mtp import PairMTP

    teacher = PairMTP(str(TEACHER))
    relabelled = []
    for atoms in structures:
        copy = atoms.copy()
        copy.arrays["type_index"] = atoms.arrays["type_index"].copy()
        copy.calc = PairCalculator(teacher)
        copy.calc.calculate(copy, ["energy", "forces", "stress"])
        relabelled += [copy]
    return relabelled


# name: (options, tolerance); varpro's gradient carries the error of its linear solve, which the system's conditioning sets
OPTION_SETS = {
    "mlp": (MLP_ALGORITHM, 1e-6),
    "recipe": ({}, 1e-3),
    "riemannian": ({"varpro": "false"}, 1e-6),
    "varpro": ({"riemannian": "false"}, 1e-3),
}


@pytest.mark.parametrize("name", OPTION_SETS)
def test_gradient(name, structures):
    """The objective's gradient matches fourth-order finite differences along 4 directions, tangent under riemannian."""
    options, tol = OPTION_SETS[name]
    stage = stages(TEACHER, structures, **options)
    riemannian, varpro = options.get("riemannian") != "false", options.get("varpro") != "false"
    n_radial = stage.potential.get_radial_coeff_count()
    radial = numpy.asarray(stage.potential.get_radial_basis_coeffs())
    P, K, R = radial.shape

    def with_rows(c, W):
        out = c.copy()
        out[:n_radial] = W.reshape(K, P, R).transpose(1, 0, 2).ravel()
        return out

    def rows(c):
        return c[:n_radial].reshape(P, K, R).transpose(1, 0, 2).reshape(K, P * R)

    def retract(c):
        q, r = numpy.linalg.qr(rows(c).T)
        return with_rows(c, (q * numpy.sign(numpy.diag(r))).T)

    x = stage.coeffs.copy()
    x[:n_radial] *= 1 + 0.01 * numpy.random.default_rng(1).standard_normal(n_radial)    # off the teacher, where the gradient vanishes
    if riemannian: x = retract(x)
    grad = stage.objective(x)[1]
    W, G = rows(x), rows(grad)
    if riemannian:
        assert numpy.linalg.norm(G @ W.T + W @ G.T) / 2 < 1e-12 * max(1.0, numpy.linalg.norm(grad))

    rng = numpy.random.default_rng(0)
    for _ in range(4):
        xi = rng.standard_normal(len(x))
        if varpro: xi[n_radial:] = 0.0
        if riemannian:
            Z = rows(xi)
            xi = with_rows(xi, Z - (Z @ W.T + W @ Z.T) / 2 @ W)
        xi /= numpy.linalg.norm(xi)
        move = retract if riemannian else (lambda c: c)
        analytic = grad @ xi
        errors = []
        for h in (1e-3, 3e-4, 1e-4):
            f = {m: stage.objective(move(x + m * h * xi))[0] for m in (-2, -1, 1, 2)}
            fd = (8 * (f[1] - f[-1]) - (f[2] - f[-2])) / (12 * h)
            errors += [abs(fd - analytic) / abs(analytic)]
        assert min(errors) < tol


def test_teacher_linear_fit(teacher_labelled):
    """From the teacher's radial functions, with its linear coefficients zeroed, the linear fit goes back below the teacher's loss.

    The teacher's loss is its regularization alone, and the fit minimizes the data and the regularization together.
    """
    stage = stages(TEACHER, teacher_labelled, **MLP_ALGORITHM, reg_param="1e-16")
    n_radial = stage.potential.get_radial_coeff_count()
    teacher = stage.coeffs.copy()
    at_teacher = stage.objective(teacher)[0]

    zeroed = teacher.copy()
    zeroed[n_radial:] = 0.0
    stage.coeffs = zeroed
    at_zero = stage.objective(zeroed)[0]
    stage.fit_linear()
    fitted = stage.objective(stage.coeffs)[0]

    assert fitted <= at_teacher
    assert fitted < 1e-15 * at_zero


@pytest.mark.parametrize("options, loss_tol, grad_tol", [(MLP_ALGORITHM, 1e-12, 1e-10), ({}, 1e-10, 1e-5)], ids=["mlp", "recipe"])
def test_threads(options, loss_tol, grad_tol, structures):
    """The objective and its gradient do not depend on how many threads share the structures, to rounding.

    Under the recipe's varpro the gradient goes through a linear solve, whose conditioning turns the order of the
    reductions into 2e-12 on the loss and 2e-6 on the gradient, relatively.
    """
    from otf_engine._mtp._mtp_testing import objective_on_threads

    stage = stages(TEACHER, structures, **options)
    coeffs = stage.coeffs
    one = objective_on_threads(str(TEACHER), labelled(structures), {"log": "none", **options}, coeffs, 1)
    for ranks in (2, 3):
        loss, grad = objective_on_threads(str(TEACHER), labelled(structures), {"log": "none", **options}, coeffs, ranks)
        assert abs(loss - one[0]) <= loss_tol * abs(one[0])
        assert numpy.max(numpy.abs(grad - one[1])) <= grad_tol * numpy.max(numpy.abs(one[1]))


def test_random_seed(structures):
    """init_random with a seed draws the same radial coefficients each time, and another seed others."""
    def drawn(seed):
        stage = stages(TEMPLATE, structures, init_random="true", random_seed=str(seed), iteration_limit="0")
        stage.run()
        return stage.coeffs

    numpy.testing.assert_array_equal(drawn(7), drawn(7))
    assert not numpy.array_equal(drawn(7), drawn(8))


def test_inverse_hessian_resumes(structures):
    """Two fits of 5 steps under bfgs_only, the second from the first's coefficients and inverse Hessian, end where one of 10 does."""
    from otf_engine._mtp import fit_mtp, write_mtp

    import tempfile

    options = {**MLP_ALGORITHM, "bfgs_only": "true", "log": "none"}
    data = labelled(structures)
    n = len(stages(TEACHER, structures).coeffs)

    hessian = numpy.zeros((n, n))
    whole = fit_mtp(str(TEACHER), data, {**options, "iteration_limit": "10"}, 1, inverse_hessian=hessian)

    hessian = numpy.zeros((n, n))
    first = fit_mtp(str(TEACHER), data, {**options, "iteration_limit": "5"}, 1, inverse_hessian=hessian)
    with tempfile.TemporaryDirectory() as work:
        write_mtp(first, f"{work}/first.mtp")
        second = fit_mtp(f"{work}/first.mtp", data, {**options, "iteration_limit": "5"}, 1, inverse_hessian=hessian)

    numpy.testing.assert_allclose(second.get_coeffs(), whole.get_coeffs(), rtol=0, atol=1e-12 * numpy.max(numpy.abs(whole.get_coeffs())))


def test_radial_only(structures):
    """radial_only moves the radial coefficients and leaves the species and linear ones."""
    stage = stages(TEACHER, structures, **MLP_ALGORITHM, radial_only="true", iteration_limit="5")
    n_radial = stage.potential.get_radial_coeff_count()
    before = stage.coeffs.copy()
    stage.run()
    after = stage.coeffs

    assert not numpy.array_equal(after[:n_radial], before[:n_radial])
    numpy.testing.assert_array_equal(after[n_radial:], before[n_radial:])


def test_riemannian_orthonormal(structures):
    """The recipe's fit keeps W W^T = I at every step it takes and lowers the loss."""
    stage = stages(TEACHER, structures, iteration_limit="20")
    start = stage.coeffs.copy()
    stage.run()
    W = radial_rows(stage)
    numpy.testing.assert_allclose(W @ W.T, numpy.eye(len(W)), atol=1e-12)
    assert stage.objective(stage.coeffs)[0] < stage.objective(start)[0]


def test_checkpoint(structures, caplog):
    """The checkpoint receives the potential during the fit; its first failure is a warning, ends checkpointing, and the fit goes on."""
    from otf_engine._mtp import fit_mtp

    data = labelled(structures)
    options = {**MLP_ALGORITHM, "iteration_limit": "30", "skip_preinit": "true", "log": "none"}
    saved = []
    fit_mtp(str(TEMPLATE), data, options, 1, checkpoint=lambda p: saved.append(numpy.array(p.get_coeffs())))
    assert len(saved) >= 2 and not numpy.array_equal(saved[0], saved[-1])

    calls = []

    def failing(potential):
        calls.append(1)
        raise OSError("disk full")

    trained = fit_mtp(str(TEMPLATE), data, options, 2, checkpoint=failing)
    assert len(calls) == 1 and trained.is_trained()
    assert "checkpoint failed, training goes on without checkpoints: OSError: disk full" in caplog.text


def test_log_cannot_open(structures, tmp_path, caplog):
    """A log that cannot be opened is a warning, and the fit goes on without it."""
    from otf_engine._mtp import fit_mtp

    log = tmp_path / "missing" / "train.log"
    trained = fit_mtp(str(TEMPLATE), labelled(structures), {**MLP_ALGORITHM, "iteration_limit": "5", "skip_preinit": "true", "log": str(log)}, 2)
    assert trained.is_trained()
    assert f"cannot open the fit's log '{log}', training without it" in caplog.text


def test_no_iterations_leaves_untrained(structures):
    """iteration_limit 0 adds the species and leaves an untrained potential untrained."""
    stage = stages(TEMPLATE, structures, iteration_limit="0")
    stage.run()
    assert not stage.potential.is_trained()
    assert stage.potential.get_species_count() == 3


def renamed(structures, mapping):
    out = []
    for atoms in structures:
        copy = atoms.copy()
        copy.calc = atoms.calc
        copy.arrays["type_index"] = numpy.array([mapping.get(int(t), int(t)) for t in atoms.arrays["type_index"]], dtype=numpy.int32)
        out += [copy]
    return out


@pytest.mark.parametrize("case, potential_path, mapping, message", [
    ("gap", TEMPLATE, {1: 3}, "has species 3 but not species 1"),
    ("missing from training", TEACHER, {2: 1}, "species 2 of the potential is not present"),
])
def test_species_errors(case, potential_path, mapping, message, structures):
    """Species must run from 0 without a gap, and cover a trained potential's."""
    from otf_engine._mtp import fit_mtp

    with pytest.raises(RuntimeError, match=message):
        fit_mtp(str(potential_path), labelled(renamed(structures, mapping)), {"iteration_limit": "1", "log": "none"}, 2)


@pytest.mark.parametrize("options, message", [
    ({"no_such_option": "1"}, "unknown option 'no_such_option'"),
    ({"varpro": "maybe"}, "is not true or false"),
])
def test_option_errors(options, message, structures):
    from otf_engine._mtp import fit_mtp

    with pytest.raises(RuntimeError, match=message):
        fit_mtp(str(TEMPLATE), labelled(structures), {"log": "none", **options}, 1)


def test_no_structures():
    from otf_engine._mtp import fit_mtp

    with pytest.raises(RuntimeError, match="no training structures"):
        fit_mtp(str(TEMPLATE), [], {"log": "none"}, 1)
