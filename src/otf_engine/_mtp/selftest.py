"""Finite-difference checks of the MTP analytical derivatives.

Every check returns the largest relative discrepancy between an analytical
derivative and its central-difference estimate. Values around 1e-6 are what
a correct implementation gives at the default step; anything above ``tol``
indicates a real disagreement.
"""

import numpy
from numpy import float64

from .neighbors import neighbors


def _energy(pot, atoms, cutoff):
    return float(pot.compute(neighbors(atoms, cutoff), compute_virials=False)["energy"])


def _rel_error(analytic_values, numeric_values):
    analytic_values = numpy.asarray(analytic_values, dtype=float64)
    numeric_values = numpy.asarray(numeric_values, dtype=float64)
    scale = max(float(numpy.abs(numeric_values).max()), 1e-30)
    return float(numpy.abs(analytic_values - numeric_values).max()) / scale


def check_forces(pot, atoms, step=1e-5, n_atoms=4):
    """Compare forces against -dE/dx by central differences.

    Only the first *n_atoms* atoms are displaced, since each one costs six
    energy evaluations.
    """
    cutoff = pot.get_max_cutoff()
    analytic_forces = numpy.asarray(pot.compute(neighbors(atoms, cutoff), compute_virials=False)["forces"], dtype=float64)

    probe = atoms.copy()
    n = min(n_atoms, len(atoms))
    numeric_forces = numpy.zeros((n, 3))
    for i in range(n):
        for d in range(3):
            original = probe.positions[i, d]
            probe.positions[i, d] = original + step
            e_plus = _energy(pot, probe, cutoff)
            probe.positions[i, d] = original - step
            e_minus = _energy(pot, probe, cutoff)
            probe.positions[i, d] = original
            numeric_forces[i, d] = -(e_plus - e_minus) / (2 * step)

    return _rel_error(analytic_forces[:n], numeric_forces)


def _coeff_block(pot, block):
    if block == "radial":
        return numpy.asarray(pot.get_radial_basis_coeffs(), dtype=float64).ravel(), pot.set_radial_basis_coeffs
    if block == "linear":
        return numpy.asarray(pot.get_linear_coeffs(), dtype=float64).copy(), pot.set_linear_coeffs
    raise ValueError(f"unknown coefficient block '{block}'")


def check_coeff_grad(pot, atoms, block="radial", step=1e-6, n_coeffs=8, seed=0):
    """Compare a site-energy gradient block against central differences.

    *n_coeffs* coefficients are sampled at random from the block; each costs
    two full evaluations.
    """
    cutoff = pot.get_max_cutoff()
    nl = neighbors(atoms, cutoff)

    if block == "radial":
        analytic_grad, _, _ = pot.eval_grad_radial(nl, False)
        analytic_grad = numpy.asarray(analytic_grad, dtype=float64)
    else:
        site_grad, _, _ = pot.eval_grad_linear(nl, False)
        analytic_grad = numpy.asarray(site_grad, dtype=float64).sum(axis=0)

    coeffs, setter = _coeff_block(pot, block)
    rng = numpy.random.default_rng(seed)
    probe_indices = rng.choice(len(coeffs), size=min(n_coeffs, len(coeffs)), replace=False)

    numeric_grad = numpy.zeros(len(probe_indices))
    for slot, k in enumerate(probe_indices):
        original = coeffs[k]
        coeffs[k] = original + step
        setter(coeffs)
        e_plus = _energy(pot, atoms, cutoff)
        coeffs[k] = original - step
        setter(coeffs)
        e_minus = _energy(pot, atoms, cutoff)
        coeffs[k] = original
        setter(coeffs)
        numeric_grad[slot] = (e_plus - e_minus) / (2 * step)

    return _rel_error(analytic_grad[probe_indices], numeric_grad)


def check_force_coeff_grad(pot, atoms, block="radial", step=1e-6, n_coeffs=4, seed=0):
    """Compare a force gradient block against central differences in coefficients."""
    cutoff = pot.get_max_cutoff()
    nl = neighbors(atoms, cutoff)

    if block == "radial":
        _, force_grad, _ = pot.eval_grad_radial(nl, False)
    else:
        _, force_grad, _ = pot.eval_grad_linear(nl, False)
    force_grad = numpy.asarray(force_grad, dtype=float64)

    coeffs, setter = _coeff_block(pot, block)
    rng = numpy.random.default_rng(seed)
    probe_indices = rng.choice(len(coeffs), size=min(n_coeffs, len(coeffs)), replace=False)

    analytic_grad, numeric_grad = [], []
    for k in probe_indices:
        original = coeffs[k]
        coeffs[k] = original + step
        setter(coeffs)
        f_plus = numpy.asarray(pot.compute(nl, compute_virials=False)["forces"], dtype=float64)
        coeffs[k] = original - step
        setter(coeffs)
        f_minus = numpy.asarray(pot.compute(nl, compute_virials=False)["forces"], dtype=float64)
        coeffs[k] = original
        setter(coeffs)

        analytic_grad += [force_grad[:, :, k].ravel()]
        numeric_grad += [((f_plus - f_minus) / (2 * step)).ravel()]

    return _rel_error(numpy.concatenate(analytic_grad), numpy.concatenate(numeric_grad))


def check_gradients(pot, atoms, tol=1e-4):
    """Run every check and return {name: relative error}.

    Raises AssertionError if any exceeds *tol*.
    """
    results = {
        "forces": check_forces(pot, atoms),
        "radial site-energy grad": check_coeff_grad(pot, atoms, "radial"),
        "linear site-energy grad": check_coeff_grad(pot, atoms, "linear"),
        "radial force grad": check_force_coeff_grad(pot, atoms, "radial"),
        "linear force grad": check_force_coeff_grad(pot, atoms, "linear"),
    }
    bad = {name: err for name, err in results.items() if not (err <= tol)}
    if bad:
        raise AssertionError(f"gradient checks exceeded tol={tol}: {bad}")
    return results
