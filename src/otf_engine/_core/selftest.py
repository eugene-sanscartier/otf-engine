"""Finite-difference checks of a potential's analytical derivatives.

Every check returns the largest relative discrepancy between an analytical
derivative and its central-difference estimate. Values around 1e-6 are what
a correct implementation gives at the default step; anything above ``tol``
indicates a real disagreement. check_forces, check_virial and check_invariance
take any Pair; the checks on the coefficient blocks take an MTPTraining.
"""

import numpy
from numpy import float64

from .neighbors import neighbors


def _energy(pot, atoms, cutoff):
    return float(pot.compute(neighbors(atoms, cutoff))["energy"])


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
    analytic_forces = numpy.asarray(pot.compute(neighbors(atoms, cutoff))["forces"], dtype=float64)

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


def check_virial(pot, atoms, step=1e-6):
    """Compare the virial against -dE/d(strain) by central differences.

    The virial's xy is sum_i x_i F_iy, the derivative of -E with respect to the
    strain that moves y by x; the off-diagonal pairs agree for an invariant potential.
    """
    cutoff = pot.get_max_cutoff()
    analytic_virial = numpy.asarray(pot.compute(neighbors(atoms, cutoff))["virials"], dtype=float64)

    numeric = numpy.zeros((3, 3))
    for a in range(3):
        for b in range(3):
            energies = []
            for sign in (1, -1):
                strain = numpy.eye(3)
                strain[b, a] += sign * step
                probe = atoms.copy()
                probe.set_cell(atoms.cell.array @ strain.T, scale_atoms=True)
                energies += [_energy(pot, probe, cutoff)]
            numeric[a, b] = -(energies[0] - energies[1]) / (2 * step)

    numeric_virial = [numeric[0, 0], numeric[1, 1], numeric[2, 2], numeric[0, 1], numeric[0, 2], numeric[1, 2]]
    return _rel_error(analytic_virial, numeric_virial)


def check_invariance(pot, atoms, seed=0):
    """Largest relative change of the energy and the rotated-back forces under a random rotation and its reflection."""
    cutoff = pot.get_max_cutoff()
    reference = pot.compute(neighbors(atoms, cutoff))
    energy, forces = float(reference["energy"]), numpy.asarray(reference["forces"], dtype=float64)

    rotation, _ = numpy.linalg.qr(numpy.random.default_rng(seed).standard_normal((3, 3)))
    worst = 0.0
    for transform in (rotation, -rotation):
        probe = atoms.copy()
        probe.set_cell(atoms.cell.array @ transform.T)
        probe.positions = atoms.positions @ transform.T
        moved = pot.compute(neighbors(probe, cutoff))
        worst = max(worst, abs(float(moved["energy"]) - energy) / max(abs(energy), 1e-30))
        worst = max(worst, _rel_error(numpy.asarray(moved["forces"]) @ transform, forces))
    return worst


def _coeff_block(pot, block):
    """The block's coefficients, their setter, and the block's first column in c = [c_radial | c_species | beta_linear]."""
    if block == "radial":
        return numpy.asarray(pot.get_radial_basis_coeffs(), dtype=float64).ravel(), pot.set_radial_basis_coeffs, 0
    if block == "species":
        return numpy.asarray(pot.get_species_coeffs(), dtype=float64).copy(), pot.set_species_coeffs, pot.get_radial_coeff_count()
    if block == "linear":
        return numpy.asarray(pot.get_linear_coeffs(), dtype=float64).copy(), pot.set_linear_coeffs, pot.get_radial_coeff_count() + pot.get_species_count()
    raise ValueError(f"unknown coefficient block '{block}'")


def check_coeff_grad(pot, atoms, block="radial", step=1e-6, n_coeffs=8, seed=0):
    """Compare a site-energy gradient block against central differences.

    *n_coeffs* coefficients are sampled at random from the block; each costs
    two full evaluations.
    """
    cutoff = pot.get_max_cutoff()
    nl = neighbors(atoms, cutoff)

    coeffs, setter, offset = _coeff_block(pot, block)
    site_grad, _, _ = pot.eval_grad(nl, forces=True)
    analytic_grad = numpy.asarray(site_grad, dtype=float64).sum(axis=0)[offset:offset + len(coeffs)]

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

    coeffs, setter, offset = _coeff_block(pot, block)
    _, force_grad, _ = pot.eval_grad(nl, forces=True)
    force_grad = numpy.asarray(force_grad, dtype=float64)[:, :, offset:offset + len(coeffs)]

    rng = numpy.random.default_rng(seed)
    probe_indices = rng.choice(len(coeffs), size=min(n_coeffs, len(coeffs)), replace=False)

    analytic_grad, numeric_grad = [], []
    for k in probe_indices:
        original = coeffs[k]
        coeffs[k] = original + step
        setter(coeffs)
        f_plus = numpy.asarray(pot.compute(nl)["forces"], dtype=float64)
        coeffs[k] = original - step
        setter(coeffs)
        f_minus = numpy.asarray(pot.compute(nl)["forces"], dtype=float64)
        coeffs[k] = original
        setter(coeffs)

        analytic_grad += [force_grad[:, :, k].ravel()]
        numeric_grad += [((f_plus - f_minus) / (2 * step)).ravel()]

    return _rel_error(numpy.concatenate(analytic_grad), numpy.concatenate(numeric_grad))


def check_loss_grad(pot, atoms, block="radial", step=1e-6, n_coeffs=8, seed=0):
    """Compare eval_loss_grad against central differences of a loss linear in E, F and virial.

    The loss weights are random and fixed, so its derivative in the
    coefficients is the one eval_loss_grad takes them for.
    """
    cutoff = pot.get_max_cutoff()
    nl = neighbors(atoms, cutoff)

    rng = numpy.random.default_rng(seed)
    dloss_denergy = float(rng.standard_normal())
    dloss_dforces = rng.standard_normal((len(atoms), 3))
    dloss_dvirial = rng.standard_normal(6)

    def loss():
        result = pot.compute(nl)
        return dloss_denergy * float(result["energy"]) + float((dloss_dforces * numpy.asarray(result["forces"])).sum()) + float(dloss_dvirial @ numpy.asarray(result["virials"]))

    coeffs, setter, offset = _coeff_block(pot, block)
    analytic_grad = numpy.asarray(pot.eval_loss_grad(nl, dloss_denergy, dloss_dforces, dloss_dvirial), dtype=float64)[offset:offset + len(coeffs)]
    probe_indices = rng.choice(len(coeffs), size=min(n_coeffs, len(coeffs)), replace=False)

    numeric_grad = numpy.zeros(len(probe_indices))
    for slot, k in enumerate(probe_indices):
        original = coeffs[k]
        coeffs[k] = original + step
        setter(coeffs)
        l_plus = loss()
        coeffs[k] = original - step
        setter(coeffs)
        l_minus = loss()
        coeffs[k] = original
        setter(coeffs)
        numeric_grad[slot] = (l_plus - l_minus) / (2 * step)

    return _rel_error(analytic_grad[probe_indices], numeric_grad)


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
        "radial loss grad": check_loss_grad(pot, atoms, "radial"),
        "species loss grad": check_loss_grad(pot, atoms, "species"),
        "linear loss grad": check_loss_grad(pot, atoms, "linear"),
    }
    bad = {name: err for name, err in results.items() if not (err <= tol)}
    if bad:
        raise AssertionError(f"gradient checks exceeded tol={tol}: {bad}")
    return results
