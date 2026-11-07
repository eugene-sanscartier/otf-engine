"""Gradient gate for the trainer's loss under varpro and riemannian, against finite differences.

For each option set, the gradient `trainer_objective` returns at a trained potential, on the first 8
structures of a cfg, is compared with fourth-order finite differences along 4 random directions:
tangent to W W^T = I and along the QR retraction under riemannian, over the radial coefficients
under varpro, over every coefficient otherwise. Under riemannian the gradient must also be tangent.
A direction passes at its best of three step sizes. VarPro's gradient carries the error of the linear
solve, which the system's conditioning sets (5e-5 to 3e-4 relatively on tmp/set.cfg at level 16,
1e-6 with reg_param 1e-8), so its option sets pass under 1e-3, and riemannian alone under 1e-6.

    python tests/check_trainer_gradient.py --potential tmp/potential-16.almtp --cfg tmp/set.cfg --species Ni Si H
"""
import argparse
import sys

import numpy

from otf_engine._mtp import MTPTraining, _mtp_ext
from otf_engine.io_cfg import read_cfg
from otf_engine.mtp_backend import _labelled

# name: (options, tolerance)
OPTION_SETS = {
    "recipe": ({}, 1e-3),
    "riemannian": ({"varpro": "false"}, 1e-6),
    "varpro": ({"riemannian": "false"}, 1e-3),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--potential", required=True, help="a trained potential")
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--species", nargs="+", required=True)
    args = parser.parse_args()

    pot = MTPTraining(args.potential)
    with open(args.cfg) as f:
        structures = list(read_cfg(f, args.species))[:8]
    labelled = [_labelled(a, pot.get_max_cutoff(), k + 1) for k, a in enumerate(structures)]
    radial = numpy.asarray(pot.get_radial_basis_coeffs())
    P, K, R = radial.shape
    n_radial = radial.size
    coeffs = numpy.concatenate([radial.ravel(), numpy.ravel(pot.get_species_coeffs()), numpy.ravel(pot.get_linear_coeffs())])

    def rows(c):
        return c[:n_radial].reshape(P, K, R).transpose(1, 0, 2).reshape(K, P * R)

    def with_rows(c, W):
        out = c.copy()
        out[:n_radial] = W.reshape(K, P, R).transpose(1, 0, 2).ravel()
        return out

    def retract(c):
        q, r = numpy.linalg.qr(rows(c).T)    # W^T = Q R, W = R^T Q^T with R^T lower triangular
        return with_rows(c, (q * numpy.sign(numpy.diag(r))).T)

    failures = 0
    for name, (extra, tol) in OPTION_SETS.items():
        options = {"log": "none"} | extra
        riemannian, varpro = options.get("riemannian") != "false", options.get("varpro") != "false"
        x = retract(coeffs) if riemannian else coeffs
        objective = lambda c: _mtp_ext.trainer_objective(args.potential, labelled, options, c)
        grad = numpy.asarray(objective(x)[1])
        W, G = rows(x), rows(grad)
        tangency = numpy.linalg.norm(G @ W.T + W @ G.T) / 2 if riemannian else 0.0

        rng = numpy.random.default_rng(0)
        worst = 0.0
        for _ in range(4):
            xi = rng.standard_normal(len(x))
            if varpro: xi[n_radial:] = 0.0
            if riemannian:
                Z = rows(xi)
                xi = with_rows(xi, Z - (Z @ W.T + W @ Z.T) / 2 @ W)
            xi /= numpy.linalg.norm(xi)
            move = (lambda c: retract(c)) if riemannian else (lambda c: c)
            analytic = grad @ xi
            errors = []
            for h in (1e-3, 3e-4, 1e-4):
                f = {m: objective(move(x + m * h * xi))[0] for m in (-2, -1, 1, 2)}
                fd = (8 * (f[1] - f[-1]) - (f[2] - f[-2])) / (12 * h)
                errors += [abs(fd - analytic) / abs(analytic)]
            worst = max(worst, min(errors))

        ok = worst < tol and tangency < 1e-12 * max(1.0, numpy.linalg.norm(grad))
        failures += not ok
        print(f"{name:<11} worst relative error {worst:.1e}, |sym(G W^T)| {tangency:.1e} — {'ok' if ok else 'FAILED'}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
