"""
Rescale the MTP potential's `scaling` parameter to minimise a heuristic
condition number on the linear coefficients.

Mirrors mlip-3's `MtprTrainer::Rescale()` in `mtpr_trainer.cpp:627-701`:
iteratively searches a multiplicative bracket around the current scaling and
picks the value that minimises `rms(linear_coeffs) / median(|linear_coeffs|)`.
"""

import numpy as np

from .linear_fit import LinearFitter


def rescale(pot, dataset, lf_kwargs, max_iter=10):
    """Adjust `pot.scaling` to reduce linear-coefficient ill-conditioning.

    Performs up to `max_iter` bracket iterations.  Each iteration tests five
    candidate scaling values around the current one, selects the best by the
    condition-number heuristic, and stops when the centre value is optimal.
    A final LinearFitter.fit() is run at the chosen scaling.

    Parameters
    ----------
    pot : MTPPotential
        Modified in-place.
    dataset : list of entry dicts
        Training data (same format as LinearFitter / build_design_matrix).
    lf_kwargs : dict
        Keyword arguments forwarded to LinearFitter (weights, weight_scaling).
    max_iter : int
        Maximum bracket iterations.
    """
    factors = [1.0 / 1.2, 1.0 / 1.1, 1.0, 1.1, 1.2]

    def _cond(s):
        pot.set_scaling(s)
        LinearFitter(pot, **lf_kwargs).fit(dataset)
        c = pot.get_linear_coeffs()
        rms = np.sqrt(np.mean(c**2))
        med = np.median(np.abs(c))
        return rms / (med + 1e-30)

    for _ in range(max_iter):
        s0 = pot.get_scaling()
        conds = [_cond(s0 * f) for f in factors]
        best = int(np.argmin(conds))
        if best == 2:  # centre is already optimal — converged
            break
        pot.set_scaling(s0 * factors[best])

    LinearFitter(pot, **lf_kwargs).fit(dataset)
