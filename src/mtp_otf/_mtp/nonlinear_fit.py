"""
Nonlinear MTP fitting — optimises radial basis coefficients.

Two backends are provided:

  NonlinearFitter(pot, backend="scipy")   — L-BFGS-B with full analytical EFS
                                            gradient; no extra dependencies.
  NonlinearFitter(pot, backend="torch")   — L-BFGS (strong Wolfe) or Adam with
                                            full analytical EFS gradient; requires
                                            PyTorch.

Both use a bi-level strategy:
  outer loop  — moves radial_basis_coeffs
  inner loop  — refits linear_coeffs + species_coeffs (LinearFitter)

LinearFitter is called only at the mlip-3 checkpoint steps
  {25, 70, 100, 150, 250, 400}  (and at the very start),
NOT at every outer iteration.  This matches mlip-3's NonLinOptimize behaviour
and reduces linear-solve overhead by 60-120× for typical run lengths.

Stress convention (matching mlip-3):
  Loss uses raw virial (eV) normalised by N atoms, NOT stress (eV/Å³) / 6.
  wgt_stress(cfg) = w_S / N  →  same as mlip-3's wgt_scale_power_stress = 1.

Loss normalisation (matching mlip-3):
  Total loss and gradient are divided by K = len(dataset) before the optimizer
  sees them, so absolute loss values are comparable across dataset sizes.
"""

import numpy as np

# Steps at which LinearFitter is called inside the outer loop — mirrors
# mlip-3's mtpr_trainer.cpp NonLinOptimize() explicit LinOptimize calls.
_LINOPT_STEPS = (25, 70, 100, 150, 250, 400)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _weighted_rmse(pot, dataset, w_E=1.0, w_F=1.0, w_S=0.1):
    """Scalar weighted RMSE across energy, forces, stresses."""
    from .design_matrix import _eval_basis

    lc = pot.get_linear_coeffs()
    sc = pot.get_species_coeffs()
    sq_sum = 0.0
    n_total = 0

    for entry in dataset:
        types = entry["types"]
        n_atoms = len(types)

        # Energy
        basis = _eval_basis(pot, entry)
        e_pred = (basis @ lc).sum() + sum(sc[t] for t in types)
        e_err = (e_pred - float(entry["energy"])) / n_atoms
        sq_sum += w_E * e_err**2
        n_total += 1

        # Forces
        if "forces" in entry and w_F > 0:
            result = pot.compute(entry["types"], entry["ilist"], entry["numneigh"], entry["firstneigh"], entry["displacements"], compute_virials=False, compute_eatom=False)
            f_err = (result["forces"] - entry["forces"]).ravel()
            sq_sum += w_F * np.dot(f_err, f_err) / (n_atoms * 3)
            n_total += 1

        # Stress
        if "stress" in entry and "volume" in entry and w_S > 0:
            result = pot.compute(entry["types"], entry["ilist"], entry["numneigh"], entry["firstneigh"], entry["displacements"], compute_virials=True, compute_eatom=False)
            v = result["virials"]
            vol = entry["volume"]
            s_pred = np.array([-v[0], -v[1], -v[2], -v[5], -v[4], -v[3]]) / vol
            s_err = s_pred - np.asarray(entry["stress"])
            sq_sum += w_S * np.dot(s_err, s_err) / 6
            n_total += 1

    return float(np.sqrt(sq_sum / max(n_total, 1)))


# ---------------------------------------------------------------------------
# Shared EFS + radial gradient computation
# ---------------------------------------------------------------------------


def _compute_efs_grad(pot, local_data, n_radial, w_E, w_F, w_S, comm):
    """Compute total loss and radial gradient for *local_data*.

    Does NOT call LinearFitter.  Linear/species coefficients are assumed
    already set on *pot*.

    Returns (total_loss, grad) both divided by K = len(full dataset) upstream.
    """
    total_loss = 0.0
    grad = np.zeros(n_radial, dtype=np.float64)

    for entry in local_data:
        n = len(entry["types"])
        has_stress = w_S > 0 and "stress" in entry and "volume" in entry

        energy, forces, virials, e_grad, fg, vg = pot.compute_with_radial_grad(
            entry["types"],
            entry["ilist"],
            entry["numneigh"],
            entry["firstneigh"],
            entry["displacements"],
            has_stress,
        )

        if w_E > 0:
            # w_E/N * E_err² — matches mlip-3: wgt_energy = w_E / N^wgt_scale_power_energy
            E_err = float(energy) - float(entry["energy"])
            total_loss += w_E / n * E_err**2
            grad += w_E / n * 2.0 * E_err * e_grad[:n_radial]

        if w_F > 0 and "forces" in entry:
            # w_F * Σ_ia f_err² — matches mlip-3: wgt_forces = w_F (wgt_scale_power_forces=0)
            f_err = (np.asarray(forces) - entry["forces"]).ravel()
            total_loss += w_F * np.dot(f_err, f_err)
            grad += w_F * 2.0 * f_err @ fg[:, :, :n_radial].reshape(n * 3, n_radial)

        if has_stress:
            vol = entry["volume"]
            # w_S/N * Σ_ab virial_err² — matches mlip-3: wgt_stress = w_S / N.
            # entry["stress"] is ASE Voigt (eV/Å³); multiply by vol → virial (eV).
            v_pred = np.array([-virials[0], -virials[1], -virials[2], -virials[5], -virials[4], -virials[3]])
            v_ref = np.asarray(entry["stress"]) * vol
            v_err = v_pred - v_ref
            total_loss += w_S / n * np.dot(v_err, v_err)
            vg_v = np.stack([-vg[0], -vg[1], -vg[2], -vg[5], -vg[4], -vg[3]])
            grad += w_S * 2.0 / n * v_err @ vg_v[:, :n_radial]

    if comm is not None:
        buf = np.array([total_loss])
        comm.Allreduce(buf.copy(), buf)
        total_loss = float(buf[0])
        comm.Allreduce(grad.copy(), grad)

    return total_loss, grad


# ---------------------------------------------------------------------------
# Scipy backend
# ---------------------------------------------------------------------------


class _ScipyNonlinearFitter:

    def __init__(self, pot, linear_fitter_kwargs, weight_energy, weight_forces, weight_stress, maxiter, eps, callback):
        self.pot = pot
        self.lf_kwargs = linear_fitter_kwargs
        self.w_E = weight_energy
        self.w_F = weight_forces
        self.w_S = weight_stress
        self.maxiter = maxiter
        self.eps = eps
        self.callback = callback
        self.loss_history_ = []

    def fit(self, dataset, comm=None):
        """Fit radial basis coefficients via L-BFGS-B with true bi-level gradient.

        LinearFitter is called at EVERY function evaluation (including line-search
        steps), so each gradient query sees optimal linear coefficients for the
        current radial point.  This computes the true bi-level gradient, which is
        larger than the frozen-linear gradient and prevents premature convergence of
        L-BFGS-B, allowing the full maxiter iterations to run.

        comm : mpi4py communicator or None
        """
        from scipy.optimize import minimize
        from .linear_fit import LinearFitter

        x0 = self.pot.get_radial_basis_coeffs().copy().ravel()
        n_radial = len(x0)

        if comm is not None:
            local_data = dataset[comm.Get_rank()::comm.Get_size()]
        else:
            local_data = dataset

        def loss_and_grad(x):
            self.pot.set_radial_basis_coeffs(x)
            LinearFitter(self.pot, **self.lf_kwargs).fit(dataset, comm=comm)
            total_loss, grad = _compute_efs_grad(self.pot, local_data, n_radial, self.w_E, self.w_F, self.w_S, comm)
            self.loss_history_.append(total_loss)
            if self.callback:
                self.callback(total_loss, self.pot)
            return total_loss, grad

        # ftol=0: disable function-value convergence criterion so the optimizer runs
        # until maxiter or line-search failure rather than terminating early when the
        # relative function decrease per step is small (default ftol ≈ 2e-9).
        # The line search can still fail (~171 steps for typical MTP) because
        # LinearFitter modifies the landscape non-smoothly; scipy handles this as
        # ABNORMAL termination but leaves x at a good local minimum.
        result = minimize(loss_and_grad, x0, method="L-BFGS-B", jac=True, options={"maxiter": self.maxiter, "ftol": 0})
        return result


# ---------------------------------------------------------------------------
# Torch backend
# ---------------------------------------------------------------------------


class _TorchNonlinearFitter:
    """PyTorch L-BFGS (strong Wolfe) or Adam outer loop.

    Same checkpoint structure as the Scipy backend: LinearFitter is called
    only at steps {25, 70, 100, 150, 250, 400} and at the very start.

    For L-BFGS: linear coefficients are fixed during the line search (closure
    does not call LinearFitter), matching the Scipy backend's behaviour within
    each segment.

    For Adam: one gradient step per iteration, LinearFitter at checkpoints.
    """

    def __init__(self, pot, linear_fitter_kwargs, weight_energy, weight_forces, weight_stress, maxiter, lr, optimizer, callback):
        self.pot = pot
        self.lf_kwargs = linear_fitter_kwargs
        self.w_E = weight_energy
        self.w_F = weight_forces
        self.w_S = weight_stress
        self.maxiter = maxiter
        self.lr = lr
        self.optimizer_name = optimizer
        self.callback = callback
        self.loss_history_ = []

    def fit(self, dataset, comm=None):
        import torch
        from .linear_fit import LinearFitter

        n_radial = (self.pot.get_coeff_count() - self.pot.get_alpha_scalar_count() - self.pot.get_species_count())
        K = len(dataset)

        x0 = np.array(self.pot.get_radial_basis_coeffs()).ravel()[:n_radial].copy()
        radial_t = torch.tensor(x0, dtype=torch.float64, requires_grad=True)

        if comm is not None:
            local_data = dataset[comm.Get_rank()::comm.Get_size()]
        else:
            local_data = dataset

        # LBFGS max_iter=5: strong-Wolfe line search does at most 5 closure evals
        # per outer step (vs 20 default), reducing overhead ~4× with little quality loss.
        if self.optimizer_name == "lbfgs":
            opt = torch.optim.LBFGS([radial_t], lr=self.lr, max_iter=5, line_search_fn="strong_wolfe")
        else:
            opt = torch.optim.Adam([radial_t], lr=self.lr)

        linopt_steps = set(s for s in _LINOPT_STEPS if s < self.maxiter)

        # Initial linear solve
        self.pot.set_radial_basis_coeffs(radial_t.detach().numpy())
        LinearFitter(self.pot, **self.lf_kwargs).fit(dataset, comm=comm)

        for step in range(self.maxiter):
            # Call LinearFitter at mlip-3 checkpoint steps (after accepted step)
            if step in linopt_steps:
                self.pot.set_radial_basis_coeffs(radial_t.detach().numpy())
                LinearFitter(self.pot, **self.lf_kwargs).fit(dataset, comm=comm)

            if self.optimizer_name == "lbfgs":

                def closure():
                    opt.zero_grad()
                    self.pot.set_radial_basis_coeffs(radial_t.detach().numpy())
                    total_loss, grad_np = _compute_efs_grad(self.pot, local_data, n_radial, self.w_E, self.w_F, self.w_S, comm)
                    total_loss /= K
                    grad_np /= K
                    radial_t.grad = torch.tensor(grad_np, dtype=torch.float64)
                    return torch.tensor(total_loss, dtype=torch.float64)

                loss_val = float(opt.step(closure))
            else:
                opt.zero_grad()
                self.pot.set_radial_basis_coeffs(radial_t.detach().numpy())
                total_loss, grad_np = _compute_efs_grad(self.pot, local_data, n_radial, self.w_E, self.w_F, self.w_S, comm)
                total_loss /= K
                grad_np /= K
                radial_t.grad = torch.tensor(grad_np, dtype=torch.float64)
                opt.step()
                loss_val = total_loss

            self.loss_history_.append(loss_val)
            if self.callback:
                self.callback(loss_val, self.pot)

        self.pot.set_radial_basis_coeffs(radial_t.detach().numpy())
        return self


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class NonlinearFitter:
    """Nonlinear MTP fitter that optimises radial basis coefficients.

    Parameters
    ----------
    pot : MTPPotential
        Potential to fit (modified in-place after fit()).
    backend : {"scipy", "torch"}
        "scipy"  — Scipy L-BFGS-B; full EFS analytical gradient; no extra deps.
                   Fastest backend.
        "torch"  — PyTorch L-BFGS (strong Wolfe) or Adam with full EFS gradient.
                   Requires PyTorch.
        Both backends use compute_with_radial_grad() for a fused EFS + full radial
        gradient in a single C++ pass; the loss / gradient formulas are identical.
        LinearFitter is called only at checkpoint steps {25, 70, 100, 150, 250, 400}
        (matching mlip-3), NOT at every iteration.
    weight_energy / weight_forces / weight_stress : float
        Observable weights (mlip-3 train defaults: 1.0 / 0.01 / 0.001).
        Stress loss uses virial (eV) / N — same convention as mlip-3.
    weight_scaling : int
        Per-config size exponent forwarded to LinearFitter / build_design_matrix.
    linear_fitter_kwargs : dict
        Keyword arguments forwarded to LinearFitter.
    maxiter : int
        Maximum outer iterations.
    eps : float
        Kept for API compatibility; not used.
    lr : float
        Learning rate (torch backend only).
    optimizer : {"lbfgs", "adam"}
        Torch optimizer (torch backend only).
    callback : callable(loss, pot) or None
        Called after each outer iteration with the current loss and potential.
    """

    def __init__(self, pot, *, backend="scipy", weight_energy=1.0, weight_forces=0.01, weight_stress=0.001, weight_scaling=1, linear_fitter_kwargs=None, maxiter=100, eps=1e-5, lr=1e-3, optimizer="lbfgs", callback=None):
        self.pot = pot
        self.backend = backend

        lf_kwargs = dict(weight_energy=weight_energy, weight_forces=weight_forces, weight_stress=weight_stress, weight_scaling=weight_scaling)
        if linear_fitter_kwargs:
            lf_kwargs.update(linear_fitter_kwargs)

        if backend == "scipy":
            self._impl = _ScipyNonlinearFitter(pot, lf_kwargs, weight_energy, weight_forces, weight_stress, maxiter, eps, callback)
        elif backend == "torch":
            self._impl = _TorchNonlinearFitter(pot, lf_kwargs, weight_energy, weight_forces, weight_stress, maxiter, lr, optimizer, callback)
        else:
            raise ValueError(f"Unknown backend '{backend}'. Choose 'scipy' or 'torch'.")

    def fit(self, dataset, comm=None):
        """Fit the potential on the dataset and update pot in-place.

        Parameters
        ----------
        dataset : list of entry dicts (see design_matrix module for format)
        comm : mpi4py communicator or None

        Returns
        -------
        self
        """
        self._impl.fit(dataset, comm=comm)
        return self

    @property
    def loss_history_(self):
        return self._impl.loss_history_
