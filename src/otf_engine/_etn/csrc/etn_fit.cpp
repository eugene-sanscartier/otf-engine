/* -*- c++ -*- ----------------------------------------------------------
   Fits an ETN to reference energies, forces and stresses.
   Algorithm of mlip-4's Train and ETN::Rescale; see etn_fit.h.
   BSD 2-Clause, see mlip-4's LICENSE.
------------------------------------------------------------------------- */

#include "etn_fit.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>

ETNFit::ETNFit(ETN& potential, std::vector<FitStructure> structures, const ETNFitOptions& options, Comm& comm, std::ostream* log, std::function<void()> checkpoint)
    : Fit(potential, std::move(structures), options, comm, log, std::move(checkpoint)) {}

/* ---------------------------------------------------------------------- */
void ETNFit::add_species() {
    const int old_spec = potential.get_species_count();
    const int new_spec = training_species_count();
    if (old_spec == new_spec) return;

    log_added_species(old_spec, new_spec);
    potential.set_species_count(new_spec);
    if (!potential.is_trained())
        potential.fill_initial_coeffs();
    else
        for (int l = 0; l < (int) potential.get_species_ranks().size(); l++) {
            const int rank = potential.get_species_ranks()[l];
            std::vector<double> a = potential.get_species_pair_coeffs(l);
            for (int pair = 0; pair < new_spec * new_spec; pair++) {
                if (pair / new_spec < old_spec && pair % new_spec < old_spec) continue;
                std::fill_n(a.begin() + pair * rank, rank, 1e-6);
                a[pair * rank] = 1e-3 * (pair + 1);
            }
            potential.set_species_pair_coeffs(l, a.data());
        }

    load_coeffs();
}

/* ---------------------------------------------------------------------- */
void ETNFit::run() {
    const auto start = std::chrono::steady_clock::now();
    comm.barrier();
    log_header("ETN");

    add_species();
    if (log) *log << "coefficients: " << coeffs.size() << std::endl;
    log_errors("Training set errors before training:");

    // mlip-4's pre-training keeps tolerance and rescale, and takes the other settings' defaults
    if (!options.skip_preinit) {
        if (log) *log << "pre-training, 75 steps at most" << std::endl;
        ETNFitOptions pretraining;
        pretraining.iteration_limit = 75;
        pretraining.max_func_eval = 75 * 20;
        pretraining.tolerance = options.tolerance;
        pretraining.rescale = options.rescale;
        fit_nonlinear(pretraining);
    }
    if (log) *log << "main fit" << std::endl;
    fit_nonlinear(options);

    potential.set_coeffs(coeffs.data());
    potential.set_trained(true);
    log_errors("Training set errors:");
    if (log) *log << "Trained in " << std::round(std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() * 10) / 10 << " s" << std::endl;
    comm.barrier();
}

/* ----------------------------------------------------------------------
   mlip-4's Train: rank 0 steps BFGS on the loss and gradient reduced onto
   it, and broadcasts the coefficients with the step count and the stop and
   rescale flags at the top of every loop.
------------------------------------------------------------------------- */
void ETNFit::fit_nonlinear(const ETNFitOptions& o) {
    const auto start = std::chrono::steady_clock::now();
    const int n = (int) coeffs.size();
    std::vector<double> loss_data(4 + n);    // [loss | energy, force and stress terms | gradient], reduced onto rank 0
    std::vector<double> grad(n), bfgs_g(n);

    std::vector<double> shared(3 + n, 0.0);    // [step | stop | rescale | coefficients], broadcast from rank 0
    double& num_iter = shared[0];
    double& break_flag = shared[1];
    double& rescale_flag = shared[2];
    std::copy(coeffs.begin(), coeffs.end(), shared.begin() + 3);

    WolfeBFGS bfgs;
    if (rank == 0) {
        bfgs.set_wolfe_c1(o.wolfe_c1);
        bfgs.set_wolfe_c2(o.wolfe_c2);
        bfgs.max_iters_no_improve = o.bfgs_max_iters_no_improve;
        bfgs.set_log(log);
        bfgs.set_x(coeffs.data(), n);
    }

    int num_func_eval = 0;
    double linf = 9e99;
    double bfgs_f = 0.0, bfgs_terms[3] = {};
    bool attempted_rescale = false;
    bool is_new_step = true;
    int convergence_check_delay = 0;
    if (rank == 0) rescale_flag = num_iter == 0;    // ETN::rescale_this_iter

    while (true) {
        comm.bcast(shared.data(), 3 + n, 0);
        std::copy(shared.begin() + 3, shared.end(), coeffs.begin());
        if (break_flag) break;

        if (o.rescale && rescale_flag) {
            rescale();
            rescale_flag = 0;
            attempted_rescale = true;
            std::copy(coeffs.begin(), coeffs.end(), shared.begin() + 3);
            if (rank == 0) {
                bfgs.set_x(coeffs.data(), n);
                bfgs.set_identity_as_inv_hess();
                convergence_check_delay = std::max(300, convergence_check_delay);
            }
            continue;
        }
        if (rank == 0 && is_new_step) {
            step_losses_.push_back({});
            is_new_step = false;
        }

        loss_data[0] = loss_grad(grad);
        std::copy(loss_terms, loss_terms + 3, loss_data.begin() + 1);
        std::copy(grad.begin(), grad.end(), loss_data.begin() + 4);
        std::vector<double> total(4 + n);
        comm.reduce(loss_data.data(), total.data(), 4 + n, ReduceOp::sum, 0);
        num_func_eval++;
        if (rank != 0) continue;

        const double inv_cfg_count = 1.0 / structure_count;
        bfgs_f = total[0] * inv_cfg_count;
        for (int k = 0; k < 3; k++) bfgs_terms[k] = total[1 + k] * inv_cfg_count;
        for (int i = 0; i < n; i++) bfgs_g[i] = total[4 + i] * inv_cfg_count;
        step_losses_.back().push_back(bfgs_f);

        bool loss_decrease;
        if (attempted_rescale) {
            loss_decrease = bfgs.make_nonlinesearch_iteration(bfgs_f, bfgs_g);
            attempted_rescale = false;
        } else
            loss_decrease = bfgs.iterate(bfgs_f, bfgs_g);
        if (!loss_decrease) {
            break_flag = 1;
            if (log) *log << "stop: BFGS cannot lower the loss" << std::endl;
        }

        if (!bfgs.in_linesearch() && !break_flag) {
            if (num_iter == 0) log_step_header(false);    // after the first step's rescaling
            log_step((int) num_iter, bfgs_f, bfgs_terms, false, bfgs.alpha(), (int) step_losses_.back().size());
            num_iter++;
            is_new_step = true;
            if (o.rescale) rescale_flag = num_iter == 0;

            if ((int) num_iter % 1500 == 0) {
                if (log) *log << "inverse Hessian made diagonal, as every 1500 steps" << std::endl;
                bfgs.diag_inv_hess();
                convergence_check_delay = std::max(convergence_check_delay, 300);
            }

            // A normal step unless this step rescales
            if (!o.rescale || !rescale_flag)
                if (!bfgs.make_nonlinesearch_iteration(bfgs_f, bfgs_g)) {
                    break_flag = 1;
                    if (log) *log << "stop: BFGS cannot lower the loss" << std::endl;
                }

            if ((int) num_iter % 100 == 0) {
                if (convergence_check_delay <= 0 && linf > bfgs_f && (linf - bfgs_f) / bfgs_f < o.tolerance && !break_flag) {
                    break_flag = 1;
                    if (log) *log << "stop: the last 100 steps lowered the loss by less than tolerance" << std::endl;
                }
                convergence_check_delay -= 100;
                linf = bfgs_f;
            }

            if (num_iter >= o.iteration_limit && !break_flag) {
                break_flag = 1;
                if (log) *log << "stop: iteration limit reached" << std::endl;
            }
            // Checked only between line searches, as mlip-4 checks it
            if (num_func_eval > o.max_func_eval && !break_flag) {
                break_flag = 1;
                if (log) *log << "stop: evaluation limit (max_func_eval) reached" << std::endl;
            }

            save_checkpoint();
        }

        // The step that met a stopping rule is dropped
        if (!break_flag) std::copy(bfgs.x().begin(), bfgs.x().end(), shared.begin() + 3);
    }
    log_phase_time((int) shared[0], start);
}

/* ----------------------------------------------------------------------
   ETN::Rescale: from alpha = 0, mlip-4's line search along the coefficients
   x0 as they are, its sign chosen to descend, on the loss summed over the
   structures, for at most 20 trials or until alpha moves by under 1e-5.
------------------------------------------------------------------------- */
void ETNFit::rescale() {
    const int n = (int) coeffs.size();
    const std::vector<double> x0 = coeffs;
    std::vector<double> grad(n);

    WolfeLineSearch linesearch;
    if (rank == 0) linesearch.log = log;
    double alpha_prev = 0.0;
    double alpha_curr = 1.0;
    int sign = 1;
    linesearch.reset(alpha_curr);

    auto loss_and_der = [&](double& loss, double& loss_der) {
        loss = loss_grad(grad);
        loss_der = 0;
        for (int i = 0; i < n; i++)
            loss_der += grad[i] * sign * x0[i];
        double local[2] = {loss, loss_der}, total[2];
        comm.allreduce(local, total, 2, ReduceOp::sum);
        loss = total[0];
        loss_der = total[1];
    };

    double loss, loss_der;
    std::fill(coeffs.begin(), coeffs.end(), 0.0);
    loss_and_der(loss, loss_der);
    if (loss_der > 0) {
        sign *= -1;
        loss_der *= -1;
    }
    linesearch.iterate(loss, loss_der);

    for (int iter = 0; iter < 20 && std::abs(alpha_curr - alpha_prev) > 1e-5; iter++) {
        for (int i = 0; i < n; i++)
            coeffs[i] = sign * alpha_curr * x0[i];
        loss_and_der(loss, loss_der);
        linesearch.iterate(loss, loss_der);
        alpha_prev = alpha_curr;
        alpha_curr = linesearch.x();
    }

    for (int i = 0; i < n; i++)
        coeffs[i] = sign * alpha_curr * x0[i];
    if (log) *log << "rescaled: coefficients times " << sign * alpha_curr << std::endl;
}

/* ----------------------------------------------------------------------
   mlip-4's LossCfg over this rank's structures: the weighted squared
   residuals of the energy, the virial (xx, yy, zz, yz, xz, xy, the last
   three doubled) and the forces, in that order, and their gradient.
------------------------------------------------------------------------- */
double ETNFit::loss_grad(std::vector<double>& grad) {
    static constexpr int STRESS_ORDER[6] = {0, 1, 2, 5, 4, 3};    // mlip-4's stress order, as virial indices
    potential.set_coeffs(coeffs.data());
    std::fill(grad.begin(), grad.end(), 0.0);
    std::fill(loss_terms, loss_terms + 3, 0.0);
    double loss = 0.0;

    for (const FitStructure& s : structures) {
        const NeighList list = s.list();
        potential.compute(list);
        const double* forces = potential.get_forces().data();
        const double* virial = potential.get_virial().data();

        double structure_loss = 0.0;
        double dloss_denergy = 0.0;
        double dloss_dvirial[6] = {};
        dloss_dforces.assign((size_t) list.n_atoms * 3, 0.0);

        if (s.has_energy) {
            const double diff = potential.get_energy() - s.energy;
            structure_loss += options.energy_weight * diff * diff;
            loss_terms[0] += options.energy_weight * diff * diff;
            dloss_denergy = 2 * options.energy_weight * diff;
        }
        if (s.has_virial)
            for (int c = 0; c < 6; c++) {
                const int k = STRESS_ORDER[c];
                const double wgt = (c < 3 ? 1 : 2) * options.stress_weight;
                const double diff = virial[k] - s.virial[k];
                structure_loss += wgt * diff * diff;
                loss_terms[2] += wgt * diff * diff;
                dloss_dvirial[k] = 2 * wgt * diff;
            }
        if (s.has_forces)
            for (int i = 0; i < list.n_atoms * 3; i++) {
                const double wgt = options.balance_species ? options.force_weight * s.species_factors[s.types[i / 3]] : options.force_weight;
                const double diff = forces[i] - s.forces[i];
                structure_loss += wgt * diff * diff;
                loss_terms[1] += wgt * diff * diff;
                dloss_dforces[i] = 2 * wgt * diff;
            }

        loss += structure_loss;
        potential.eval_loss_grad(list, dloss_denergy, dloss_dforces.data(), dloss_dvirial, grad.data());
    }
    return loss;
}
