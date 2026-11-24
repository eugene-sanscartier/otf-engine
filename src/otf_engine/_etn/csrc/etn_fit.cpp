/* -*- c++ -*- ----------------------------------------------------------
   Fits an ETN to reference energies, forces and stresses.
   Algorithm of mlip-4's Train and ETN::Rescale; see etn_fit.h.
   BSD 2-Clause, see mlip-4's LICENSE.
------------------------------------------------------------------------- */

#include "etn_fit.h"

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
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
        potential.reset_coeffs();
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
void ETNFit::fit_species_energies() {
    const int S = potential.get_species_count();
    const int first = (int) coeffs.size() - S;
    potential.set_coeffs(coeffs.data());

    std::vector<double> local(S * S + S, 0.0), total(S * S + S);    // [counts counts^T | counts residual]
    for (const FitStructure& s : structures) {
        if (!s.has_energy) continue;
        potential.compute(s.list());
        std::vector<double> counts(S, 0.0);
        for (int i : s.ilist) counts[s.types[i]] += 1;
        double residual = s.energy - potential.get_energy();
        for (int a = 0; a < S; a++) residual += counts[a] * coeffs[first + a];
        for (int a = 0; a < S; a++) {
            for (int b = 0; b < S; b++) local[a * S + b] += counts[a] * counts[b];
            local[S * S + a] += counts[a] * residual;
        }
    }
    comm.allreduce(local.data(), total.data(), S * S + S, ReduceOp::sum);

    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> matrix(total.data(), S, S);
    Eigen::Map<Eigen::VectorXd> vector(total.data() + S * S, S);
    matrix.diagonal().array() += 1e-12 * matrix.diagonal().maxCoeff();
    const Eigen::VectorXd energies = matrix.ldlt().solve(vector);
    std::ostringstream text;
    for (int a = 0; a < S; a++) {
        coeffs[first + a] = energies[a];
        text << (a > 0 ? " " : "") << energies[a];
    }
    log_event(log, "init", "species energies %s", text.str().c_str());
}

/* ---------------------------------------------------------------------- */
void ETNFit::run() {
    const auto start = std::chrono::steady_clock::now();
    comm.barrier();
    log_header("ETN");

    add_species();
    log_event(log, "coefficients", "%zu", coeffs.size());
    if (options.varpro && options.varpro_fraction < 1.0) draw_varpro_subset(0);
    log_errors("before training");    // the potential as given, before update_min_dist moves its radial basis
    update_min_dist();

    if (options.init_random && !potential.is_trained()) {
        if (rank == 0) {
            std::random_device rand_device;
            std::default_random_engine generator(options.random_seed >= 0 ? (unsigned) options.random_seed : rand_device());
            std::uniform_real_distribution<> phase(0.0, 2 * M_PI);
            log_event(log, "init", "init_random, init_scale %g", options.init_scale);
            for (double& c : coeffs) c = options.init_scale * std::sin(phase(generator));
        }
        comm.bcast(coeffs.data(), (int) coeffs.size(), 0);
    }
    if (options.init_species && !potential.is_trained()) fit_species_energies();

    // mlip-4's pre-training keeps tolerance and rescale, and takes the other settings' defaults
    if (!options.skip_preinit) {
        log_event(log, "phase", "pre-training, iteration_limit 75");
        ETNFitOptions pretraining;
        pretraining.iteration_limit = 75;
        pretraining.max_func_eval = 75 * 20;
        pretraining.tolerance = options.tolerance;
        pretraining.rescale = options.rescale;
        fit_nonlinear(pretraining);
    }
    log_event(log, "phase", "main, iteration_limit %d", options.iteration_limit);
    if (options.reg_at_main) ridge.clear();
    fit_nonlinear(options);

    potential.set_coeffs(coeffs.data());
    potential.set_trained(true);
    log_errors("after training");
    log_fit_time(start);
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

    const std::vector<int> linear = options.varpro ? linear_columns() : std::vector<int>();
    std::vector<bool> is_linear(n, false);
    for (int k : linear) is_linear[k] = true;

    WolfeBFGS bfgs;
    if (rank == 0) {
        bfgs.set_wolfe_c1(o.wolfe_c1);
        bfgs.set_wolfe_c2(o.wolfe_c2);
        bfgs.max_iters_no_improve = o.bfgs_max_iters_no_improve;
        bfgs.set_x(coeffs.data(), n);
    }

    int num_func_eval = 0;
    double linf = 9e99;
    double bfgs_f = 0.0, bfgs_terms[3] = {};
    double kept_f = 0.0, kept_terms[3] = {};    // the loss at the point BFGS keeps, which a stalled step returns to
    const char* inv_hess = "";                  // what the fit set the inverse Hessian to before the next step
    bool attempted_rescale = false;
    bool is_new_step = true;
    int convergence_check_delay = 0;
    const bool rescales = o.rescale && !options.varpro;
    // varpro_redraw_every: the main fit's draw at each step, the last one kept over the last 20% of its steps
    const int redraw_every = &o == &options && options.varpro && options.varpro_fraction < 1.0 ? options.varpro_redraw_every : 0;
    auto varpro_draw_at = [&](int step) { return redraw_every > 0 ? std::min(step, (int) std::ceil(0.8 * o.iteration_limit) - 1) / redraw_every : 0; };
    int varpro_drawn = 0;
    if (rank == 0) rescale_flag = num_iter == 0;    // ETN::rescale_this_iter

    while (true) {
        comm.bcast(shared.data(), 3 + n, 0);
        std::copy(shared.begin() + 3, shared.end(), coeffs.begin());
        if (break_flag) break;

        if (rescales && rescale_flag) {
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

        if (varpro_draw_at((int) num_iter) != varpro_drawn) {
            varpro_drawn = varpro_draw_at((int) num_iter);
            draw_varpro_subset(varpro_drawn);
        }
        loss_data[0] = reduced_loss_grad(grad);
        for (int i : linear) grad[i] = 0.0;
        std::copy(loss_terms, loss_terms + 3, loss_data.begin() + 1);
        std::copy(grad.begin(), grad.end(), loss_data.begin() + 4);
        std::vector<double> total(4 + n);
        comm.reduce(loss_data.data(), total.data(), 4 + n, ReduceOp::sum, 0);
        num_func_eval++;
        if (rank != 0) continue;

        const double inv_cfg_count = 1.0 / structure_count;
        // Under varpro BFGS minimises the loss plus the ridge, whose minimum over the linear coefficients
        // solve_linear finds, so that the gradient without their part is the reduced loss's
        double ridge_term = 0.0;
        for (int a = 0; a < (int) linear.size(); a++) ridge_term += ridge[a] * coeffs[linear[a]] * coeffs[linear[a]];
        bfgs_f = (total[0] + ridge_term) * inv_cfg_count;
        for (int k = 0; k < 3; k++) bfgs_terms[k] = total[1 + k] * inv_cfg_count;
        for (int i = 0; i < n; i++) bfgs_g[i] = total[4 + i] * inv_cfg_count;
        step_losses_.back().push_back(bfgs_f);
        if (attempted_rescale || num_func_eval == 1) {
            kept_f = bfgs_f;
            std::copy(bfgs_terms, bfgs_terms + 3, kept_terms);
        }

        // Under varpro, which does not rescale, the first step moves the other coefficients by 1% of their norm
        if (options.varpro && num_func_eval == 1) {
            double x_norm = 0.0, g_norm = 0.0;
            for (int i = 0; i < n; i++)
                if (!is_linear[i]) {
                    x_norm += coeffs[i] * coeffs[i];
                    g_norm += bfgs_g[i] * bfgs_g[i];
                }
            if (g_norm > 0) bfgs.scale_inv_hess(0.01 * std::sqrt(x_norm / g_norm));
        }

        bool loss_decrease;
        if (attempted_rescale) {
            loss_decrease = bfgs.make_nonlinesearch_iteration(bfgs_f, bfgs_g);
            attempted_rescale = false;
        } else
            loss_decrease = bfgs.iterate(bfgs_f, bfgs_g);
        if (!loss_decrease) {
            break_flag = 1;
            log_event(log, "stop", "bfgs, %s", bfgs.message().c_str());
        }

        if (!bfgs.in_linesearch() && !break_flag) {
            if (num_iter == 0) log_step_header({false, false, true});    // after the first step's rescaling
            if (bfgs.alpha() != 0.0) {
                kept_f = bfgs_f;
                std::copy(bfgs_terms, bfgs_terms + 3, kept_terms);
            }
            log_step((int) num_iter, kept_f, kept_terms, bfgs.alpha(), (int) step_losses_.back().size(), NAN, *bfgs.inv_hess_set() ? bfgs.inv_hess_set() : inv_hess);
            inv_hess = "";
            num_iter++;
            is_new_step = true;
            if (rescales) rescale_flag = num_iter == 0;

            if ((int) num_iter % 1500 == 0) {
                bfgs.diag_inv_hess();
                inv_hess = "diag";
                convergence_check_delay = std::max(convergence_check_delay, 300);
            }

            // A new varpro draw restarts BFGS's line search from this point, under the new loss; otherwise a normal
            // step unless this step rescales
            if (varpro_draw_at((int) num_iter) != varpro_draw_at((int) num_iter - 1))
                bfgs.set_x(bfgs.x().data(), n);
            else if (!rescales || !rescale_flag)
                if (!bfgs.make_nonlinesearch_iteration(bfgs_f, bfgs_g)) {
                    break_flag = 1;
                    log_event(log, "stop", "bfgs, %s", bfgs.message().c_str());
                }

            if ((int) num_iter % 100 == 0) {
                if (convergence_check_delay <= 0 && linf > bfgs_f && (linf - bfgs_f) / bfgs_f < o.tolerance && !break_flag) {
                    break_flag = 1;
                    log_event(log, "stop", "(loss[k-100] - loss[k]) / loss[k] = %.1e < tolerance %g", (linf - bfgs_f) / bfgs_f, o.tolerance);
                }
                convergence_check_delay -= 100;
                linf = bfgs_f;
            }

            if (num_iter >= o.iteration_limit && !break_flag) {
                break_flag = 1;
                log_event(log, "stop", "iteration_limit %d", o.iteration_limit);
            }
            // Checked only between line searches, as mlip-4 checks it
            if (num_func_eval > o.max_func_eval && !break_flag) {
                break_flag = 1;
                log_event(log, "stop", "max_func_eval %d", o.max_func_eval);
            }

            save_checkpoint();
        }

        // The step that met a stopping rule is dropped
        if (!break_flag) std::copy(bfgs.x().begin(), bfgs.x().end(), shared.begin() + 3);
    }
    if (options.varpro) solve_linear();
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
    log_event(log, "rescale", "alpha % .7e  loss %.7e", 0.0, loss / structure_count);
    if (loss_der > 0) {
        sign *= -1;
        loss_der *= -1;
    }
    if (!linesearch.iterate(loss, loss_der)) log_event(log, "rescale", "%s", linesearch.message.c_str());

    for (int iter = 0; iter < 20 && std::abs(alpha_curr - alpha_prev) > 1e-5; iter++) {
        for (int i = 0; i < n; i++)
            coeffs[i] = sign * alpha_curr * x0[i];
        loss_and_der(loss, loss_der);
        log_event(log, "rescale", "alpha % .7e  loss %.7e", sign * alpha_curr, loss / structure_count);
        if (!linesearch.iterate(loss, loss_der)) log_event(log, "rescale", "%s", linesearch.message.c_str());
        alpha_prev = alpha_curr;
        alpha_curr = linesearch.x();
    }

    for (int i = 0; i < n; i++)
        coeffs[i] = sign * alpha_curr * x0[i];
    log_event(log, "rescale", "alpha % .7e", sign * alpha_curr);
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

    // varpro's last solve kept each structure's columns at these coefficients but the linear ones: the outputs are their product with them
    std::vector<int> linear;
    bool from_rows = false;
    if (!linear_rows_at.empty() && linear_rows.size() == structures.size()) {
        linear = linear_columns();
        std::vector<double> at = coeffs;
        for (int k : linear) at[k] = 0.0;
        from_rows = at == linear_rows_at;
    }

    for (size_t index = 0; index < structures.size(); index++) {
        const FitStructure& s = structures[index];
        const NeighList list = s.list();
        double energy;
        const double *forces, *virial;
        if (from_rows && !linear_rows[index].empty()) {
            const int stride = 7 + 3 * list.n_atoms;
            row_outputs.assign(stride, 0.0);
            for (size_t k = 0; k < linear.size(); k++) {
                const double x = coeffs[linear[k]];
                const double* row = linear_rows[index].data() + k * stride;
                for (int r = 0; r < stride; r++) row_outputs[r] += x * row[r];
            }
            energy = row_outputs[0];
            virial = row_outputs.data() + 1;
            forces = row_outputs.data() + 7;
        } else {
            potential.compute(list);
            energy = potential.get_energy();
            forces = potential.get_forces().data();
            virial = potential.get_virial().data();
        }

        double structure_loss = 0.0;
        double dloss_denergy = 0.0;
        double dloss_dvirial[6] = {};
        dloss_dforces.assign((size_t) list.n_atoms * 3, 0.0);

        if (s.has_energy) {
            const double wgt = energy_weight(s);
            const double diff = energy - s.energy;
            structure_loss += wgt * diff * diff;
            loss_terms[0] += wgt * diff * diff;
            dloss_denergy = 2 * wgt * diff;
        }
        if (s.has_virial)
            for (int c = 0; c < 6; c++) {
                const int k = STRESS_ORDER[c];
                const double wgt = virial_weight(s, k);
                const double diff = virial[k] - s.virial[k];
                structure_loss += wgt * diff * diff;
                loss_terms[2] += wgt * diff * diff;
                dloss_dvirial[k] = 2 * wgt * diff;
            }
        if (s.has_forces)
            for (int i = 0; i < list.n_atoms * 3; i++) {
                const double wgt = force_weight(s, i);
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

/* ----------------------------------------------------------------------
   Under varpro_fraction, x(theta) minimizes the fraction's loss and the
   loss takes every structure, so dL/dx is not zero: its path through the
   solve is -d/dtheta of sum w (J x - y) . J mu over the fraction, at fixed
   x and mu = A^-1 dL/dx, with w the solve's weights. Each structure's rows
   give J x - y and J mu, and eval_loss_grad the derivative of each factor.
------------------------------------------------------------------------- */
double ETNFit::reduced_loss_grad(std::vector<double>& grad) {
    const bool subset = options.varpro && options.varpro_fraction < 1.0;
    if (options.varpro) solve_linear(subset);
    const double loss = loss_grad(grad);
    if (!subset) return loss;

    const std::vector<int> linear = linear_columns();
    const int m = (int) linear.size();
    std::vector<double> local_slope(m), slope(m);    // dL/dx with the ridge
    for (int a = 0; a < m; a++) local_slope[a] = grad[linear[a]];
    comm.allreduce(local_slope.data(), slope.data(), m, ReduceOp::sum);
    for (int a = 0; a < m; a++) slope[a] += 2 * ridge[a] * coeffs[linear[a]];
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> matrix(linear_system.data(), m, m);
    const Eigen::VectorXd mu = matrix.ldlt().solve(Eigen::Map<const Eigen::VectorXd>(slope.data(), m));
    std::vector<double> at_mu = coeffs;
    for (int a = 0; a < m; a++) at_mu[linear[a]] = mu[a];

    const double subset_scale = 1.0 / options.varpro_fraction;
    for (bool by_mu : {false, true}) {
        // at x, seeded by J mu; at mu, seeded by J x - y
        potential.set_coeffs(by_mu ? at_mu.data() : coeffs.data());
        for (size_t index = 0; index < structures.size(); index++) {
            if (linear_rows[index].empty()) continue;
            const FitStructure& s = structures[index];
            const NeighList list = s.list();
            const int stride = 7 + 3 * list.n_atoms;
            std::vector<double> seed(stride, 0.0);
            for (int a = 0; a < m; a++) {
                const double factor = by_mu ? coeffs[linear[a]] : mu[a];
                const double* row = linear_rows[index].data() + (size_t) a * stride;
                for (int r = 0; r < stride; r++) seed[r] += factor * row[r];
            }
            const double energy_seed = s.has_energy ? -subset_scale * energy_weight(s) * (seed[0] - (by_mu ? s.energy : 0.0)) : 0.0;
            double virial_seed[6] = {};
            if (s.has_virial)
                for (int k = 0; k < 6; k++) virial_seed[k] = -subset_scale * virial_weight(s, k) * (seed[1 + k] - (by_mu ? s.virial[k] : 0.0));
            dloss_dforces.assign((size_t) list.n_atoms * 3, 0.0);
            if (s.has_forces)
                for (int i = 0; i < list.n_atoms * 3; i++) dloss_dforces[i] = -subset_scale * force_weight(s, i) * (seed[7 + i] - (by_mu ? s.forces[i] : 0.0));
            potential.eval_loss_grad(list, energy_seed, dloss_dforces.data(), virial_seed, grad.data());
        }
    }
    potential.set_coeffs(coeffs.data());
    return loss;
}

double ETNFit::objective(const std::vector<double>& at, std::vector<double>& grad) {
    std::copy(at.begin(), at.end(), coeffs.begin());
    if (options.varpro && options.varpro_fraction < 1.0) draw_varpro_subset(0);
    const int n = (int) coeffs.size();
    std::vector<double> local_grad(n);
    const double local = reduced_loss_grad(local_grad);
    double loss = 0.0;
    grad.assign(n, 0.0);
    comm.allreduce(&local, &loss, 1, ReduceOp::sum);
    comm.allreduce(local_grad.data(), grad.data(), n, ReduceOp::sum);
    if (options.varpro) {
        const std::vector<int> linear = linear_columns();
        for (int a = 0; a < (int) linear.size(); a++) {
            loss += ridge[a] * coeffs[linear[a]] * coeffs[linear[a]];
            grad[linear[a]] = 0.0;
        }
    }
    loss /= structure_count;
    for (double& g : grad) g /= structure_count;
    return loss;
}

/* ---------------------------------------------------------------------- */
double ETNFit::energy_weight(const FitStructure& s) const {
    return options.energy_weight;
}

double ETNFit::force_weight(const FitStructure& s, int i) const {
    return options.balance_species ? options.force_weight * s.species_factors[s.types[i / 3]] : options.force_weight;
}

double ETNFit::virial_weight(const FitStructure& s, int k) const {
    return (k < 3 ? 1 : 2) * options.stress_weight;
}

/* ---------------------------------------------------------------------- */
std::vector<int> ETNFit::linear_columns() const {
    const auto [begin, end] = core_block(potential.get_core_count() - 1, false);
    std::vector<int> columns;
    for (int k = begin; k < end; k++) columns.push_back(k);
    const int S = potential.get_species_count();
    for (int a = 0; a < S; a++) columns.push_back((int) coeffs.size() - S + a);
    return columns;
}

std::vector<std::pair<int, int>> ETNFit::blocks() const {
    std::vector<std::pair<int, int>> ranges;
    int at = 0;
    auto add = [&](int size) {
        ranges.push_back({at, at + size});
        at += size;
    };
    const int S = potential.get_species_count();
    for (int l = 0; l < (int) potential.get_species_ranks().size(); l++) add((int) potential.get_radial_basis_coeffs(l).size());
    for (int l = 0; l < (int) potential.get_species_ranks().size(); l++) add((int) potential.get_species_pair_coeffs(l).size());
    for (bool imaginary : {false, true})
        for (int i = 0; i < potential.get_core_count(); i++) add((int) potential.get_core_coeffs(i, imaginary).size());
    add(S);
    return ranges;
}

std::pair<int, int> ETNFit::core_block(int i, bool imaginary) const {
    const int l_count = (int) potential.get_species_ranks().size();
    return blocks()[2 * l_count + (imaginary ? potential.get_core_count() : 0) + i];
}

/* ---------------------------------------------------------------------- */
void ETNFit::solve_linear(bool subset) {
    const std::vector<int> columns = linear_columns();
    const int m = (int) columns.size();
    const int width = (int) coeffs.size();
    std::vector<double> site_energy_grad, force_grad, virial_grad(6 * (size_t) width);
    potential.set_coeffs(coeffs.data());

    std::vector<double> local(m * m + m, 0.0), total(m * m + m);    // [J^T W J | J^T W y]
    // the columns are kept for loss_grad, at the coefficients they were built at
    linear_rows.assign(structures.size(), {});
    linear_rows_at = coeffs;
    for (int k : columns) linear_rows_at[k] = 0.0;
    const double subset_scale = subset ? 1.0 / options.varpro_fraction : 1.0;
    for (size_t index = 0; index < structures.size(); index++) {
        const FitStructure& s = structures[index];
        if (subset && !varpro_subset[index]) continue;
        const NeighList list = s.list();
        const int n3 = list.n_atoms * 3;
        // rows: energy, virial (6), forces (n3), each column's outputs at a unit coefficient, from one eval_grad
        std::vector<double> rows((size_t) m * (7 + n3), 0.0);
        site_energy_grad.resize((size_t) list.inum * width);
        force_grad.resize((size_t) n3 * width);
        potential.eval_grad(list, site_energy_grad.data(), force_grad.data(), virial_grad.data(), false);
        for (int k = 0; k < m; k++) {
            double* row = rows.data() + (size_t) k * (7 + n3);
            for (int ii = 0; ii < list.inum; ii++) row[0] += site_energy_grad[(size_t) ii * width + columns[k]];
            for (int v = 0; v < 6; v++) row[1 + v] = virial_grad[(size_t) v * width + columns[k]];
            for (int q = 0; q < n3; q++) row[7 + q] = force_grad[(size_t) q * width + columns[k]];
        }

        auto add = [&](int r, double wgt, double reference) {
            for (int a = 0; a < m; a++) {
                const double ja = rows[(size_t) a * (7 + n3) + r];
                if (ja == 0.0) continue;
                for (int b = 0; b < m; b++) local[a * m + b] += wgt * ja * rows[(size_t) b * (7 + n3) + r];
                local[m * m + a] += wgt * ja * reference;
            }
        };
        if (s.has_energy) add(0, subset_scale * energy_weight(s), s.energy);
        if (s.has_virial)
            for (int k = 0; k < 6; k++) add(1 + k, subset_scale * virial_weight(s, k), s.virial[k]);
        if (s.has_forces)
            for (int i = 0; i < n3; i++) add(7 + i, subset_scale * force_weight(s, i), s.forces[i]);
        linear_rows[index] = std::move(rows);
    }
    comm.allreduce(local.data(), total.data(), m * m + m, ReduceOp::sum);

    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> matrix(total.data(), m, m);
    Eigen::Map<Eigen::VectorXd> vector(total.data() + m * m, m);
    if (ridge.empty())
        for (int a = 0; a < m; a++) ridge.push_back(options.reg_param * std::max(1.0, matrix(a, a)));
    for (int a = 0; a < m; a++) matrix(a, a) += ridge[a];
    linear_system.assign(total.begin(), total.begin() + (size_t) m * m);
    const Eigen::VectorXd solution = matrix.ldlt().solve(vector);
    for (int a = 0; a < m; a++) coeffs[columns[a]] = solution[a];
}
