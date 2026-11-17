/* -*- c++ -*- ----------------------------------------------------------
   Fits an MTP to reference energies, forces and stresses.
   Algorithm of mlip-3's `mlp train`; see mtp_fit.h.
   Copyright (c) 2023, Alexander Shapeev (Skoltech). BSD 2-Clause, see LICENSE.mlip-3.
------------------------------------------------------------------------- */

#include "mtp_fit.h"

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <deque>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>

// Whether mlp's BFGS refits the linear coefficients before this step.
static bool linear_fit_step(int step) {
    return step == 25 || step == 70 || step == 100 || step == 150 || step == 250 || step == 400;
}

MTPFit::MTPFit(MTP& potential, std::vector<FitStructure> structures, const MTPFitOptions& options, Comm& comm, std::ostream* log, std::function<void()> checkpoint)
    : Fit(potential, std::move(structures), options, comm, log, std::move(checkpoint)) {
    n_radial = potential.get_radial_coeff_count();
    n_linear = potential.get_species_count() + potential.get_alpha_scalar_count();
    reg_vector.assign(n_linear, options.reg_param);
    // radial_only and bfgs_only step mlp's BFGS on a trained potential, without varpro or riemannian
    if (options.radial_only || options.bfgs_only) this->options.varpro = this->options.riemannian = false;
    if (this->options.varpro) this->options.reg_once = true;    // a regularization that moved during BFGS would change the objective

    // A basic moment has degree 1, a product the sum of its factors'.
    std::vector<int> moment_degree(potential.get_alpha_moment_count(), 0);
    std::fill_n(moment_degree.begin(), potential.get_alpha_index_basic_count(), 1);
    for (const std::array<int, 4>& times : potential.get_alpha_index_times()) {
        const int degree = moment_degree[times[0]] + moment_degree[times[1]];
        int& out = moment_degree[times[3]];
        if (out != 0 && out != degree) scalable = false;
        out = degree;
    }
    for (int moment : potential.get_alpha_moment_mapping())
        scalar_degree.push_back(moment_degree[moment]);
}

/* ---------------------------------------------------------------------- */
void MTPFit::run() {
    const auto start = std::chrono::steady_clock::now();
    comm.barrier();
    log_header("MTP");

    add_species();
    if (log) *log << "coefficients: " << coeffs.size() << " (" << n_radial << " radial, " << n_linear << " linear)" << std::endl;

    // Every rank checks the same options and sizes, so every rank raises together.
    if ((options.radial_only || options.bfgs_only) && !potential.is_trained())
        throw std::runtime_error("fit_mtp: radial_only and bfgs_only need a trained potential");
    const int trained = options.radial_only || options.varpro ? n_radial : (int) coeffs.size();
    if (options.inverse_hessian_size != 0 && options.inverse_hessian_size != trained)
        throw std::runtime_error("fit_mtp: inverse_hessian must be " + std::to_string(trained) + " x " + std::to_string(trained) + " for the coefficients trained");
    if (options.riemannian && options.inverse_hessian_size != 0)
        throw std::runtime_error("fit_mtp: riemannian takes no inverse_hessian; pass riemannian=false");
    log_errors("Training set errors before training:");    // the potential as given, before update_min_dist moves its radial basis
    update_min_dist();

    if (options.init_random && !potential.is_trained()) {
        if (rank == 0) {
            std::random_device rand_device;
            std::default_random_engine generator(options.random_seed >= 0 ? (unsigned) options.random_seed : rand_device());
            std::uniform_real_distribution<> uniform(-1.0, 1.0);

            if (log) *log << "radial coefficients drawn at random" << std::endl;
            const int C = potential.get_species_count();
            const int K = potential.get_radial_func_count();
            const int R = potential.get_radial_basis_size();
            for (int p = 0; p < C * C; p++)
                for (int k = 0; k < K; k++) {
                    double* c = coeffs.data() + (p * K + k) * R;
                    for (int l = 0; l < R; l++)
                        c[l] = 1e-7 * options.init_noise * uniform(generator);
                    c[std::min(k, R - 1)] = 1e-7 * (1 + uniform(generator));
                }
        }
        comm.bcast(coeffs.data(), (int) coeffs.size(), 0);
    }

    // pre-training, for the initial scaling
    if (!potential.is_trained() && options.iteration_limit > 0 && !options.skip_preinit) {
        if (log) *log << "pre-training, 75 steps at most" << std::endl;
        orthogonalize();
        fit_linear();
        if (log) *log << "linear fit" << std::endl;
        rescale();
        fit_nonlinear(75);
        rescale();
    }

    if (options.iteration_limit > 0) {
        if (log) *log << "main fit" << std::endl;
        if (options.no_orthogonalize) orthogonalizing = false;
        // the main fit's first linear solve rebuilds the regularization, which reg_once then keeps
        if (options.reg_at_main) {
            reg_built = false;
            reg_init = true;
        }
        if (options.riemannian)
            fit_riemannian(options.iteration_limit);
        else
            fit_nonlinear(options.iteration_limit);
        if (!options.radial_only && !options.bfgs_only) {
            orthogonalize();
            fit_linear();
            if (log) *log << "linear fit" << std::endl;
            rescale();
        }
    }

    potential.set_coeffs(coeffs.data());
    log_errors("Training set errors:");
    if (log) *log << "Trained in " << std::round(std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() * 10) / 10 << " s" << std::endl;
    comm.barrier();
}

/* ----------------------------------------------------------------------
   Appends the species of the training set that the potential lacks, all of
   them when it is untrained. Every species of the potential must occur in
   the training set, and a species is numbered by its index.
------------------------------------------------------------------------- */
void MTPFit::add_species() {
    const int old_spec = potential.is_trained() ? potential.get_species_count() : 0;
    const int new_spec = training_species_count();

    if (old_spec < new_spec) {
        log_added_species(old_spec, new_spec);

        const int K = potential.get_radial_func_count();
        const int R = potential.get_radial_basis_size();
        const int KR = K * R;

        // mlip-3's steps on its coefficient vector: the linear coefficients
        // are shifted, then the radial block grows over them.
        if (!potential.is_trained()) coeffs.resize(potential.get_alpha_scalar_count());
        const int old_radial_count = old_spec * old_spec * KR;
        const std::vector<double> old_radial(coeffs.begin(), coeffs.begin() + old_radial_count);

        const int nlin_old = (int) coeffs.size() - old_radial_count;
        coeffs.resize(old_radial_count + nlin_old + new_spec - old_spec);
        double* lin = coeffs.data() + old_radial_count;
        for (int i = nlin_old + new_spec - old_spec - 1; i >= new_spec; i--)
            lin[i] = lin[i - new_spec + old_spec];
        for (int i = 0; i < new_spec - old_spec; i++)
            lin[i + old_spec] = 0;

        coeffs.resize(new_spec * new_spec * KR + nlin_old + new_spec - old_spec);
        int pair = 0, old_pair = 0;
        for (int p1 = 0; p1 < new_spec; p1++)
            for (int p2 = 0; p2 < new_spec; p2++) {
                double* c = coeffs.data() + pair * KR;
                if (p1 < old_spec && p2 < old_spec) {
                    std::copy(old_radial.begin() + old_pair * KR, old_radial.begin() + (old_pair + 1) * KR, c);
                    old_pair++;
                } else
                    for (int k = 0; k < K; k++) {
                        for (int l = 0; l < R; l++)
                            c[k * R + l] = 1e-6;
                        c[k * R + std::min(k, R)] = 1e-3 * (pair + 1);
                    }
                pair++;
            }

        potential.set_species_count(new_spec);
        potential.set_coeffs(coeffs.data());
        n_radial = potential.get_radial_coeff_count();
        n_linear = new_spec + potential.get_alpha_scalar_count();
    }

    reg_vector.resize(n_linear);
}

void MTPFit::update_min_dist() {
    double min_dist = 999;
    for (const FitStructure& s : structures)
        for (size_t p = 0; p < s.firstneigh.size(); p++) {
            const double* r = s.displacements.data() + p * 3;
            min_dist = std::min(min_dist, std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]));
        }

    double total_min_dist = min_dist;
    comm.allreduce(&min_dist, &total_min_dist, 1, ReduceOp::min);

    if (!options.no_mindist_update) {
        if (log) *log << "min_dist set to " << total_min_dist << std::endl;
        potential.set_min_cutoff(0.99 * total_min_dist);
    }
}

/* ----------------------------------------------------------------------
   BFGS over every coefficient. Rank 0 steps; the others evaluate their
   share at the coefficients it broadcasts.
------------------------------------------------------------------------- */
void MTPFit::fit_nonlinear(int max_iter) {
    const auto start = std::chrono::steady_clock::now();
    comm.barrier();
    log_step_header(true);

    // BFGS steps the leading n coefficients: all of them, or the radial block
    const int n = options.radial_only || options.varpro ? n_radial : (int) coeffs.size();
    std::vector<double> bfgs_g(n);
    double bfgs_f = 0.0, bfgs_terms[4] = {};

    if (rank == 0) {
        bfgs.set_log(log);
        bfgs.set_x(coeffs.data(), n);
        if (options.inverse_hessian)
            bfgs.set_inverse_hessian(options.inverse_hessian);
        else
            bfgs.reset_hessian();
    }

    int num_step = 0;
    double linf = 9e99;         // the loss 50 steps ago
    double loss_prev = 9e99;
    const int max_line_search = 999;
    int curr_line_search = 0;
    bool converge = false;
    bool linesearch = false;

    while (!converge) {
        if (!linesearch) {
            curr_line_search = 0;
            if (rank == 0) bfgs.set_x(coeffs.data(), n);

            if (!options.radial_only && !options.bfgs_only && !options.varpro && linear_fit_step(num_step)) {
                orthogonalize();
                fit_linear();
                if (log) *log << "linear fit" << std::endl;
                if (rank == 0) bfgs.set_x(coeffs.data(), n);
                // a new regularization invalidates the inverse Hessian
                if (reg_init) {
                    if (rank == 0) bfgs.reset_hessian();
                    reg_init = false;
                }
            }

            save_checkpoint();
        }

        if (rank == 0) std::copy(bfgs.x().begin(), bfgs.x().end(), coeffs.begin());
        comm.bcast(coeffs.data(), (int) coeffs.size(), 0);
        bfgs_f = evaluate(bfgs_g, n, true, bfgs_terms);

        if (rank == 0) {
            if (!bfgs.iterate(bfgs_f, bfgs_g)) {
                converge = true;
                if (log) *log << "stop: BFGS cannot lower the loss" << std::endl;
            }
            while (std::abs(bfgs.x(0) - coeffs[0]) > 0.5)    // prevents too large steps
                bfgs.reduce_step(0.25);

            linesearch = bfgs.in_linesearch();
            if (linesearch)
                curr_line_search++;
            else {
                log_step(num_step, bfgs_f, bfgs_terms, true, bfgs.alpha(), curr_line_search + 1);
                if (std::abs(loss_prev - bfgs_f) < 1e-16 && !converge) {
                    converge = true;
                    if (log) *log << "stop: the last step left the loss unchanged" << std::endl;
                }

                loss_prev = bfgs_f;
                num_step++;

                if (num_step % 50 == 0 && num_step > 100) {
                    if ((linf - bfgs_f) / bfgs_f < options.tolerance && !converge) {
                        converge = true;
                        if (log) *log << "stop: the last 50 steps lowered the loss by less than tolerance" << std::endl;
                    }
                    linf = bfgs_f;
                }

                if (num_step >= max_iter && !converge) {
                    converge = true;
                    if (log) *log << "stop: iteration limit reached" << std::endl;
                }
            }

            if (curr_line_search > max_line_search && !converge) {
                if (log) *log << "stop: the line search took over " << max_line_search << " evaluations" << std::endl;
                converge = true;
            }
        }

        comm.bcast(&converge, 1, 0);
        comm.bcast(&linesearch, 1, 0);
        comm.bcast(&num_step, 1, 0);
    }

    if (rank == 0 && options.inverse_hessian)
        std::copy(bfgs.inverse_hessian().begin(), bfgs.inverse_hessian().end(), options.inverse_hessian);

    // the potential counts as trained from its first fit on
    potential.set_trained(true);
    // the next linear fit rebuilds the regularization
    reg_init = true;

    log_phase_time(num_step, start);
}

/* ----------------------------------------------------------------------
   Riemannian L-BFGS (Huang, Gallivan, Absil 2015, in the form of Boumal's
   book) over the radial functions on the Stiefel manifold W W^T = I, with
   the embedded metric: the gradient is the tangent projection of the
   Euclidean one, the retraction is QR, vector transport is projection, and
   the step is a backtracking Armijo search along the retraction curve.
   Without varpro the linear block is a Euclidean factor, refitted at mlp's
   steps. Every rank takes the same steps from the broadcast loss and gradient.
------------------------------------------------------------------------- */
void MTPFit::fit_riemannian(int max_iter) {
    const auto start = std::chrono::steady_clock::now();
    comm.barrier();
    log_step_header(true);

    // the coordinates stepped: the radial block, and the linear one without varpro
    const int n = options.varpro ? n_radial : (int) coeffs.size();
    std::vector<double> g(n), g_new(n), p(n), x0(n);
    std::deque<std::vector<double>> s_memory, y_memory;    // tangent at the current point
    double terms[4] = {};
    auto dot = [n](const std::vector<double>& a, const std::vector<double>& b) {
        double sum = 0.0;
        for (int i = 0; i < n; i++) sum += a[i] * b[i];
        return sum;
    };

    // the loss at coeffs, and its Riemannian gradient into out, on every rank
    auto evaluate_tangent = [&](std::vector<double>& out) {
        comm.bcast(coeffs.data(), (int) coeffs.size(), 0);
        double loss = evaluate(out, n, false, terms);
        comm.bcast(&loss, 1, 0);
        comm.bcast(out.data(), n, 0);
        project_tangent(coeffs.data(), out.data());
        return loss;
    };

    // the start on the manifold, a lower-triangular change of the radial functions, with the linear coefficients refitted
    retract(coeffs);
    if (!options.varpro) {
        fit_linear();
        reg_init = false;
    }
    double f = evaluate_tangent(g);

    int num_step = 0;
    double linf = 9e99;    // the loss 50 steps ago
    double t = 0.0;        // the step length of the last step, and its evaluations
    int halvings = 0;
    while (true) {
        if (!options.varpro && linear_fit_step(num_step)) {
            fit_linear();
            if (log) *log << "linear fit" << std::endl;
            f = evaluate_tangent(g);
            // a new regularization invalidates the memory
            if (reg_init) {
                s_memory.clear();
                y_memory.clear();
                reg_init = false;
            }
        }

        log_step(num_step, f, terms, true, t, halvings + 1);
        save_checkpoint();

        if (num_step >= max_iter) {
            if (log) *log << "stop: iteration limit reached" << std::endl;
            break;
        }
        if (num_step % 50 == 0 && num_step > 100) {
            if ((linf - f) / f < options.tolerance) {
                if (log) *log << "stop: the last 50 steps lowered the loss by less than tolerance" << std::endl;
                break;
            }
            linf = f;
        }
        const double g_norm = std::sqrt(dot(g, g));
        if (g_norm == 0.0) {
            if (log) *log << "stop: the gradient is zero" << std::endl;
            break;
        }

        // the two-loop recursion, scaled by the newest pair, or a unit step down the gradient
        std::vector<double> alpha(s_memory.size());
        p = g;
        for (int i = (int) s_memory.size() - 1; i >= 0; i--) {
            alpha[i] = dot(s_memory[i], p) / dot(y_memory[i], s_memory[i]);
            for (int j = 0; j < n; j++) p[j] -= alpha[i] * y_memory[i][j];
        }
        const double gamma = s_memory.empty() ? 1.0 / g_norm : dot(s_memory.back(), y_memory.back()) / dot(y_memory.back(), y_memory.back());
        for (double& x : p) x *= -gamma;
        for (size_t i = 0; i < s_memory.size(); i++) {
            const double beta = dot(y_memory[i], p) / dot(y_memory[i], s_memory[i]);
            for (int j = 0; j < n; j++) p[j] -= (alpha[i] + beta) * s_memory[i][j];
        }
        project_tangent(coeffs.data(), p.data());
        double slope = dot(g, p);
        if (!(slope < 0.0)) {
            s_memory.clear();
            y_memory.clear();
            for (int j = 0; j < n; j++) p[j] = -g[j] / g_norm;
            slope = -g_norm;
        }

        // backtracking Armijo along the retraction curve
        std::copy_n(coeffs.begin(), n, x0.begin());
        double f_new = f;
        t = 1.0;
        halvings = 0;
        for (; halvings <= 40; halvings++, t *= 0.5) {
            for (int j = 0; j < n; j++) coeffs[j] = x0[j] + t * p[j];
            retract(coeffs);
            f_new = evaluate_tangent(g_new);
            if (f_new <= f + 1e-4 * t * slope) break;
        }
        if (halvings > 40) {
            std::copy(x0.begin(), x0.end(), coeffs.begin());
            f = evaluate_tangent(g);
            if (log) *log << "stop: the line search found no decrease in 41 halvings" << std::endl;
            break;
        }

        // the step and the change of the gradient, with the old vectors transported by projection onto the new tangent space
        std::vector<double> s(n), y = g;
        for (int j = 0; j < n; j++) s[j] = t * p[j];
        project_tangent(coeffs.data(), s.data());
        project_tangent(coeffs.data(), y.data());
        for (int j = 0; j < n; j++) y[j] = g_new[j] - y[j];
        for (size_t i = 0; i < s_memory.size(); i++) {
            project_tangent(coeffs.data(), s_memory[i].data());
            project_tangent(coeffs.data(), y_memory[i].data());
        }
        // the cautious update of Huang, Absil and Gallivan keeps the inverse Hessian positive definite
        if (dot(s, y) >= 1e-4 * g_norm * dot(s, s)) {
            s_memory.push_back(s);
            y_memory.push_back(y);
            if ((int) s_memory.size() > options.lbfgs_memory) {
                s_memory.pop_front();
                y_memory.pop_front();
            }
        }
        f = f_new;
        g.swap(g_new);
        num_step++;
    }

    potential.set_coeffs(coeffs.data());
    potential.set_trained(true);
    reg_init = true;

    log_phase_time(num_step, start);
}

/* ----------------------------------------------------------------------
   Least squares for the species and linear coefficients. Rank 0 solves the summed system.
------------------------------------------------------------------------- */
void MTPFit::fit_linear() {
    potential.set_coeffs(coeffs.data());
    assemble_linear();
    if (rank == 0) solve_linear();
    broadcast_linear();
}

// Sums the upper triangle of the least-squares system over every rank's structures, into
// lin_matrix and lin_vector on rank 0.
void MTPFit::assemble_linear() {
    const int n = n_linear;
    const int width = (int) coeffs.size();
    lin_matrix.assign((size_t) n * n, 0.0);
    lin_vector.assign(n, 0.0);

    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> matrix(lin_matrix.data(), n, n);
    Eigen::Map<Eigen::VectorXd> vector(lin_vector.data(), n);

    // E, F and virial are linear in these coefficients, with the gradients as components: a structure's equations are
    // the components of each value it has, and its references the values, each scaled by the square root of its weight
    int most_atoms = 0;
    for (const FitStructure& s : structures) most_atoms = std::max(most_atoms, (int) s.ilist.size());
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> equations(1 + 3 * most_atoms + 9, n);
    Eigen::VectorXd references(equations.rows());
    std::vector<double> scratch_rows;
    if (options.varpro) linear_rows.resize(structures.size());
    for (size_t index = 0; index < structures.size(); index++) {
        const FitStructure& s = structures[index];
        const NeighList list = s.list();
        site_energy_grad.resize((size_t) list.inum * width);
        force_grad.resize((size_t) list.n_atoms * 3 * width);
        virial_grad.resize(6 * (size_t) width);
        potential.eval_grad(list, site_energy_grad.data(), force_grad.data(), virial_grad.data(), false);

        // the energy, force and virial rows, kept under varpro
        std::vector<double>& row_store = options.varpro ? linear_rows[index] : scratch_rows;
        row_store.resize((size_t) (1 + 3 * list.n_atoms + 6) * n);
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> rows(row_store.data(), 1 + 3 * list.n_atoms + 6, n);
        auto component = [&](const std::vector<double>& grad, size_t row) { return Eigen::Map<const Eigen::RowVectorXd>(&grad[row * width + n_radial], n); };
        rows.row(0).setZero();
        for (int ii = 0; ii < list.inum; ii++) rows.row(0) += component(site_energy_grad, ii);
        for (int q = 0; q < 3 * list.n_atoms; q++) rows.row(1 + q) = component(force_grad, q);
        for (int k = 0; k < 6; k++) rows.row(1 + 3 * list.n_atoms + k) = component(virial_grad, k);

        int m = 0;
        if (s.has_energy) {
            const double root = std::sqrt(energy_weight(s));
            equations.row(m) = root * rows.row(0);
            references[m++] = root * s.energy;
        }
        if (options.force_weight != 0 && s.has_forces)
            for (int ii = 0; ii < list.inum; ii++) {
                const int atom = list.ilist[ii];
                const double root = std::sqrt(force_weight(s, atom, s.forces.data()));
                for (int a = 0; a < 3; a++) {
                    equations.row(m) = root * rows.row(1 + atom * 3 + a);
                    references[m++] = root * s.forces[atom * 3 + a];
                }
            }
        if (options.stress_weight != 0 && s.has_virial) {
            const double root = std::sqrt(stress_weight(s));
            for (int a = 0; a < 3; a++)
                for (int b = 0; b < 3; b++) {
                    equations.row(m) = root * rows.row(1 + 3 * list.n_atoms + VIRIAL_INDEX[a][b]);
                    references[m++] = root * s.virial[VIRIAL_INDEX[a][b]];
                }
        }

        matrix.selfadjointView<Eigen::Upper>().rankUpdate(equations.topRows(m).transpose());
        vector.noalias() += equations.topRows(m).transpose() * references.head(m);
    }
    if (options.varpro) {
        linear_rows_at.assign(coeffs.begin(), coeffs.begin() + n_radial);
        linear_rows_at.push_back(potential.get_scaling());
    }

    std::vector<double> matrix_sum(rank == 0 ? (size_t) n * n : 0), vector_sum(rank == 0 ? n : 0);
    comm.reduce(lin_matrix.data(), matrix_sum.data(), n * n, ReduceOp::sum, 0);
    comm.reduce(lin_vector.data(), vector_sum.data(), n, ReduceOp::sum, 0);
    if (rank == 0) {
        lin_matrix.swap(matrix_sum);
        lin_vector.swap(vector_sum);
    }
}

// Hands rank 0's linear coefficients and regularization to every rank.
void MTPFit::broadcast_linear() {
    comm.bcast(coeffs.data(), (int) coeffs.size(), 0);
    comm.bcast(reg_vector.data(), n_linear, 0);
    comm.bcast(&reg_init, 1, 0);
    potential.set_coeffs(coeffs.data());
}

// Regularizes the upper triangle held in lin_matrix and solves it into the
// linear coefficients, by Gaussian elimination without row swaps.
void MTPFit::solve_linear() {
    const int n = n_linear;
    double* A = lin_matrix.data();
    double* b = lin_vector.data();
    double* x = coeffs.data() + n_radial;

    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            A[j * n + i] = A[i * n + j];

    // the regularization is rebuilt when the diagonal has moved two orders of magnitude from it
    if (options.reg_once && reg_built) reg_init = false;
    if (!reg_init && !options.reg_once)
        for (int i = 0; i < n; i++)
            if (reg_vector[i] < 1e-2 * options.reg_param * std::max(1.0, A[i * n + i]) / structure_count || reg_vector[i] > 1e2 * options.reg_param * std::max(1.0, A[i * n + i]) / structure_count) {
                reg_init = true;
                if (log) *log << "linear regularization rebuilt, inverse Hessian reset" << std::endl;
                break;
            }

    if (reg_init) {
        for (int i = 0; i < n; i++)
            reg_vector[i] = options.reg_param * std::max(1.0, A[i * n + i]) / structure_count;
        reg_built = true;
    }

    for (int i = 0; i < n; i++)
        A[i * n + i] += reg_vector[i] * structure_count;

    for (int i = 0; i < n - 1; i++) {
        const double m_ii = A[i * n + i];
        for (int j = i + 1; j < n; j++) {
            const double ratio = A[j * n + i] / m_ii;
            for (int k = i; k < n; k++)
                A[j * n + k] -= ratio * A[i * n + k];
            b[j] -= ratio * b[i];
        }
    }

    x[n - 1] = b[n - 1] / A[(n - 1) * n + (n - 1)];
    for (int i = n - 2; i >= 0; i--) {
        double temp = b[i];
        for (int j = i + 1; j < n; j++)
            temp -= A[i * n + j] * x[j];
        x[i] = temp / A[i * n + i];
    }
}

/* ----------------------------------------------------------------------
   Picks the scaling, among 5 around the current one, with the best-conditioned
   linear fit. A basis function of degree d in the radial functions scales as
   scaling^d, so each fit rescales the system last assembled, which is
   reassembled only when orthogonalizing moves the radial functions.
------------------------------------------------------------------------- */
void MTPFit::rescale() {
    const double start_scaling = potential.get_scaling();
    double chosen_condition_number = 0.0;
    double min_scaling = potential.get_scaling();
    double max_scaling = potential.get_scaling();
    std::vector<double> abs_linear(n_linear);
    int ind;

    double assembled_scaling = 0.0;    // none assembled yet
    std::vector<double> assembled_matrix, assembled_vector, radial(n_radial), factors(n_linear);

    auto fit_at = [&](double scaling) {
        potential.set_scaling(scaling);
        if (!scalable) {
            orthogonalize();
            fit_linear();
            return;
        }

        std::copy(coeffs.begin(), coeffs.begin() + n_radial, radial.begin());
        orthogonalize();
        double moved = 0.0, largest = 0.0;
        for (int i = 0; i < n_radial; i++) {
            moved = std::max(moved, std::abs(coeffs[i] - radial[i]));
            largest = std::max(largest, std::abs(radial[i]));
        }
        if (assembled_scaling == 0.0 || moved > 1e-12 * largest) {
            potential.set_coeffs(coeffs.data());
            assemble_linear();
            assembled_scaling = scaling;
            assembled_matrix = lin_matrix;
            assembled_vector = lin_vector;
        }

        if (rank == 0) {
            const int n = n_linear;
            const int species = n_linear - (int) scalar_degree.size();
            for (int i = 0; i < n; i++)
                factors[i] = i < species ? 1.0 : std::pow(scaling / assembled_scaling, scalar_degree[i - species]);
            for (int i = 0; i < n; i++) {
                lin_vector[i] = assembled_vector[i] * factors[i];
                for (int j = i; j < n; j++)
                    lin_matrix[i * n + j] = assembled_matrix[i * n + j] * factors[i] * factors[j];
            }
            solve_linear();
        }
        broadcast_linear();
    };

    do {
        double condition_number[5];
        const double scaling = potential.get_scaling();
        const double scalings[5] = {scaling / 1.2, scaling / 1.1, scaling, scaling * 1.1, scaling * 1.2};
        for (int j = 0; j < 5; j++) {
            fit_at(scalings[j]);

            double rms = 0;
            for (int i = 0; i < n_linear; i++) {
                abs_linear[i] = std::abs(coeffs[n_radial + i]);
                rms += abs_linear[i] * abs_linear[i];
            }
            rms = std::sqrt(rms);
            std::sort(abs_linear.begin(), abs_linear.end());
            condition_number[j] = rms / abs_linear[abs_linear.size() / 2];
        }

        ind = 2;
        for (int j = 0; j < 5; j++)
            if (condition_number[j] < condition_number[ind]) ind = j;

        fit_at(scalings[ind]);
        chosen_condition_number = condition_number[ind];

        // stop once the choice falls strictly inside the range already visited
        if (min_scaling < scalings[ind] && scalings[ind] < max_scaling)
            ind = 2;
        else {
            min_scaling = std::min(min_scaling, scalings[ind]);
            max_scaling = std::max(max_scaling, scalings[ind]);
        }
    } while (ind != 2);

    if (log) *log << "rescaled: scaling " << start_scaling << " to " << potential.get_scaling() << ", condition number " << chosen_condition_number << std::endl;
}

double MTPFit::objective(const std::vector<double>& at, std::vector<double>& grad) {
    // fixes the regularization at the potential's own radial coefficients, where BFGS's first evaluation fixes it
    if (options.varpro) fit_linear();
    std::copy(at.begin(), at.end(), coeffs.begin());
    double terms[4];
    grad.assign(coeffs.size(), 0.0);
    double total = evaluate(grad, (int) grad.size(), !options.riemannian, terms);
    comm.bcast(&total, 1, 0);
    comm.bcast(grad.data(), (int) grad.size(), 0);
    // the linear coefficients are a function of the radial ones: their partial gradient is zero at the optimum
    if (options.varpro) std::fill(grad.begin() + n_radial, grad.end(), 0.0);
    // under riemannian, the Riemannian gradient at the radial functions in at, which must be orthonormal
    if (options.riemannian) project_tangent(coeffs.data(), grad.data());
    return total;
}

double MTPFit::evaluate(std::vector<double>& grad, int n, bool radial, double terms[4]) {
    if (options.varpro) fit_linear();    // the linear coefficients at their optimum for these radial ones
    local_grad.resize(coeffs.size());
    double loss = loss_grad(local_grad) / structure_count;
    for (double& g : local_grad) g /= structure_count;
    double local_terms[4] = {loss_terms[0] / structure_count, loss_terms[1] / structure_count, loss_terms[2] / structure_count, loss};
    if (rank == 0) add_penalty(loss, local_grad, radial);
    local_terms[3] = loss - local_terms[3];

    double total = 0.0;
    comm.reduce(&loss, &total, 1, ReduceOp::sum, 0);
    comm.reduce(local_terms, terms, 4, ReduceOp::sum, 0);
    comm.reduce(local_grad.data(), grad.data(), n, ReduceOp::sum, 0);
    return total;
}

/* ---------------------------------------------------------------------- */
double MTPFit::loss_grad(std::vector<double>& grad) {
    potential.set_coeffs(coeffs.data());
    std::fill(grad.begin(), grad.end(), 0.0);
    double loss = 0.0;
    std::fill_n(loss_terms, 3, 0.0);
    const bool from_rows = linear_rows.size() == structures.size() && linear_rows_at.size() == (size_t) n_radial + 1 && std::equal(coeffs.begin(), coeffs.begin() + n_radial, linear_rows_at.begin()) && linear_rows_at[n_radial] == potential.get_scaling();

    for (size_t index = 0; index < structures.size(); index++) {
        const FitStructure& s = structures[index];
        const NeighList list = s.list();
        double energy;
        const double *forces, *virial;
        if (from_rows) {
            const int m = 1 + 3 * list.n_atoms + 6;
            linear_values.resize(m);
            Eigen::Map<Eigen::VectorXd>(linear_values.data(), m).noalias() = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(linear_rows[index].data(), m, n_linear) * Eigen::Map<const Eigen::VectorXd>(coeffs.data() + n_radial, n_linear);
            energy = linear_values[0];
            forces = linear_values.data() + 1;
            virial = linear_values.data() + 1 + 3 * list.n_atoms;
        } else {
            potential.compute(list);
            energy = potential.get_energy();
            forces = potential.get_forces().data();
            virial = potential.get_virial().data();
        }

        dloss_dforces.assign((size_t) list.n_atoms * 3, 0.0);
        double dloss_dvirial[6] = {};
        double dloss_denergy = 0.0;

        // the energy is added last, for less round-off
        if (options.force_weight != 0 && s.has_forces)
            for (int ii = 0; ii < list.inum; ii++) {
                const int i = list.ilist[ii];
                const double wgt = force_weight(s, i, forces);    // mlp's loss weighs by the predicted force
                for (int a = 0; a < 3; a++) {
                    const double diff = forces[i * 3 + a] - s.forces[i * 3 + a];
                    loss += wgt * diff * diff;
                    loss_terms[1] += wgt * diff * diff;
                    dloss_dforces[i * 3 + a] = 2.0 * wgt * diff;
                }
            }

        if (options.stress_weight != 0 && s.has_virial) {
            const double wgt = stress_weight(s);
            for (int a = 0; a < 3; a++)
                for (int b = 0; b < 3; b++) {
                    const int k = VIRIAL_INDEX[a][b];
                    const double diff = virial[k] - s.virial[k];
                    loss += wgt * diff * diff;
                    loss_terms[2] += wgt * diff * diff;
                    dloss_dvirial[k] += 2.0 * wgt * diff;
                }
        }

        if (options.energy_weight != 0 && s.has_energy) {
            const double wgt = energy_weight(s);
            const double diff = energy - s.energy;
            loss += wgt * diff * diff;
            loss_terms[0] += wgt * diff * diff;
            dloss_denergy = 2.0 * wgt * diff;
        }

        potential.eval_loss_grad(list, dloss_denergy, dloss_dforces.data(), dloss_dvirial, grad.data());
    }

    return loss;
}

void MTPFit::add_penalty(double& loss, std::vector<double>& grad, bool radial) {
    const int K = potential.get_radial_func_count();
    const double coeff = options.penalty_weight;

    // the norm of each radial function over all species pairs, towards 1
    for (int k = 0; radial && k < K; k++) {
        const double norm = row_dot(coeffs.data(), k, coeffs.data(), k);
        loss += coeff * (norm - 1) * (norm - 1);
        row_axpy(grad.data(), k, coeff * 4 * (norm - 1), coeffs.data(), k);
    }

    // the overlap of each two radial functions, towards 0
    for (int k1 = 0; radial && k1 < K; k1++)
        for (int k2 = k1 + 1; k2 < K; k2++) {
            const double scal = row_dot(coeffs.data(), k1, coeffs.data(), k2);
            loss += coeff * scal * scal;
            row_axpy(grad.data(), k1, coeff * 2 * scal, coeffs.data(), k2);
            row_axpy(grad.data(), k2, coeff * 2 * scal, coeffs.data(), k1);
        }

    // the regularization of the linear fit
    for (int i = 0; i < n_linear; i++)
        loss += coeffs[n_radial + i] * coeffs[n_radial + i] * reg_vector[i];
    for (int i = 0; i < n_linear; i++)
        grad[n_radial + i] += 2 * coeffs[n_radial + i] * reg_vector[i];
}

// Gram-Schmidt over the radial functions, each a vector over species pairs
// and basis functions. The projection divides by the squared norm of the
// function being reduced, not of the one projected out.
void MTPFit::orthogonalize() {
    if (!orthogonalizing) return;

    const int C = potential.get_species_count();
    const int K = potential.get_radial_func_count();
    const int R = potential.get_radial_basis_size();

    for (int k = 0; k < K; k++) {
        for (int k2 = 0; k2 < k; k2++) {
            const double norm = row_dot(coeffs.data(), k, coeffs.data(), k);
            const double scal = row_dot(coeffs.data(), k, coeffs.data(), k2);
            for (int p = 0; p < C * C; p++)
                for (int l = 0; l < R; l++)
                    coeffs[p * K * R + k * R + l] -= coeffs[p * K * R + k2 * R + l] * scal / norm;
        }

        double norm = 0;
        for (int p = 0; p < C * C; p++)
            for (int l = 0; l < R; l++) {
                const int idx = p * K * R + k * R + l;
                norm += coeffs[idx] * coeffs[idx] + 1e-10;
            }
        norm = std::sqrt(norm);
        for (int p = 0; p < C * C; p++)
            for (int l = 0; l < R; l++)
                coeffs[p * K * R + k * R + l] /= norm;
    }
}

double MTPFit::row_dot(const double* a, int k1, const double* b, int k2) const {
    const int P = potential.get_species_count() * potential.get_species_count();
    const int K = potential.get_radial_func_count();
    const int R = potential.get_radial_basis_size();
    double sum = 0.0;
    for (int p = 0; p < P; p++)
        for (int l = 0; l < R; l++) sum += a[(p * K + k1) * R + l] * b[(p * K + k2) * R + l];
    return sum;
}

void MTPFit::row_axpy(double* y, int k1, double factor, const double* x, int k2) const {
    const int P = potential.get_species_count() * potential.get_species_count();
    const int K = potential.get_radial_func_count();
    const int R = potential.get_radial_basis_size();
    for (int p = 0; p < P; p++)
        for (int l = 0; l < R; l++) y[(p * K + k1) * R + l] += factor * x[(p * K + k2) * R + l];
}

std::vector<double> MTPFit::radial_overlaps(const double* a, const double* b) const {
    const int K = potential.get_radial_func_count();
    std::vector<double> overlaps(K * K);
    for (int k1 = 0; k1 < K; k1++)
        for (int k2 = 0; k2 < K; k2++) overlaps[k1 * K + k2] = row_dot(a, k1, b, k2);
    return overlaps;
}

void MTPFit::project_tangent(const double* w, double* v) const {
    const int K = potential.get_radial_func_count();
    const std::vector<double> overlaps = radial_overlaps(v, w);
    for (int k1 = 0; k1 < K; k1++)
        for (int k2 = 0; k2 < K; k2++) row_axpy(v, k1, -0.5 * (overlaps[k1 * K + k2] + overlaps[k2 * K + k1]), w, k2);
}

void MTPFit::retract(std::vector<double>& c) const {
    const int P = potential.get_species_count() * potential.get_species_count();
    const int K = potential.get_radial_func_count();
    const int R = potential.get_radial_basis_size();

    // modified Gram-Schmidt, twice for orthonormality to rounding
    for (int pass = 0; pass < 2; pass++)
        for (int k = 0; k < K; k++) {
            for (int k2 = 0; k2 < k; k2++) row_axpy(c.data(), k, -row_dot(c.data(), k, c.data(), k2), c.data(), k2);
            const double norm = std::sqrt(row_dot(c.data(), k, c.data(), k));
            for (int p = 0; p < P; p++)
                for (int l = 0; l < R; l++) c[(p * K + k) * R + l] /= norm;
        }
}

/* ---------------------------------------------------------------------- */
double MTPFit::energy_weight(const FitStructure& s) const {
    return options.energy_weight / std::pow(s.ilist.size(), options.weight_scaling) * options.select_factor;
}

double MTPFit::force_weight(const FitStructure& s, int i, const double* forces) const {
    double wgt = options.force_weight / std::pow(s.ilist.size(), options.weight_scaling_forces);
    if (options.scale_by_force > 0.0) {
        const double* f = forces + i * 3;
        wgt *= options.scale_by_force / (f[0] * f[0] + f[1] * f[1] + f[2] * f[2] + options.scale_by_force);
    }
    if (options.balance_species)
        wgt *= s.species_factors[s.types[i]];
    return wgt * options.select_factor;
}

double MTPFit::stress_weight(const FitStructure& s) const {
    return options.stress_weight / s.ilist.size() * options.select_factor;
}
