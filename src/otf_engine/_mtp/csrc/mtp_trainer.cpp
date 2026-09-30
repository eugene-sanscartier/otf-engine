/* -*- c++ -*- ----------------------------------------------------------
   Fits an MTP to reference energies, forces and stresses.
   Algorithm of mlip-3's `mlp train`; see mtp_trainer.h.
   Copyright (c) 2023, Alexander Shapeev (Skoltech). BSD 2-Clause, see LICENSE.mlip-3.
------------------------------------------------------------------------- */

#include "mtp_trainer.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>

// Virial component of each entry of the symmetric 3x3 stress.
static const int VIRIAL_INDEX[3][3] = {{0, 3, 4}, {3, 1, 5}, {4, 5, 2}};

MTPTrainer::MTPTrainer(MTPTraining& potential, std::vector<TrainingStructure> structures, const TrainerOptions& options, MPI_Comm comm, std::ostream* log, std::function<void()> checkpoint)
    : potential(potential), structures(std::move(structures)), options(options), comm(comm), log(log), checkpoint(std::move(checkpoint)) {
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int local_count = (int) this->structures.size();
    MPI_Allreduce(&local_count, &structure_count, 1, MPI_INT, MPI_SUM, comm);

    n_radial = potential.get_radial_coeff_count();
    n_linear = potential.get_species_count() + potential.get_alpha_scalar_count();
    coeffs.resize(n_radial + n_linear);
    potential.get_coeffs(coeffs.data());
    reg_vector.assign(n_linear, reg_param);

    // A basic moment has degree 1, a product the sum of its factors'.
    std::vector<int> moment_degree(potential.get_alpha_moment_count(), 0);
    std::fill_n(moment_degree.begin(), potential.get_alpha_index_basic_count(), 1);
    const int* times = potential.get_alpha_index_times();
    for (int t = 0; t < potential.get_alpha_index_times_count(); t++) {
        const int degree = moment_degree[times[4 * t]] + moment_degree[times[4 * t + 1]];
        int& out = moment_degree[times[4 * t + 3]];
        if (out != 0 && out != degree) scalable = false;
        out = degree;
    }
    const int* mapping = potential.get_alpha_moment_mapping();
    for (int s = 0; s < potential.get_alpha_scalar_count(); s++)
        scalar_degree.push_back(moment_degree[mapping[s]]);
}

/* ---------------------------------------------------------------------- */
void MTPTrainer::train() {
    MPI_Barrier(comm);
    if (log) *log << structure_count << " configurations found in the training set" << std::endl;

    add_species();
    update_min_dist();

    if (options.init_random && !potential.is_trained()) {
        if (rank == 0) {
            std::random_device rand_device;
            std::default_random_engine generator(rand_device());
            std::uniform_real_distribution<> uniform(-1.0, 1.0);

            if (log) *log << "Random initialization of radial coefficients" << std::endl;
            const int C = potential.get_species_count();
            const int K = potential.get_radial_func_count();
            const int R = potential.get_radial_basis_size();
            for (int p = 0; p < C * C; p++)
                for (int k = 0; k < K; k++) {
                    double* c = coeffs.data() + (p * K + k) * R;
                    for (int l = 0; l < R; l++)
                        c[l] = 5e-7 * uniform(generator);
                    c[std::min(k, R - 1)] = 1e-7 * (1 + uniform(generator));
                }
        }
        MPI_Bcast(coeffs.data(), (int) coeffs.size(), MPI_DOUBLE, 0, comm);
    }

    // pre-training, for the initial scaling
    if (!potential.is_trained() && options.iteration_limit > 0 && !options.skip_preinit) {
        fit_linear();
        rescale();
        if (log) *log << "Pre-training started" << std::endl;
        fit_nonlinear(75);
        rescale();
        if (log) *log << "Pre-training ended" << std::endl;
    }

    if (options.iteration_limit > 0) {
        if (log) {
            *log << "Iteration limit is " << options.iteration_limit << std::endl;
            *log << "Convergence tolerance is " << options.tolerance << std::endl;
            if (options.energy_weight != 0 || options.force_weight != 0 || options.stress_weight != 0) {
                *log << "Energy weight: " << options.energy_weight << std::endl;
                *log << "Force weight: " << options.force_weight << std::endl;
                *log << "Stress weight: " << options.stress_weight << std::endl;
            }
        }

        fit_nonlinear(options.iteration_limit);
        fit_linear();
        rescale();
    }

    potential.set_coeffs(coeffs.data());
    MPI_Barrier(comm);
}

/* ----------------------------------------------------------------------
   Appends the species of the training set that the potential lacks, all of
   them when it is untrained. Every species of the potential must occur in
   the training set, and a species is numbered by its index.
------------------------------------------------------------------------- */
void MTPTrainer::add_species() {
    int local_count = 0;
    for (const TrainingStructure& s : structures)
        for (int t : s.types) {
            if (t < 0)
                throw std::runtime_error("MTPTrainer: a training structure has species " + std::to_string(t));
            local_count = std::max(local_count, t + 1);
        }
    int type_count = 0;
    MPI_Allreduce(&local_count, &type_count, 1, MPI_INT, MPI_MAX, comm);

    std::vector<int> present(type_count, 0), present_anywhere(type_count, 0);
    for (const TrainingStructure& s : structures)
        for (int t : s.types)
            present[t] = 1;
    MPI_Allreduce(present.data(), present_anywhere.data(), type_count, MPI_INT, MPI_MAX, comm);

    const int old_spec = potential.is_trained() ? potential.get_species_count() : 0;
    const int new_spec = type_count;
    for (int t = 0; t < std::max(old_spec, new_spec); t++)
        if (t >= new_spec || !present_anywhere[t])
            throw std::runtime_error(t < old_spec ? "MTPTrainer: species " + std::to_string(t) + " of the potential is not present in the training set" : "MTPTrainer: the training set has species " + std::to_string(new_spec - 1) + " but not species " + std::to_string(t));

    if (old_spec < new_spec) {
        if (log) {
            *log << "Following atomic numbers will be added to the MTP potential: ";
            for (int t = old_spec; t < new_spec; t++)
                *log << t << (t != new_spec - 1 ? ", " : "");
            *log << std::endl;
        }

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

void MTPTrainer::update_min_dist() {
    double min_dist = 999;
    for (const TrainingStructure& s : structures)
        for (size_t p = 0; p < s.firstneigh.size(); p++) {
            const double* r = s.displacements.data() + p * 3;
            min_dist = std::min(min_dist, std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]));
        }

    double total_min_dist = min_dist;
    MPI_Allreduce(&min_dist, &total_min_dist, 1, MPI_DOUBLE, MPI_MIN, comm);

    if (!options.no_mindist_update) {
        if (log) *log << "Minimal interatomic distance in the training set is " << total_min_dist << ". MTP's mindist will be updated" << std::endl;
        potential.set_min_cutoff(0.99 * total_min_dist);
    }
}

/* ----------------------------------------------------------------------
   BFGS over every coefficient. Rank 0 steps; the others evaluate their
   share at the coefficients it broadcasts.
------------------------------------------------------------------------- */
void MTPTrainer::fit_nonlinear(int max_iter) {
    MPI_Barrier(comm);
    if (log) *log << "MTPR training started on " << size << " core(s)" << std::endl;

    const int n = (int) coeffs.size();
    std::vector<double> grad(n), bfgs_g(n);
    double bfgs_f = 0.0;

    if (rank == 0) {
        bfgs.set_x(coeffs.data(), n);
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

            if (num_step == 25 || num_step == 70 || num_step == 100 || num_step == 150 || num_step == 250 || num_step == 400) {
                fit_linear();
                if (rank == 0) bfgs.set_x(coeffs.data(), n);
                // a new regularization invalidates the inverse Hessian
                if (reg_init) {
                    if (rank == 0) bfgs.reset_hessian();
                    reg_init = false;
                }
            }

            if (rank == 0 && checkpoint) {
                potential.set_coeffs(coeffs.data());
                checkpoint();
            }
        }

        if (rank == 0) coeffs = bfgs.x();
        MPI_Bcast(coeffs.data(), n, MPI_DOUBLE, 0, comm);

        double loss = loss_grad(grad) / structure_count;
        for (double& g : grad) g /= structure_count;
        if (rank == 0) add_penalty(loss, grad);

        MPI_Barrier(comm);
        MPI_Reduce(&loss, &bfgs_f, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
        MPI_Reduce(grad.data(), bfgs_g.data(), n, MPI_DOUBLE, MPI_SUM, 0, comm);

        if (rank == 0) {
            if (!bfgs.iterate(bfgs_f, bfgs_g))
                converge = true;
            while (std::abs(bfgs.x(0) - coeffs[0]) > 0.5)    // prevents too large steps
                bfgs.reduce_step(0.25);

            linesearch = bfgs.in_linesearch();
            if (linesearch)
                curr_line_search++;
            else {
                if (loss_prev < bfgs_f && log) *log << "*" << std::endl;

                if (std::abs(loss_prev - bfgs_f) < 1e-16) {
                    converge = true;
                    if (log) *log << "BFGS ended due to small decr. for 1 iteration" << std::endl;
                }

                loss_prev = bfgs_f;
                if (log) *log << "BFGS iter " << num_step << ": f=" << bfgs_f << std::endl;
                num_step++;

                if (num_step % 50 == 0 && num_step > 100) {
                    if ((linf - bfgs_f) / bfgs_f < options.tolerance) {
                        converge = true;
                        if (log) *log << "BFGS ended due to small decr. in 50 iterations" << std::endl;
                    }
                    linf = bfgs_f;
                }

                if (num_step >= max_iter) {
                    converge = true;
                    if (log) *log << "step limit reached" << std::endl;
                }
            }

            if (curr_line_search > max_line_search) {
                if (log) *log << "BFGS has exceeded the linesearch limit. Cannot converge further" << std::endl;
                converge = true;
            }
        }

        MPI_Barrier(comm);
        MPI_Bcast(&converge, 1, MPI_C_BOOL, 0, comm);
        MPI_Bcast(&linesearch, 1, MPI_C_BOOL, 0, comm);
        MPI_Bcast(&num_step, 1, MPI_INT, 0, comm);
    }

    // the potential counts as trained from its first fit on
    potential.set_trained(true);
    // the next linear fit rebuilds the regularization
    reg_init = true;

    if (log) *log << "MTPR training ended" << std::endl;
}

/* ----------------------------------------------------------------------
   Least squares for the species and linear coefficients, after
   orthonormalizing the radial functions. Rank 0 solves the summed system.
------------------------------------------------------------------------- */
void MTPTrainer::fit_linear() {
    orthogonalize();
    potential.set_coeffs(coeffs.data());
    assemble_linear();
    if (rank == 0) solve_linear(structure_count);
    broadcast_linear();
}

// Sums the upper triangle of the least-squares system over every rank's structures, into
// lin_matrix and lin_vector on rank 0.
void MTPTrainer::assemble_linear() {
    const int n = n_linear;
    const int width = (int) coeffs.size();
    lin_matrix.assign((size_t) n * n, 0.0);
    lin_vector.assign(n, 0.0);

    std::vector<double> energy_cmpnts(n), force_weights;
    for (const TrainingStructure& s : structures) {
        const NeighList list = s.list();
        site_energy_grad.resize((size_t) list.inum * width);
        force_grad.resize((size_t) list.n_atoms * 3 * width);
        virial_grad.resize(6 * (size_t) width);
        potential.eval_grad(list, site_energy_grad.data(), force_grad.data(), virial_grad.data(), false);

        // E, F and virial are linear in these coefficients, with the gradients as components
        std::fill(energy_cmpnts.begin(), energy_cmpnts.end(), 0.0);
        for (int ii = 0; ii < list.inum; ii++)
            for (int k = 0; k < n; k++)
                energy_cmpnts[k] += site_energy_grad[(size_t) ii * width + n_radial + k];
        auto force_cmpnt = [&](int atom, int k, int a) { return force_grad[((size_t) atom * 3 + a) * width + n_radial + k]; };
        auto stress_cmpnt = [&](int k, int a, int b) { return virial_grad[(size_t) VIRIAL_INDEX[a][b] * width + n_radial + k]; };

        if (s.has_energy) {
            const double wgt = energy_weight(s);
            for (int i = 0; i < n; i++)
                for (int j = i; j < n; j++)
                    lin_matrix[i * n + j] += wgt * energy_cmpnts[i] * energy_cmpnts[j];
            for (int i = 0; i < n; i++)
                lin_vector[i] += wgt * energy_cmpnts[i] * s.energy;
        }

        if (options.force_weight > 0 && s.has_forces) {
            force_weights.resize(list.inum);
            for (int ii = 0; ii < list.inum; ii++)
                force_weights[ii] = force_weight(s, list.ilist[ii]);

            for (int i = 0; i < n; i++)
                for (int j = i; j < n; j++)
                    for (int ii = 0; ii < list.inum; ii++)
                        for (int a = 0; a < 3; a++)
                            lin_matrix[i * n + j] += force_weights[ii] * force_cmpnt(list.ilist[ii], i, a) * force_cmpnt(list.ilist[ii], j, a);

            for (int ii = 0; ii < list.inum; ii++) {
                const int atom = list.ilist[ii];
                for (int i = 0; i < n; i++)
                    for (int a = 0; a < 3; a++)
                        lin_vector[i] += force_weights[ii] * force_cmpnt(atom, i, a) * s.forces[atom * 3 + a];
            }
        }

        if (options.stress_weight > 0 && s.has_virial) {
            const double wgt = stress_weight(s);
            for (int i = 0; i < n; i++)
                for (int j = i; j < n; j++)
                    for (int a = 0; a < 3; a++)
                        for (int b = 0; b < 3; b++)
                            lin_matrix[i * n + j] += wgt * stress_cmpnt(i, a, b) * stress_cmpnt(j, a, b);

            for (int i = 0; i < n; i++)
                for (int a = 0; a < 3; a++)
                    for (int b = 0; b < 3; b++)
                        lin_vector[i] += wgt * stress_cmpnt(i, a, b) * s.virial[VIRIAL_INDEX[a][b]];
        }
    }

    std::vector<double> matrix_sum(rank == 0 ? (size_t) n * n : 0), vector_sum(rank == 0 ? n : 0);
    MPI_Reduce(lin_matrix.data(), matrix_sum.data(), n * n, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(lin_vector.data(), vector_sum.data(), n, MPI_DOUBLE, MPI_SUM, 0, comm);
    if (rank == 0) {
        lin_matrix.swap(matrix_sum);
        lin_vector.swap(vector_sum);
    }
}

// Hands rank 0's linear coefficients and regularization to every rank.
void MTPTrainer::broadcast_linear() {
    MPI_Bcast(coeffs.data(), (int) coeffs.size(), MPI_DOUBLE, 0, comm);
    MPI_Bcast(reg_vector.data(), n_linear, MPI_DOUBLE, 0, comm);
    MPI_Bcast(&reg_init, 1, MPI_C_BOOL, 0, comm);
    potential.set_coeffs(coeffs.data());
}

// Regularizes the upper triangle held in lin_matrix and solves it into the
// linear coefficients, by Gaussian elimination without row swaps.
void MTPTrainer::solve_linear(int ts_size) {
    const int n = n_linear;
    double* A = lin_matrix.data();
    double* b = lin_vector.data();
    double* x = coeffs.data() + n_radial;

    // mlp logs 8 significant digits from its first linear fit on
    if (log) log->precision(8);

    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            A[j * n + i] = A[i * n + j];

    // the regularization is rebuilt when the diagonal has moved two orders of magnitude from it
    if (!reg_init)
        for (int i = 0; i < n; i++)
            if (reg_vector[i] < 1e-2 * reg_param * std::max(1.0, A[i * n + i]) / ts_size || reg_vector[i] > 1e2 * reg_param * std::max(1.0, A[i * n + i]) / ts_size) {
                reg_init = true;
                if (log) *log << "Regularization parameters updated. Hessian in BFGS is reset" << std::endl;
                break;
            }

    if (reg_init)
        for (int i = 0; i < n; i++)
            reg_vector[i] = reg_param * std::max(1.0, A[i * n + i]) / ts_size;

    for (int i = 0; i < n; i++)
        A[i * n + i] += reg_vector[i] * ts_size;

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
void MTPTrainer::rescale() {
    double min_scaling = potential.get_scaling();
    double max_scaling = potential.get_scaling();
    std::vector<double> abs_linear(n_linear);
    int ind;

    double assembled_scaling = 0.0;    // none assembled yet
    std::vector<double> assembled_matrix, assembled_vector, radial(n_radial), factors(n_linear);

    auto fit_at = [&](double scaling) {
        potential.set_scaling(scaling);
        if (!scalable) {
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
            solve_linear(structure_count);
        }
        broadcast_linear();
    };

    do {
        double condition_number[5];
        const double scaling = potential.get_scaling();
        const double scalings[5] = {scaling / 1.2, scaling / 1.1, scaling, scaling * 1.1, scaling * 1.2};
        if (log) *log << "Rescaling..." << std::endl;

        for (int j = 0; j < 5; j++) {
            if (log) *log << "   scaling = " << scalings[j] << ", condition number = " << std::flush;
            fit_at(scalings[j]);

            double rms = 0;
            for (int i = 0; i < n_linear; i++) {
                abs_linear[i] = std::abs(coeffs[n_radial + i]);
                rms += abs_linear[i] * abs_linear[i];
            }
            rms = std::sqrt(rms);
            std::sort(abs_linear.begin(), abs_linear.end());
            condition_number[j] = rms / abs_linear[abs_linear.size() / 2];
            if (log) *log << condition_number[j] << std::endl;
        }

        ind = 2;
        for (int j = 0; j < 5; j++)
            if (condition_number[j] < condition_number[ind]) ind = j;

        if (log) *log << "Rescaling to " << scalings[ind] << "... " << std::flush;
        fit_at(scalings[ind]);
        if (log) *log << "done" << std::endl;

        // stop once the choice falls strictly inside the range already visited
        if (min_scaling < scalings[ind] && scalings[ind] < max_scaling)
            ind = 2;
        else {
            min_scaling = std::min(min_scaling, scalings[ind]);
            max_scaling = std::max(max_scaling, scalings[ind]);
        }
    } while (ind != 2);
}

/* ---------------------------------------------------------------------- */
double MTPTrainer::loss_grad(std::vector<double>& grad) {
    potential.set_coeffs(coeffs.data());
    std::fill(grad.begin(), grad.end(), 0.0);
    double loss = 0.0;

    for (const TrainingStructure& s : structures) {
        const NeighList list = s.list();
        potential.compute(list);
        const double energy = potential.get_energy();
        const double* forces = potential.get_forces();
        const double* virial = potential.get_virial();

        dloss_dforces.assign((size_t) list.n_atoms * 3, 0.0);
        double dloss_dvirial[6] = {};
        double dloss_denergy = 0.0;

        // the energy is added last, for less round-off
        if (options.force_weight != 0 && s.has_forces)
            for (int ii = 0; ii < list.inum; ii++) {
                const int i = list.ilist[ii];
                const double wgt = force_weight(s, i);
                for (int a = 0; a < 3; a++) {
                    const double diff = forces[i * 3 + a] - s.forces[i * 3 + a];
                    loss += wgt * diff * diff;
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
                    dloss_dvirial[k] += 2.0 * wgt * diff;
                }
        }

        if (options.energy_weight != 0 && s.has_energy) {
            const double wgt = energy_weight(s);
            const double diff = energy - s.energy;
            loss += wgt * diff * diff;
            dloss_denergy = 2.0 * wgt * diff;
        }

        potential.eval_loss_grad(list, dloss_denergy, dloss_dforces.data(), dloss_dvirial, grad.data());
    }

    return loss;
}

void MTPTrainer::add_penalty(double& loss, std::vector<double>& grad) {
    const int C = potential.get_species_count();
    const int K = potential.get_radial_func_count();
    const int R = potential.get_radial_basis_size();
    const double coeff = options.penalty_weight;

    // the norm of each radial function over all species pairs, towards 1
    for (int k = 0; k < K; k++) {
        double norm = 0;
        for (int p = 0; p < C * C; p++)
            for (int l = 0; l < R; l++) {
                const int idx = p * K * R + k * R + l;
                norm += coeffs[idx] * coeffs[idx];
            }

        loss += coeff * (norm - 1) * (norm - 1);
        for (int p = 0; p < C * C; p++)
            for (int l = 0; l < R; l++) {
                const int idx = p * K * R + k * R + l;
                grad[idx] += coeff * 4 * (norm - 1) * coeffs[idx];
            }
    }

    // the overlap of each two radial functions, towards 0
    for (int k1 = 0; k1 < K; k1++)
        for (int k2 = k1 + 1; k2 < K; k2++) {
            double scal = 0;
            for (int p = 0; p < C * C; p++)
                for (int l = 0; l < R; l++)
                    scal += coeffs[p * K * R + k1 * R + l] * coeffs[p * K * R + k2 * R + l];

            loss += coeff * scal * scal;
            for (int p = 0; p < C * C; p++)
                for (int l = 0; l < R; l++) {
                    const int idx1 = p * K * R + k1 * R + l;
                    const int idx2 = p * K * R + k2 * R + l;
                    grad[idx1] += coeff * 2 * scal * coeffs[idx2];
                    grad[idx2] += coeff * 2 * scal * coeffs[idx1];
                }
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
void MTPTrainer::orthogonalize() {
    const int C = potential.get_species_count();
    const int K = potential.get_radial_func_count();
    const int R = potential.get_radial_basis_size();

    for (int k = 0; k < K; k++) {
        for (int k2 = 0; k2 < k; k2++) {
            double norm = 0;
            for (int p = 0; p < C * C; p++)
                for (int l = 0; l < R; l++) {
                    const int idx = p * K * R + k * R + l;
                    norm += coeffs[idx] * coeffs[idx];
                }

            double scal = 0;
            for (int p = 0; p < C * C; p++)
                for (int l = 0; l < R; l++)
                    scal += coeffs[p * K * R + k * R + l] * coeffs[p * K * R + k2 * R + l];

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

/* ---------------------------------------------------------------------- */
double MTPTrainer::energy_weight(const TrainingStructure& s) const {
    return options.energy_weight / std::pow(s.ilist.size(), options.weight_scaling) * options.select_factor;
}

double MTPTrainer::force_weight(const TrainingStructure& s, int i) const {
    double wgt = options.force_weight / std::pow(s.ilist.size(), options.weight_scaling_forces);
    if (options.scale_by_force > 0.0) {
        const double* f = s.forces.data() + i * 3;
        wgt *= options.scale_by_force / (f[0] * f[0] + f[1] * f[1] + f[2] * f[2] + options.scale_by_force);
    }
    return wgt * options.select_factor;
}

double MTPTrainer::stress_weight(const TrainingStructure& s) const {
    return options.stress_weight / std::pow(s.ilist.size(), 1) * options.select_factor;
}
