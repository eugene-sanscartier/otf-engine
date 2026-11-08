/* -*- c++ -*- ----------------------------------------------------------
   Fits an MTP to reference energies, forces and stresses.
   Algorithm of mlip-3's `mlp train`; see mtp_trainer.h.
   Copyright (c) 2023, Alexander Shapeev (Skoltech). BSD 2-Clause, see LICENSE.mlip-3.
------------------------------------------------------------------------- */

#include "mtp_trainer.h"

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

// Virial component of each entry of the symmetric 3x3 stress.
static const int VIRIAL_INDEX[3][3] = {{0, 3, 4}, {3, 1, 5}, {4, 5, 2}};

// Whether mlp's BFGS refits the linear coefficients before this step.
static bool linear_fit_step(int step) {
    return step == 25 || step == 70 || step == 100 || step == 150 || step == 250 || step == 400;
}

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
    reg_vector.assign(n_linear, options.reg_param);
    // radial_only and bfgs_only step mlp's BFGS on a trained potential, without varpro or riemannian
    if (options.radial_only || options.bfgs_only) this->options.varpro = this->options.riemannian = false;
    if (this->options.varpro) this->options.reg_once = true;    // a regularization that moved during BFGS would change the objective

    // N atoms of k species, n_s of species s: each atom of s weighs N / (k n_s), its species N / k.
    if (options.balance_species)
        for (TrainingStructure& s : this->structures) {
            std::vector<int> counts;
            for (int i : s.ilist) {
                if (s.types[i] >= (int) counts.size()) counts.resize(s.types[i] + 1, 0);
                counts[s.types[i]]++;
            }
            const int k = (int) std::count_if(counts.begin(), counts.end(), [](int n) { return n > 0; });
            s.species_factors.assign(counts.size(), 0.0);
            for (size_t t = 0; t < counts.size(); t++)
                if (counts[t] > 0) s.species_factors[t] = (double) s.ilist.size() / (k * counts[t]);
        }

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
    const auto start = std::chrono::steady_clock::now();
    MPI_Barrier(comm);
    if (log) *log << structure_count << " configurations found in the training set" << std::endl;

    auto log_errors = [&](const std::string& title) {
        if (!potential.is_trained()) return;
        potential.set_coeffs(coeffs.data());
        const std::string table = error_table(potential, structures, options.species, comm, title);
        if (log) *log << table;
    };

    add_species();
    if (log) *log << coeffs.size() << " coefficients (" << n_radial << " radial, " << n_linear << " linear)" << std::endl;

    // Every rank checks the same options and sizes, so every rank raises together.
    if ((options.radial_only || options.bfgs_only) && !potential.is_trained())
        throw std::runtime_error("train_mtp: radial_only and bfgs_only need a trained potential");
    const int trained = options.radial_only || options.varpro ? n_radial : (int) coeffs.size();
    if (inverse_hessian_size != 0 && inverse_hessian_size != trained)
        throw std::runtime_error("train_mtp: inverse_hessian must be " + std::to_string(trained) + " x " + std::to_string(trained) + " for the coefficients trained");
    if (options.riemannian && inverse_hessian_size != 0)
        throw std::runtime_error("train_mtp: riemannian takes no inverse_hessian; pass riemannian=false");
    log_errors("Training set errors before training:");    // the potential as given, before update_min_dist moves its radial basis
    update_min_dist();

    if (options.init_random && !potential.is_trained()) {
        if (rank == 0) {
            std::random_device rand_device;
            std::default_random_engine generator(options.random_seed >= 0 ? (unsigned) options.random_seed : rand_device());
            std::uniform_real_distribution<> uniform(-1.0, 1.0);

            if (log) *log << "Random initialization of radial coefficients" << std::endl;
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
            if (options.balance_species) *log << "Force weights balanced over species" << std::endl;
        }

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
            fit_linear();
            rescale();
        }
    }

    potential.set_coeffs(coeffs.data());
    log_errors("Training set errors:");
    if (log) *log << "Trained in " << std::round(std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() * 10) / 10 << " s" << std::endl;
    MPI_Barrier(comm);
}

/* ----------------------------------------------------------------------
   Errors as mlip-3's ErrorMonitor measures them: energies, energies per
   atom, forces per atom, and the virial as the Frobenius norm of the 3x3.
   Relative to the reference, as the ErrorMonitor's Max(diff)/Max(value)
   and RMS(diff)/RMS(value), except for energies, whose reference is their
   residual from per-species energies fitted to the structures by least
   squares, and whose max |diff| is relative to the residuals' range.
------------------------------------------------------------------------- */
std::string error_table(PairMTP& potential, const std::vector<TrainingStructure>& structures, const std::vector<std::string>& species, MPI_Comm comm, const std::string& title) {
    enum { ENE, EPA, FRC, VIR, KINDS };    // then the forces of each species
    const int species_count = potential.get_species_count();
    const int rows = KINDS + species_count;
    const int C = species_count;

    // the normal equations of the per-species energies: the sums of counts * counts and of counts * energy, and the
    // number of energies
    std::vector<double> fit(C * C + C + 1, 0.0), total_fit(C * C + C + 1), counts(C);
    for (const TrainingStructure& s : structures) {
        if (!s.has_energy) continue;
        std::fill(counts.begin(), counts.end(), 0.0);
        for (int t : s.types)
            if (t < C) counts[t] += 1.0;
        for (int a = 0; a < C; a++) {
            for (int b = 0; b < C; b++) fit[a * C + b] += counts[a] * counts[b];
            fit[C * C + a] += counts[a] * s.energy;
        }
        fit[C * C + C] += 1.0;
    }
    MPI_Allreduce(fit.data(), total_fit.data(), (int) fit.size(), MPI_DOUBLE, MPI_SUM, comm);

    // Gauss-Jordan, pivoting on the largest remaining diagonal; species beyond the independent compositions get no energy
    double* G = total_fit.data();
    double* g = G + C * C;
    std::vector<int> pivots;
    std::vector<double> species_energy(C, 0.0);
    double largest = 0.0;
    for (int a = 0; a < C; a++) largest = std::max(largest, G[a * C + a]);
    for (int step = 0; step < C; step++) {
        int k = -1;
        for (int a = 0; a < C; a++)
            if (std::find(pivots.begin(), pivots.end(), a) == pivots.end() && (k < 0 || G[a * C + a] > G[k * C + k])) k = a;
        if (G[k * C + k] <= 1e-12 * largest) break;
        pivots.push_back(k);
        for (int i = 0; i < C; i++) {
            if (i == k) continue;
            const double ratio = G[i * C + k] / G[k * C + k];
            for (int j = 0; j < C; j++) G[i * C + j] -= ratio * G[k * C + j];
            g[i] -= ratio * g[k];
        }
    }
    for (int k : pivots) species_energy[k] = g[k] / G[k * C + k];
    // with no more energies than independent compositions, the residuals vanish by construction
    const bool residuals = total_fit[C * C + C] > (double) pivots.size();

    // per row: the count, and the sums of |diff|, diff^2 and value^2; the maxima of |diff|, value and -value, where
    // the value is the energy's residual, or the norm of the reference force or virial; and where the max |diff| is,
    // as structure number * 1e6 + atom number
    std::vector<double> sums(rows * 4, 0.0), maxima(rows * 3, -1e300), at(rows, 0.0);
    auto add = [&](int row, double dltsq, double value, double where) {
        double* sum = &sums[row * 4];
        double* max = &maxima[row * 3];
        const double delta = std::sqrt(dltsq);
        sum[0] += 1.0;
        sum[1] += delta;
        sum[2] += dltsq;
        sum[3] += value * value;
        if (delta > max[0]) at[row] = where;
        max[0] = std::max(max[0], delta);
        max[1] = std::max(max[1], value);
        max[2] = std::max(max[2], -value);
    };

    for (const TrainingStructure& s : structures) {
        const NeighList list = s.list();
        potential.compute(list);
        const double* forces = potential.get_forces();
        const double* virial = potential.get_virial();

        if (s.has_energy) {
            const double diff = potential.get_energy() - s.energy;
            double residual = s.energy;
            for (int t : s.types)
                if (t < C) residual -= species_energy[t];
            add(ENE, diff * diff, residual, s.number * 1e6);
            add(EPA, diff * diff / ((double) s.n_atoms() * s.n_atoms()), residual / s.n_atoms(), s.number * 1e6);
        }
        if (s.has_forces)
            for (int ii = 0; ii < list.inum; ii++) {
                const int i = list.ilist[ii];
                double dltsq = 0.0, valsq = 0.0;
                for (int a = 0; a < 3; a++) {
                    dltsq += (forces[i * 3 + a] - s.forces[i * 3 + a]) * (forces[i * 3 + a] - s.forces[i * 3 + a]);
                    valsq += s.forces[i * 3 + a] * s.forces[i * 3 + a];
                }
                add(FRC, dltsq, std::sqrt(valsq), s.number * 1e6 + i + 1);
                if (s.types[i] < species_count) add(KINDS + s.types[i], dltsq, std::sqrt(valsq), s.number * 1e6 + i + 1);
            }
        if (s.has_virial) {
            double dltsq = 0.0, valsq = 0.0;
            for (int a = 0; a < 3; a++)
                for (int b = 0; b < 3; b++) {
                    const int k = VIRIAL_INDEX[a][b];
                    dltsq += (virial[k] - s.virial[k]) * (virial[k] - s.virial[k]);
                    valsq += s.virial[k] * s.virial[k];
                }
            add(VIR, dltsq, std::sqrt(valsq), s.number * 1e6);
        }
    }

    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    std::vector<double> total_sums(rows * 4), total_maxima(rows * 3), total_at(rows);
    MPI_Reduce(sums.data(), total_sums.data(), rows * 4, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Allreduce(maxima.data(), total_maxima.data(), rows * 3, MPI_DOUBLE, MPI_MAX, comm);
    // the max |diff|'s place, the first in number among the ranks holding it
    for (int r = 0; r < rows; r++)
        if (maxima[r * 3] != total_maxima[r * 3]) at[r] = 1e300;
    MPI_Reduce(at.data(), total_at.data(), rows, MPI_DOUBLE, MPI_MIN, 0, comm);
    if (rank != 0) return "";

    char line[160];
    std::ostringstream out;
    out << title << "\n";
    std::snprintf(line, sizeof line, "%-12s %7s %9s %9s %9s %9s %9s\n", "", "n", "MAE", "RMSE", "max", "rel max", "rel RMSE");
    out << line;
    auto row = [&](int r, const std::string& name, const char* unit) {
        const double* sum = &total_sums[r * 4];
        const double* max = &total_maxima[r * 3];
        if (sum[0] == 0) return;
        const double rmse = std::sqrt(sum[2] / sum[0]);
        const bool energy = r == ENE || r == EPA;
        const bool scaled = !energy || residuals;
        // a relative error over a zero reference scale, as of energies with vanishing residuals, is shown as a dash
        auto relative = [](double value, double scale) {
            char field[16];
            if (scale > 0) std::snprintf(field, sizeof field, " %9.2e", value / scale);
            else std::snprintf(field, sizeof field, " %8s—", "");
            return std::string(field);
        };
        const std::string rel_max = relative(max[0], !scaled ? 0.0 : energy ? max[1] + max[2] : max[1]);
        const std::string rel_rmse = relative(rmse, scaled ? std::sqrt(sum[3] / sum[0]) : 0.0);
        std::snprintf(line, sizeof line, "%-12s %7ld %9.2e %9.2e %9.2e%s%s  %s\n", name.c_str(), (long) sum[0], sum[1] / sum[0], rmse, max[0], rel_max.c_str(), rel_rmse.c_str(), unit);
        out << line;
    };
    row(ENE, "energy", "eV");
    row(EPA, "energy/atom", "eV/atom");
    row(FRC, "force", "eV/A");
    for (int t = 0; t < species_count; t++)
        row(KINDS + t, "  " + (t < (int) species.size() ? species[t] : "type " + std::to_string(t)), "eV/A");
    row(VIR, "virial", "eV");
    if (total_sums[EPA * 4]) {
        std::snprintf(line, sizeof line, "max energy/atom error: %.2e eV/atom (structure %ld)\n", total_maxima[EPA * 3], (long) (total_at[EPA] / 1e6));
        out << line;
    }
    if (total_sums[FRC * 4]) {
        std::snprintf(line, sizeof line, "max force error: %.2e eV/A (structure %ld, atom %ld)\n", total_maxima[FRC * 3], (long) (total_at[FRC] / 1e6), (long) std::fmod(total_at[FRC], 1e6));
        out << line;
    }
    return out.str();
}

/* ----------------------------------------------------------------------
   Appends the species of the training set that the potential lacks, all of
   them when it is untrained. Every species of the potential must occur in
   the training set, and a species is numbered by its index.
------------------------------------------------------------------------- */
void MTPTrainer::add_species() {
    // the species count, and minus the lowest species when it is negative, so that every rank raises together
    int local[2] = {0, 0}, global[2] = {0, 0};
    for (const TrainingStructure& s : structures)
        for (int t : s.types) {
            local[0] = std::max(local[0], t + 1);
            local[1] = std::max(local[1], -t);
        }
    MPI_Allreduce(local, global, 2, MPI_INT, MPI_MAX, comm);
    if (global[1] > 0)
        throw std::runtime_error("MTPTrainer: a training structure has species " + std::to_string(-global[1]));
    const int type_count = global[0];

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
        if (log) *log << "min_dist = " << total_min_dist << ", mindist updated" << std::endl;
        potential.set_min_cutoff(0.99 * total_min_dist);
    }
}

/* ----------------------------------------------------------------------
   BFGS over every coefficient. Rank 0 steps; the others evaluate their
   share at the coefficients it broadcasts.
------------------------------------------------------------------------- */
void MTPTrainer::fit_nonlinear(int max_iter) {
    const auto start = std::chrono::steady_clock::now();
    MPI_Barrier(comm);
    if (log) {
        *log << "MTPR training started on " << size << " core(s)" << std::endl;
        char header[128];
        std::snprintf(header, sizeof header, "%17s%-15s%-16s%-16s%-16s%s", "", "loss", "energy", "force", "stress", "penalty");
        *log << header << std::endl;
    }

    // BFGS steps the leading n coefficients: all of them, or the radial block
    const int n = options.radial_only || options.varpro ? n_radial : (int) coeffs.size();
    std::vector<double> bfgs_g(n);
    double bfgs_f = 0.0, bfgs_terms[4] = {};

    if (rank == 0) {
        bfgs.set_x(coeffs.data(), n);
        if (inverse_hessian)
            bfgs.set_inverse_hessian(inverse_hessian);
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

        if (rank == 0) std::copy(bfgs.x().begin(), bfgs.x().end(), coeffs.begin());
        MPI_Bcast(coeffs.data(), (int) coeffs.size(), MPI_DOUBLE, 0, comm);
        bfgs_f = evaluate(bfgs_g, n, true, bfgs_terms);

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
                if (log) {
                    // mlp's line in columns, with each term of the loss and its share
                    char line[160];
                    int length = std::snprintf(line, sizeof line, "BFGS iter %4d:  %.7e", num_step, bfgs_f);
                    for (double term : bfgs_terms)
                        length += std::snprintf(line + length, sizeof line - length, "  %.1e (%3.0f%%)", term, 100 * term / bfgs_f);
                    *log << line << std::endl;
                }
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

        MPI_Bcast(&converge, 1, MPI_C_BOOL, 0, comm);
        MPI_Bcast(&linesearch, 1, MPI_C_BOOL, 0, comm);
        MPI_Bcast(&num_step, 1, MPI_INT, 0, comm);
    }

    if (rank == 0 && inverse_hessian)
        std::copy(bfgs.inverse_hessian().begin(), bfgs.inverse_hessian().end(), inverse_hessian);

    // the potential counts as trained from its first fit on
    potential.set_trained(true);
    // the next linear fit rebuilds the regularization
    reg_init = true;

    if (log) {
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        char line[96];
        std::snprintf(line, sizeof line, "%d iterations in %.1f s @ %.2g s/iteration", num_step, seconds, seconds / std::max(num_step, 1));
        *log << line << std::endl;
        *log << "MTPR training ended" << std::endl;
    }
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
void MTPTrainer::fit_riemannian(int max_iter) {
    const auto start = std::chrono::steady_clock::now();
    MPI_Barrier(comm);
    if (log) {
        *log << "Riemannian L-BFGS started on " << size << " core(s)" << std::endl;
        char header[128];
        std::snprintf(header, sizeof header, "%17s%-15s%-16s%-16s%-16s%-16s%s", "", "loss", "energy", "force", "stress", "penalty", "|WW^T-I|");
        *log << header << std::endl;
    }

    // the coordinates stepped: the radial block, and the linear one without varpro
    const int n = options.varpro ? n_radial : (int) coeffs.size();
    const int K = potential.get_radial_func_count();
    std::vector<double> g(n), g_new(n), p(n), x0(n);
    std::deque<std::vector<double>> s_memory, y_memory;    // tangent at the current point
    double terms[4] = {};
    int evaluations = 0;
    auto dot = [n](const std::vector<double>& a, const std::vector<double>& b) {
        double sum = 0.0;
        for (int i = 0; i < n; i++) sum += a[i] * b[i];
        return sum;
    };

    // the loss at coeffs, and its Riemannian gradient into out, on every rank
    auto evaluate_tangent = [&](std::vector<double>& out) {
        MPI_Bcast(coeffs.data(), (int) coeffs.size(), MPI_DOUBLE, 0, comm);
        double loss = evaluate(out, n, false, terms);
        MPI_Bcast(&loss, 1, MPI_DOUBLE, 0, comm);
        MPI_Bcast(out.data(), n, MPI_DOUBLE, 0, comm);
        project_tangent(coeffs.data(), out.data());
        evaluations++;
        return loss;
    };

    // the start on the manifold, a lower-triangular change of the radial functions, with the linear coefficients refitted
    retract(coeffs);
    if (!options.varpro) {
        fit_linear(false);
        reg_init = false;
    }
    double f = evaluate_tangent(g);

    int num_step = 0;
    double linf = 9e99;    // the loss 50 steps ago
    while (true) {
        if (!options.varpro && linear_fit_step(num_step)) {
            fit_linear(false);
            f = evaluate_tangent(g);
            // a new regularization invalidates the memory
            if (reg_init) {
                s_memory.clear();
                y_memory.clear();
                reg_init = false;
            }
        }

        if (log) {
            const std::vector<double> gram = radial_overlaps(coeffs.data(), coeffs.data());
            double violation = 0.0;
            for (int k1 = 0; k1 < K; k1++)
                for (int k2 = 0; k2 < K; k2++)
                    violation += (gram[k1 * K + k2] - (k1 == k2)) * (gram[k1 * K + k2] - (k1 == k2));
            char line[192];
            int length = std::snprintf(line, sizeof line, "Riemann iter %4d:  %.7e", num_step, f);
            for (double term : terms)
                length += std::snprintf(line + length, sizeof line - length, "  %.1e (%3.0f%%)", term, 100 * term / f);
            std::snprintf(line + length, sizeof line - length, "  %.1e", std::sqrt(violation));
            *log << line << std::endl;
        }
        if (rank == 0 && checkpoint) {
            potential.set_coeffs(coeffs.data());
            checkpoint();
        }

        if (num_step >= max_iter) {
            if (log) *log << "step limit reached" << std::endl;
            break;
        }
        if (num_step % 50 == 0 && num_step > 100) {
            if ((linf - f) / f < options.tolerance) {
                if (log) *log << "L-BFGS ended due to small decr. in 50 iterations" << std::endl;
                break;
            }
            linf = f;
        }
        const double g_norm = std::sqrt(dot(g, g));
        if (g_norm == 0.0) break;

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
        double t = 1.0, f_new = f;
        int halvings = 0;
        for (; halvings <= 40; halvings++, t *= 0.5) {
            for (int j = 0; j < n; j++) coeffs[j] = x0[j] + t * p[j];
            retract(coeffs);
            f_new = evaluate_tangent(g_new);
            if (f_new <= f + 1e-4 * t * slope) break;
        }
        if (halvings > 40) {
            std::copy(x0.begin(), x0.end(), coeffs.begin());
            f = evaluate_tangent(g);
            if (log) *log << "L-BFGS line search failed" << std::endl;
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

    if (log) {
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        char line[128];
        std::snprintf(line, sizeof line, "%d iterations, %d evaluations in %.1f s @ %.2g s/iteration", num_step, evaluations, seconds, seconds / std::max(num_step, 1));
        *log << line << std::endl;
        *log << "MTPR training ended" << std::endl;
    }
}

/* ----------------------------------------------------------------------
   Least squares for the species and linear coefficients, after
   orthonormalizing the radial functions unless told not to. Rank 0 solves the summed system.
------------------------------------------------------------------------- */
void MTPTrainer::fit_linear(bool orthonormalize) {
    if (orthonormalize) orthogonalize();
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

    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> matrix(lin_matrix.data(), n, n);
    Eigen::Map<Eigen::VectorXd> vector(lin_vector.data(), n);

    // E, F and virial are linear in these coefficients, with the gradients as components: a structure's equations are
    // the components of each value it has, and its references the values, each scaled by the square root of its weight
    int most_atoms = 0;
    for (const TrainingStructure& s : structures) most_atoms = std::max(most_atoms, (int) s.ilist.size());
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> equations(1 + 3 * most_atoms + 9, n);
    Eigen::VectorXd references(equations.rows());
    std::vector<double> scratch_rows;
    if (options.varpro) linear_rows.resize(structures.size());
    for (size_t index = 0; index < structures.size(); index++) {
        const TrainingStructure& s = structures[index];
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
                const double root = std::sqrt(force_weight(s, atom));
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
    if (options.reg_once && reg_built) reg_init = false;
    if (!reg_init && !options.reg_once)
        for (int i = 0; i < n; i++)
            if (reg_vector[i] < 1e-2 * options.reg_param * std::max(1.0, A[i * n + i]) / ts_size || reg_vector[i] > 1e2 * options.reg_param * std::max(1.0, A[i * n + i]) / ts_size) {
                reg_init = true;
                if (log) *log << "Regularization parameters updated. Hessian in BFGS is reset" << std::endl;
                break;
            }

    if (reg_init) {
        for (int i = 0; i < n; i++)
            reg_vector[i] = options.reg_param * std::max(1.0, A[i * n + i]) / ts_size;
        reg_built = true;
    }

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

double MTPTrainer::objective(const std::vector<double>& at, std::vector<double>& grad) {
    // fixes the regularization at the potential's own radial coefficients, where BFGS's first evaluation fixes it
    if (options.varpro) fit_linear(false);
    std::copy(at.begin(), at.end(), coeffs.begin());
    double terms[4];
    grad.assign(coeffs.size(), 0.0);
    double total = evaluate(grad, (int) grad.size(), !options.riemannian, terms);
    MPI_Bcast(&total, 1, MPI_DOUBLE, 0, comm);
    MPI_Bcast(grad.data(), (int) grad.size(), MPI_DOUBLE, 0, comm);
    // the linear coefficients are a function of the radial ones: their partial gradient is zero at the optimum
    if (options.varpro) std::fill(grad.begin() + n_radial, grad.end(), 0.0);
    // under riemannian, the Riemannian gradient at the radial functions in at, which must be orthonormal
    if (options.riemannian) project_tangent(coeffs.data(), grad.data());
    return total;
}

double MTPTrainer::evaluate(std::vector<double>& grad, int n, bool radial, double terms[4]) {
    if (options.varpro) fit_linear(false);    // the linear coefficients at their optimum for these radial ones
    local_grad.resize(coeffs.size());
    double loss = loss_grad(local_grad) / structure_count;
    for (double& g : local_grad) g /= structure_count;
    double local_terms[4] = {loss_terms[0] / structure_count, loss_terms[1] / structure_count, loss_terms[2] / structure_count, loss};
    if (rank == 0) add_penalty(loss, local_grad, radial);
    local_terms[3] = loss - local_terms[3];

    double total = 0.0;
    MPI_Reduce(&loss, &total, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(local_terms, terms, 4, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(local_grad.data(), grad.data(), n, MPI_DOUBLE, MPI_SUM, 0, comm);
    return total;
}

/* ---------------------------------------------------------------------- */
double MTPTrainer::loss_grad(std::vector<double>& grad) {
    potential.set_coeffs(coeffs.data());
    std::fill(grad.begin(), grad.end(), 0.0);
    double loss = 0.0;
    std::fill_n(loss_terms, 3, 0.0);
    const bool from_rows = linear_rows.size() == structures.size() && linear_rows_at.size() == (size_t) n_radial + 1 && std::equal(coeffs.begin(), coeffs.begin() + n_radial, linear_rows_at.begin()) && linear_rows_at[n_radial] == potential.get_scaling();

    for (size_t index = 0; index < structures.size(); index++) {
        const TrainingStructure& s = structures[index];
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
            forces = potential.get_forces();
            virial = potential.get_virial();
        }

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

void MTPTrainer::add_penalty(double& loss, std::vector<double>& grad, bool radial) {
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
void MTPTrainer::orthogonalize() {
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

double MTPTrainer::row_dot(const double* a, int k1, const double* b, int k2) const {
    const int P = potential.get_species_count() * potential.get_species_count();
    const int K = potential.get_radial_func_count();
    const int R = potential.get_radial_basis_size();
    double sum = 0.0;
    for (int p = 0; p < P; p++)
        for (int l = 0; l < R; l++) sum += a[(p * K + k1) * R + l] * b[(p * K + k2) * R + l];
    return sum;
}

void MTPTrainer::row_axpy(double* y, int k1, double factor, const double* x, int k2) const {
    const int P = potential.get_species_count() * potential.get_species_count();
    const int K = potential.get_radial_func_count();
    const int R = potential.get_radial_basis_size();
    for (int p = 0; p < P; p++)
        for (int l = 0; l < R; l++) y[(p * K + k1) * R + l] += factor * x[(p * K + k2) * R + l];
}

std::vector<double> MTPTrainer::radial_overlaps(const double* a, const double* b) const {
    const int K = potential.get_radial_func_count();
    std::vector<double> overlaps(K * K);
    for (int k1 = 0; k1 < K; k1++)
        for (int k2 = 0; k2 < K; k2++) overlaps[k1 * K + k2] = row_dot(a, k1, b, k2);
    return overlaps;
}

void MTPTrainer::project_tangent(const double* w, double* v) const {
    const int K = potential.get_radial_func_count();
    const std::vector<double> overlaps = radial_overlaps(v, w);
    for (int k1 = 0; k1 < K; k1++)
        for (int k2 = 0; k2 < K; k2++) row_axpy(v, k1, -0.5 * (overlaps[k1 * K + k2] + overlaps[k2 * K + k1]), w, k2);
}

void MTPTrainer::retract(std::vector<double>& c) const {
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
double MTPTrainer::energy_weight(const TrainingStructure& s) const {
    return options.energy_weight / std::pow(s.ilist.size(), options.weight_scaling) * options.select_factor;
}

double MTPTrainer::force_weight(const TrainingStructure& s, int i) const {
    double wgt = options.force_weight / std::pow(s.ilist.size(), options.weight_scaling_forces);
    if (options.scale_by_force > 0.0) {
        const double* f = s.forces.data() + i * 3;
        wgt *= options.scale_by_force / (f[0] * f[0] + f[1] * f[1] + f[2] * f[2] + options.scale_by_force);
    }
    if (options.balance_species)
        wgt *= s.species_factors[s.types[i]];
    return wgt * options.select_factor;
}

double MTPTrainer::stress_weight(const TrainingStructure& s) const {
    return options.stress_weight / s.ilist.size() * options.select_factor;
}
