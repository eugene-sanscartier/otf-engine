/* -*- c++ -*- ----------------------------------------------------------
   Fit, the base of MTPFit and ETNFit.

   A fit runs SPMD over a Comm. Each rank holds a copy of the potential and
   a share of the structures, and the collectives of the Comm combine them.
------------------------------------------------------------------------- */

#pragma once

#include "comm.h"
#include "error_table.h"
#include "fit_log.h"
#include "fit_structure.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <ostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// The options of both fits, with their defaults.
struct FitOptions {
    double energy_weight = 1.0;
    double force_weight = 0.01;
    double stress_weight = 0.001;
    bool balance_species = true;         // weighs each force of species s by N / (k n_s), for N atoms of k species and n_s of species s
    double varpro_fraction = 0.25;       // below 1, VarPro builds its system from this fraction of the structures, with weights divided by it
    bool no_mindist_update = false;      // keeps min_dist; otherwise the fit sets it to 0.99 of the shortest training distance
    std::vector<std::string> species;    // names of the species in the error tables

    // Returns the options line of the log that gives the loss weights and balance_species.
    std::string loss_weights() const {
        std::ostringstream text;
        text << "energy_weight " << energy_weight << ", force_weight " << force_weight << ", stress_weight " << stress_weight << (balance_species ? ", balance_species" : "");
        return text.str();
    }
};

// Minimum average number of structures that varpro_fraction draws. Below it, the fit takes every structure.
constexpr int VARPRO_MIN_DRAWN = 32;

// Optional columns of the step lines of a phase: the MTP penalty, the loss after a linear fit, the inverse Hessian set.
struct StepColumns {
    bool penalty, linear, inv_hess;
};

// A fit of PotentialT (MTP or ETN) with the options OptionsT.
template <class PotentialT, class OptionsT>
class Fit {
  public:
    using Potential = PotentialT;
    using Options = OptionsT;

    // Sets up the fit of potential on the share of structures of this rank.
    // log: the stream of the fit log, or null. checkpoint: saves the potential once it holds coeffs, or null.
    // Throws on every rank together when no rank has a structure.
    Fit(Potential& potential, std::vector<FitStructure> structures, const Options& options, Comm& comm, std::ostream* log, std::function<void()> checkpoint)
        : potential(potential), structures(std::move(structures)), options(options), comm(comm), rank(comm.rank()), size(comm.size()), log(log), checkpoint(std::move(checkpoint)) {
        if (options.balance_species)
            for (FitStructure& s : this->structures) s.balance_species();

        int local_count = (int) this->structures.size();
        comm.allreduce(&local_count, &structure_count, 1, ReduceOp::sum);
        if (structure_count == 0)
            throw std::runtime_error("no training structures");
        if (this->options.varpro_fraction * structure_count < VARPRO_MIN_DRAWN) this->options.varpro_fraction = 1.0;

        load_coeffs();
    }

  protected:
    // Copies the coefficients of the potential into coeffs.
    void load_coeffs() {
        coeffs.resize(potential.coeff_count());
        potential.get_coeffs(coeffs.data());
    }

    // Logs model (MTP, ETN), the structure and rank counts, and the options.
    void log_header(const char* model) {
        log_event(log, "fit", "%s, %d structures, %d ranks", model, structure_count, size);
        for (const std::string& line : options.summary())
            log_event(log, "options", "%s", line.c_str());
    }

    // Logs the species old_spec .. new_spec - 1 that training adds to the potential.
    void log_added_species(int old_spec, int new_spec) {
        std::string added;
        for (int t = old_spec; t < new_spec; t++)
            added += ' ' + std::to_string(t);
        if (new_spec > old_spec) log_event(log, "species", "added%s", added.c_str());
    }

    // Sets the optional columns of the phase and logs the column names of log_step.
    void log_step_header(StepColumns phase_columns) {
        columns = phase_columns;
        if (!log) return;
        char line[200];
        int length = std::snprintf(line, sizeof line, "%5s  %-13s  %-14s  %-14s  %-14s", "step", "loss", "energy", "force", "stress");
        if (columns.penalty) length += std::snprintf(line + length, sizeof line - length, "  %-14s", "penalty");
        length += std::snprintf(line + length, sizeof line - length, "  %-8s  %6s", "alpha", "nalpha");
        if (columns.linear) length += std::snprintf(line + length, sizeof line - length, "  %-13s", "linear");
        if (columns.inv_hess) std::snprintf(line + length, sizeof line - length, "  %s", "inv_hess");
        *log << line << std::endl;
    }

    // Logs an accepted step in the columns of the phase.
    // terms: the energy, force, stress and penalty terms, each logged with its share of loss.
    // alpha: the step length. nalpha: the loss evaluations. linear: the loss after a linear fit at the step, - if NaN.
    // inv_hess: what BFGS set the inverse Hessian to in the step, - if empty.
    void log_step(int step, double loss, const double* terms, double alpha, int nalpha, double linear, const char* inv_hess) {
        if (!log) return;
        char line[200];
        int length = std::snprintf(line, sizeof line, "%5d  %.7e", step, loss);
        for (int k = 0; k < (columns.penalty ? 4 : 3); k++)
            length += std::snprintf(line + length, sizeof line - length, "  %.1e (%3.0f%%)", terms[k], 100 * terms[k] / loss);
        length += std::snprintf(line + length, sizeof line - length, "  %.2e  %6d", alpha, nalpha);
        if (columns.linear) length += std::isnan(linear) ? std::snprintf(line + length, sizeof line - length, "  %-13s", "-") : std::snprintf(line + length, sizeof line - length, "  %.7e", linear);
        if (columns.inv_hess) std::snprintf(line + length, sizeof line - length, "  %s", *inv_hess ? inv_hess : "-");
        *log << line << std::endl;
    }

    // Logs the time of a phase since start, its step count and its time per step.
    void log_phase_time(int steps, std::chrono::steady_clock::time_point start) {
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        log_event(log, "timing", "%.3g s @ %d steps, %.2g s / step", seconds, steps, seconds / std::max(steps, 1));
    }

    // Logs the time of the whole fit since start.
    void log_fit_time(std::chrono::steady_clock::time_point start) {
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        log_event(log, "timing", "%.3g s for the fit", seconds);
    }

    // Returns the species count of the training set over every rank. Throws on every rank together unless the species
    // run from 0 to N-1, each present in the training set, and every species of a trained potential is present.
    int training_species_count() {
        // [0] the species count, [1] minus the lowest species when it is negative
        int local[2] = {0, 0}, global[2] = {0, 0};
        for (const FitStructure& s : structures)
            for (int t : s.types) {
                local[0] = std::max(local[0], t + 1);
                local[1] = std::max(local[1], -t);
            }
        comm.allreduce(local, global, 2, ReduceOp::max);
        if (global[1] > 0)
            throw std::runtime_error("a training structure has species " + std::to_string(-global[1]));
        const int type_count = global[0];

        std::vector<int> present(type_count, 0), present_anywhere(type_count, 0);
        for (const FitStructure& s : structures)
            for (int t : s.types)
                present[t] = 1;
        comm.allreduce(present.data(), present_anywhere.data(), type_count, ReduceOp::max);

        const int old_spec = potential.is_trained() ? potential.get_species_count() : 0;
        for (int t = 0; t < std::max(old_spec, type_count); t++)
            if (t >= type_count || !present_anywhere[t])
                throw std::runtime_error(t < old_spec ? "species " + std::to_string(t) + " of the potential is not present in the training set" : "the training set has species " + std::to_string(type_count - 1) + " but not species " + std::to_string(t));
        return type_count;
    }

    // Logs the errors of the potential at coeffs on the training set, under "errors: when", unless it is untrained.
    void log_errors(const std::string& when) {
        if (!potential.is_trained()) return;
        potential.set_coeffs(coeffs.data());
        const std::string table = error_table(potential, structures, options.species, comm, "errors: " + when);
        if (log) *log << table;
    }

    // Sets min_dist to 0.99 of the shortest distance in the training set, unless no_mindist_update.
    void update_min_dist() {
        double min_dist = 999;
        for (const FitStructure& s : structures)
            for (size_t p = 0; p < s.firstneigh.size(); p++) {
                const double* r = s.displacements.data() + p * 3;
                min_dist = std::min(min_dist, std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]));
            }

        double total_min_dist = min_dist;
        comm.allreduce(&min_dist, &total_min_dist, 1, ReduceOp::min);

        if (!options.no_mindist_update) {
            log_event(log, "init", "min_dist %g", total_min_dist);
            potential.set_min_cutoff(0.99 * total_min_dist);
        }
    }

    // Draws each structure of this rank into varpro_subset with probability varpro_fraction, seeded by random_seed,
    // draw and the structure number. Throws on every rank together when no rank draws a structure.
    void draw_varpro_subset(int draw) {
        varpro_subset.assign(structures.size(), false);
        int local = 0, total = 0;
        for (size_t k = 0; k < structures.size(); k++) {
            std::seed_seq seed{options.random_seed, draw, structures[k].number};
            std::mt19937 generator(seed);
            varpro_subset[k] = std::uniform_real_distribution<>(0.0, 1.0)(generator) < options.varpro_fraction;
            local += varpro_subset[k] ? 1 : 0;
        }
        comm.allreduce(&local, &total, 1, ReduceOp::sum);
        if (total == 0)
            throw std::runtime_error("varpro_fraction " + std::to_string(options.varpro_fraction) + " draws no structure");
    }

    // Sets the potential to coeffs and calls checkpoint, if this rank has one.
    void save_checkpoint() {
        if (!checkpoint) return;
        potential.set_coeffs(coeffs.data());
        checkpoint();
    }

    Potential& potential;
    std::vector<FitStructure> structures;
    Options options;
    Comm& comm;
    const int rank, size;
    int structure_count = 0;    // over all ranks
    std::ostream* log;
    std::function<void()> checkpoint;

    std::vector<double> coeffs;           // current coefficients, in the order of get_coeffs
    StepColumns columns;
    double loss_terms[3] = {};            // energy, force and stress terms of the last loss_grad on this rank
    std::vector<double> dloss_dforces;    // per-structure scratch
    std::vector<bool> varpro_subset;      // [structures.size()] whether each structure of this rank is in the draw
};
