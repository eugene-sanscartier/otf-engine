/* -*- c++ -*- ----------------------------------------------------------
   What MTPFit and ETNFit share.

   A fit is SPMD: every rank of a Comm holds a copy of the potential
   and a share of the structures, and the Comm's collectives combine them.
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
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// The options of both fits, with their defaults.
struct FitOptions {
    double energy_weight = 1.0;
    double force_weight = 0.01;
    double stress_weight = 0.001;
    bool balance_species = true;         // weigh each force by N / (k n_s), so every species of a structure has an equal share of its force weight
    std::vector<std::string> species;    // the species' names in the error tables

    // The weights and balance_species, as the log lists them.
    std::string summary() const {
        std::ostringstream text;
        text << "energy_weight " << energy_weight << ", force_weight " << force_weight << ", stress_weight " << stress_weight << (balance_species ? ", balance_species" : "");
        return text.str();
    }
};

// Potential is MTP or ETN, Options its fit's options.
template <class PotentialT, class OptionsT>
class Fit {
  public:
    using Potential = PotentialT;
    using Options = OptionsT;

    // log receives the fit log on the rank given one; checkpoint saves the potential, which then
    // holds the coefficients. Throws on every rank together when no rank has a structure.
    Fit(Potential& potential, std::vector<FitStructure> structures, const Options& options, Comm& comm, std::ostream* log, std::function<void()> checkpoint)
        : potential(potential), structures(std::move(structures)), options(options), comm(comm), rank(comm.rank()), size(comm.size()), log(log), checkpoint(std::move(checkpoint)) {
        if (options.balance_species)
            for (FitStructure& s : this->structures) s.balance_species();

        int local_count = (int) this->structures.size();
        comm.allreduce(&local_count, &structure_count, 1, ReduceOp::sum);
        if (structure_count == 0)
            throw std::runtime_error("no training structures");

        load_coeffs();
    }

  protected:
    // Sets coeffs to the potential's coefficients.
    void load_coeffs() {
        coeffs.resize(potential.coeff_count());
        potential.get_coeffs(coeffs.data());
    }

    // Logs the fit of model (MTP, ETN): its structures, ranks and options.
    void log_header(const char* model) {
        log_event(log, "fit", "%s, %d structures, %d ranks", model, structure_count, size);
        log_event(log, "options", "%s", options.summary().c_str());
    }

    // Logs the species old_spec .. new_spec - 1 that training adds to the potential.
    void log_added_species(int old_spec, int new_spec) {
        std::string added;
        for (int t = old_spec; t < new_spec; t++)
            added += ' ' + std::to_string(t);
        if (new_spec > old_spec) log_event(log, "species", "added%s", added.c_str());
    }

    // Logs the column names of log_step, with the MTP's penalty and linear if mtp.
    void log_step_header(bool mtp) {
        if (!log) return;
        char line[200];
        int length = std::snprintf(line, sizeof line, "%5s  %-13s  %-14s  %-14s  %-14s", "step", "loss", "energy", "force", "stress");
        if (mtp) length += std::snprintf(line + length, sizeof line - length, "  %-14s", "penalty");
        length += std::snprintf(line + length, sizeof line - length, "  %-8s  %6s", "alpha", "nalpha");
        if (mtp) length += std::snprintf(line + length, sizeof line - length, "  %-13s", "linear");
        std::snprintf(line + length, sizeof line - length, "  %s", "inv_hess");
        *log << line << std::endl;
    }

    // Logs an accepted step: its loss, the energy, force and stress terms (and the penalty's if mtp) of terms with their
    // share of the loss, its step length alpha, its loss evaluations nalpha, the loss after a linear fit at the step if mtp
    // (- if NaN), and what BFGS set the inverse Hessian to in the step, inv_hess (- if "").
    void log_step(int step, double loss, const double* terms, bool mtp, double alpha, int nalpha, double linear, const char* inv_hess) {
        if (!log) return;
        char line[200];
        int length = std::snprintf(line, sizeof line, "%5d  %.7e", step, loss);
        for (int k = 0; k < (mtp ? 4 : 3); k++)
            length += std::snprintf(line + length, sizeof line - length, "  %.1e (%3.0f%%)", terms[k], 100 * terms[k] / loss);
        length += std::snprintf(line + length, sizeof line - length, "  %.2e  %6d", alpha, nalpha);
        if (mtp) length += std::isnan(linear) ? std::snprintf(line + length, sizeof line - length, "  %-13s", "-") : std::snprintf(line + length, sizeof line - length, "  %.7e", linear);
        std::snprintf(line + length, sizeof line - length, "  %s", *inv_hess ? inv_hess : "-");
        *log << line << std::endl;
    }

    // Logs the steps a phase took since start, and their time.
    void log_phase_time(int steps, std::chrono::steady_clock::time_point start) {
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        log_event(log, "time", "%.3g s @ %d steps, %.2g s / step", seconds, steps, seconds / std::max(steps, 1));
    }

    // Logs the time of the whole fit since start.
    void log_fit_time(std::chrono::steady_clock::time_point start) {
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        log_event(log, "time", "%.3g s for the fit", seconds);
    }

    // The training set's species count, over every rank, as mlip-3's AddSpecies checks it: species run from 0,
    // each present in the training set, and a trained potential's species are all present. Throws on every rank together.
    int training_species_count() {
        // the species count, and minus the lowest species when it is negative, so that every rank raises together
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

    // Hands the potential at coeffs to checkpoint, on the rank given one.
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

    std::vector<double> coeffs;           // what potential holds, in its coefficient order
    double loss_terms[3] = {};            // the energy, force and stress terms of the last loss_grad on this rank
    std::vector<double> dloss_dforces;    // per-structure scratch
};
