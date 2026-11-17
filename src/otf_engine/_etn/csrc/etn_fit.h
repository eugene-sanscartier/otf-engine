/* -*- c++ -*- ----------------------------------------------------------
   Fits an ETN to reference energies, forces and stresses.
   Algorithm of mlip-4's Train (mlip-4/src/training/train.cpp at 79a39bf)
   and ETN::Rescale (src/pots/etn/etn.cpp), on ETN's kernels.
   BSD 2-Clause, see mlip-4's LICENSE.

   SPMD over the ranks of a communicator, as every Fit is (see fit.h).
------------------------------------------------------------------------- */

#pragma once

#include "etn.h"
#include "fit.h"
#include "wolfe_bfgs.h"

#include <vector>

// mlip-4's TrainingParams with their defaults, under mlp's names where mlp has the option (iteration_limit,
// tolerance, skip_preinit). The loss doubles stress_weight on the virial's off-diagonal components, as mlip-4's
// LossCfg does.
struct ETNFitOptions : FitOptions {
    int iteration_limit = 1000;
    int max_func_eval = 10000;
    double tolerance = 5e-4;          // stop when 100 steps decrease the loss by less than this, relatively
    bool rescale = true;              // ETN::Rescale at the first step of each fit
    bool skip_preinit = false;        // skip the 75-step fit before the main one
    double wolfe_c1 = 0.1;
    double wolfe_c2 = 0.5;
    int bfgs_max_iters_no_improve = 10;

    // The options as the log lists them.
    std::string summary() const {
        std::ostringstream text;
        text << "iteration_limit " << iteration_limit << ", tolerance " << tolerance << ", max_func_eval " << max_func_eval << ", " << FitOptions::summary() << (rescale ? "" : ", no rescale") << (skip_preinit ? ", skip_preinit" : "");
        return text.str();
    }
};

class ETNFit : public Fit<ETN, ETNFitOptions> {
  public:
    // log receives the fit log on the rank given one; checkpoint is called
    // with the potential holding the coefficients at every step outside a line search.
    ETNFit(ETN& potential, std::vector<FitStructure> structures, const ETNFitOptions& options, Comm& comm, std::ostream* log = nullptr, std::function<void()> checkpoint = nullptr);

    // Leaves the fitted coefficients in the potential, on every rank, and logs
    // the errors on the training set before and after.
    void run();

    // On rank 0, the loss at each evaluation of each step, over every phase of run.
    const std::vector<std::vector<double>>& step_losses() const { return step_losses_; }

  private:
    // mlip-3's AddSpecies for an ETN: the potential gains the training set's species it lacks. An untrained one
    // takes mlip-4's initial values at the new count; a trained one gives each new species a site energy of 0,
    // and each new species pair of every A_l 1e-6 with 1e-3 (pair + 1) on its first channel, mlip-3's values for
    // a new MTP pair.
    void add_species();
    // mlip-4's Train without its pre-training: BFGS steps on the loss divided by the
    // structure count, from a rescale, until a stopping rule of o holds.
    void fit_nonlinear(const ETNFitOptions& o);
    // ETN::Rescale: a line search over alpha in coeffs = sign * alpha * coeffs, on the
    // loss summed over the structures.
    void rescale();
    // Loss and dL/dc over this rank's structures, at coeffs, with its terms in loss_terms.
    double loss_grad(std::vector<double>& grad);

    std::vector<std::vector<double>> step_losses_;
};
