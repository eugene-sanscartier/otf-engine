/* -*- c++ -*- ----------------------------------------------------------
   Fits an ETN to reference energies, forces and stresses.
   Algorithm of mlip-4's Train (mlip-4/src/training/train.cpp at 79a39bf)
   and ETN::Rescale (src/pots/etn/etn.cpp), on ETNTraining's kernels.
   BSD 2-Clause, see mlip-4's LICENSE.

   SPMD over the ranks of a communicator, as every Trainer is (see trainer.h).
------------------------------------------------------------------------- */

#pragma once

#include "etn_training.h"
#include "trainer.h"
#include "wolfe_bfgs.h"

#include <vector>

// mlip-4's TrainingParams with their defaults, under mlp's names where mlp has the option (iteration_limit,
// tolerance, skip_preinit). The loss doubles stress_weight on the virial's off-diagonal components, as mlip-4's
// LossCfg does.
struct ETNTrainerOptions : TrainerOptions {
    int iteration_limit = 1000;
    int max_func_eval = 10000;
    double tolerance = 5e-4;          // stop when 100 steps decrease the loss by less than this, relatively
    bool rescale = true;              // ETN::Rescale at the first step of each fit
    bool skip_preinit = false;        // skip the 75-step fit before the main one
    double wolfe_c1 = 0.1;
    double wolfe_c2 = 0.5;
    int bfgs_max_iters_no_improve = 10;
};

class ETNTrainer : public Trainer<ETNTraining, ETNTrainerOptions> {
  public:
    // log receives mlip-4's fit log on the rank given one; checkpoint is called
    // with the potential holding the coefficients at every step outside a line search.
    ETNTrainer(ETNTraining& potential, std::vector<TrainingStructure> structures, const ETNTrainerOptions& options, MPI_Comm comm, std::ostream* log = nullptr, std::function<void()> checkpoint = nullptr);

    // Leaves the fitted coefficients in the potential, on every rank, and logs
    // the errors on the training set before and after.
    void train();

  private:
    // mlip-3's AddSpecies for an ETN: the potential gains the training set's species it lacks. An untrained one
    // takes mlip-4's initial values at the new count; a trained one gives each new species a site energy of 0,
    // and each new species pair of every A_l 1e-6 with 1e-3 (pair + 1) on its first channel, mlip-3's values for
    // a new MTP pair.
    void add_species();
    // mlip-4's Train without its pre-training: BFGS steps on the loss divided by the
    // structure count, from a rescale, until a stopping rule of o holds.
    void fit(const ETNTrainerOptions& o);
    // ETN::Rescale: a line search over alpha in coeffs = sign * alpha * coeffs, on the
    // loss summed over the structures.
    void rescale();
    // Loss and dL/dc over this rank's structures, at coeffs.
    double loss_grad(std::vector<double>& grad);
};
