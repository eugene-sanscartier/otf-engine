/* -*- c++ -*- ----------------------------------------------------------
   Fits an ETN to reference energies, forces and stresses.
   Algorithm of mlip-4's Train (mlip-4/src/training/train.cpp at 79a39bf)
   and ETN::Rescale (src/pots/etn/etn.cpp), on ETNTraining's kernels.
   BSD 2-Clause, see mlip-4's LICENSE.

   The trainer is SPMD, as MTPTrainer is: every rank of a communicator holds
   a copy of the potential and a share of the structures, and the MPI
   collectives combine them. Built with MTP_MPI it uses <mpi.h>; otherwise
   thread_mpi.h, whose ranks are threads.
------------------------------------------------------------------------- */

#pragma once

#include "error_table.h"
#include "etn_training.h"
#include "training_structure.h"
#include "wolfe_bfgs.h"

#ifdef MTP_MPI
#include <mpi.h>
#else
#include "thread_mpi.h"
#endif

#include <functional>
#include <ostream>
#include <string>
#include <vector>

// mlip-4's TrainingParams and LossCfg weights, under their names and with their defaults.
struct ETNTrainerOptions {
    double energy_weight = 1.0;
    double force_weight = 0.01;
    double stress_weight = 0.001;     // doubled on the virial's off-diagonal components
    int max_iter_count = 1000;
    int max_func_eval = 10000;
    double linstop = 5e-4;            // stop when 100 steps decrease the loss by less than this, relatively
    bool rescale = true;              // ETN::Rescale at the first step of each fit
    bool pretrain = true;             // a 75-step fit before the main one
    double wolfe_c1 = 0.1;
    double wolfe_c2 = 0.5;
    int bfgs_max_iters_no_improve = 10;
    bool balance_species = false;     // MTPTrainer's equal share of a structure's force weight per species; not mlip-4's
    std::vector<std::string> species;    // the species' names in the error tables
};

class ETNTrainer {
  public:
    // log receives mlip-4's fit log on the rank given one; checkpoint is called
    // with the potential holding the coefficients at every step outside a line search.
    ETNTrainer(ETNTraining& potential, std::vector<TrainingStructure> structures, const ETNTrainerOptions& options, MPI_Comm comm, std::ostream* log = nullptr, std::function<void()> checkpoint = nullptr);

    // Leaves the fitted coefficients in the potential, on every rank, and logs
    // the errors on the training set before and after.
    void train();

  private:
    // mlip-4's Train without its pre-training: BFGS steps on the loss divided by the
    // structure count, from a rescale, until a stopping rule of o holds.
    void fit(const ETNTrainerOptions& o);
    // ETN::Rescale: a line search over alpha in coeffs = sign * alpha * coeffs, on the
    // loss summed over the structures.
    void rescale();
    // Loss and dL/dc over this rank's structures, at coeffs.
    double loss_grad(std::vector<double>& grad);

    ETNTraining& potential;
    std::vector<TrainingStructure> structures;
    ETNTrainerOptions options;
    MPI_Comm comm;
    int rank = 0, size = 1;
    int structure_count = 0;    // over all ranks
    std::ostream* log;
    std::function<void()> checkpoint;

    std::vector<double> coeffs;    // what potential holds, in mlip-4's params order

    // Per-structure scratch
    std::vector<double> dloss_dforces;
};
