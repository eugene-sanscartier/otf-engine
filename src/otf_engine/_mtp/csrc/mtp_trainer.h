/* -*- c++ -*- ----------------------------------------------------------
   Fits an MTP to reference energies, forces and stresses.
   Algorithm of mlip-3's `mlp train` (mlip-3/src/mtpr_trainer.cpp,
   non_linear_regression.cpp, basic_trainer.h, and MLMTPR::AddPenaltyGrad and
   Orthogonalize in mtpr.cpp), on MTPTraining's kernels.
   Copyright (c) 2023, Alexander Shapeev (Skoltech). BSD 2-Clause, see LICENSE.mlip-3.

   The trainer is SPMD: every rank of a communicator holds a copy of the
   potential and a share of the structures, and the MPI collectives combine
   them. Built with MTP_MPI it uses <mpi.h>; otherwise thread_mpi.h, whose
   ranks are threads.
------------------------------------------------------------------------- */

#pragma once

#include "bfgs.h"
#include "mtp_training.h"

#ifdef MTP_MPI
#include <mpi.h>
#else
#include "thread_mpi.h"
#endif

#include <functional>
#include <ostream>
#include <vector>

// A structure to fit: its neighbor list at the potential's cutoff and its reference values.
struct TrainingStructure {
    std::vector<int> types, ilist, numneigh, firstneigh;
    std::vector<double> displacements;    // [sum(numneigh) * 3]

    bool has_energy = false, has_forces = false, has_virial = false;
    double energy = 0.0;
    std::vector<double> forces;    // [n_atoms * 3]
    double virial[6] = {};         // xx,yy,zz,xy,xz,yz in eV, mlip-3's PlusStress

    int n_atoms() const { return (int) types.size(); }
    NeighList list() const {
        return NeighList{n_atoms(), types.data(), (int) ilist.size(), ilist.data(), numneigh.data(), firstneigh.data(), displacements.data()};
    }
};

// `mlp train`'s options, under its names and with its defaults.
struct TrainerOptions {
    double energy_weight = 1.0;
    double force_weight = 0.01;
    double stress_weight = 0.001;
    double penalty_weight = 1e-6;
    double scale_by_force = 0.0;    // > 0 weighs each force by scale_by_force / (|F|^2 + scale_by_force)
    double select_factor = 1.0;     // scales every weight
    int weight_scaling = 1;         // energy weights are divided by n_atoms^weight_scaling
    int weight_scaling_forces = 0;  // force weights by n_atoms^weight_scaling_forces
    int iteration_limit = 1000;
    double tolerance = 1e-3;        // stop when 50 BFGS steps decrease the loss by less than this, relatively
    bool no_mindist_update = false; // keep min_dist rather than setting it to 0.99 of the shortest training distance
    bool init_random = false;       // randomize the radial coefficients of an untrained potential
    bool skip_preinit = false;      // skip the 75-step pre-training of an untrained potential
};

class MTPTrainer {
  public:
    // log receives mlp's fit log on the rank given one; checkpoint is called
    // on rank 0 wherever mlp saves the potential during the fit.
    MTPTrainer(MTPTraining& potential, std::vector<TrainingStructure> structures, const TrainerOptions& options, MPI_Comm comm, std::ostream* log = nullptr, std::function<void()> checkpoint = nullptr);

    // Leaves the fitted coefficients in the potential, on every rank, with the
    // species of the training set it lacked added.
    void train();

  private:
    void add_species();
    void update_min_dist();
    void fit_nonlinear(int max_iter);
    void fit_linear();
    void solve_linear(int structure_count);
    void rescale();

    // Loss and dL/dc over this rank's structures, without the penalty.
    double loss_grad(std::vector<double>& grad);
    // Adds the penalty on the radial functions' norms and overlaps and on the
    // linear coefficients' size.
    void add_penalty(double& loss, std::vector<double>& grad);
    void orthogonalize();

    double energy_weight(const TrainingStructure& s) const;
    double force_weight(const TrainingStructure& s, int i) const;
    double stress_weight(const TrainingStructure& s) const;

    MTPTraining& potential;
    std::vector<TrainingStructure> structures;
    TrainerOptions options;
    MPI_Comm comm;
    int rank = 0, size = 1;
    int structure_count = 0;    // over all ranks
    std::ostream* log;
    std::function<void()> checkpoint;

    int n_radial, n_linear;       // coefficients before the species block, and from it on
    std::vector<double> coeffs;   // [n_radial | species | linear], what potential holds

    // Linear least squares for the species and linear coefficients
    std::vector<double> lin_matrix, lin_vector;
    std::vector<double> reg_vector;    // regularization on its diagonal
    bool reg_init = true;              // whether reg_vector is rebuilt at the next solve
    const double reg_param = 1e-10;

    BFGS bfgs;

    // Per-structure scratch
    std::vector<double> forces, dloss_dforces;
    std::vector<double> site_energy_grad, force_grad, virial_grad;
};
