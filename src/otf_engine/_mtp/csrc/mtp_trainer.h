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
#include <string>
#include <vector>

// A structure to fit: its neighbor list at the potential's cutoff and its reference values.
struct TrainingStructure {
    std::vector<int> types, ilist, numneigh, firstneigh;
    std::vector<double> displacements;    // [sum(numneigh) * 3]

    bool has_energy = false, has_forces = false, has_virial = false;
    double energy = 0.0;
    std::vector<double> forces;    // [n_atoms * 3]
    double virial[6] = {};         // xx,yy,zz,xy,xz,yz in eV, mlip-3's PlusStress
    int number = 0;                // the caller's number for it, which the error table names
    std::vector<double> species_factors;    // force-weight factor by species index under balance_species

    int n_atoms() const { return (int) types.size(); }
    NeighList list() const {
        return NeighList{n_atoms(), types.data(), (int) ilist.size(), ilist.data(), numneigh.data(), firstneigh.data(), displacements.data()};
    }
};

// The errors of potential on the structures of every rank of comm, as a table under title, with a
// force row per species (named by species, or by index past its end) and the structures with the
// largest energy and force errors; on rank 0, and empty on the others.
std::string error_table(PairMTP& potential, const std::vector<TrainingStructure>& structures, const std::vector<std::string>& species, MPI_Comm comm, const std::string& title);

// `mlp train`'s options, under its names and with its defaults, and the trainer's own, whose defaults
// are the recipe; varpro, reg_once, no_orthogonalize, riemannian and reg_at_main false and init_noise 5
// give mlp's algorithm.
struct TrainerOptions {
    double energy_weight = 1.0;
    double force_weight = 0.01;
    double stress_weight = 0.001;
    double penalty_weight = 1e-6;
    double reg_param = 1e-10;       // the linear coefficients' regularization, times max(1, A_ii) / N once the first linear fit builds it
    double scale_by_force = 0.0;    // > 0 weighs each force by scale_by_force / (|F|^2 + scale_by_force)
    double select_factor = 1.0;     // scales every weight
    int weight_scaling = 1;         // energy weights are divided by n_atoms^weight_scaling
    int weight_scaling_forces = 0;  // force weights by n_atoms^weight_scaling_forces
    int iteration_limit = 1000;
    double tolerance = 1e-3;        // stop when 50 BFGS steps decrease the loss by less than this, relatively
    bool no_mindist_update = false; // keep min_dist rather than setting it to 0.99 of the shortest training distance
    bool init_random = false;       // randomize the radial coefficients of an untrained potential
    bool skip_preinit = false;      // skip the 75-step pre-training of an untrained potential
    bool bfgs_only = false;         // BFGS over every coefficient, no linear fit or rescale; a trained potential
    bool radial_only = false;       // BFGS over the radial coefficients only, species and linear ones kept, no linear fit or rescale; a trained potential
    bool varpro = true;             // variable projection: the radial coefficients stepped, the linear ones solved exactly at every evaluation with the regularization built once
    bool reg_once = true;           // build the linear regularization at the first linear fit only
    bool reg_at_main = true;        // rebuild the linear regularization at the main fit's first linear solve, under reg_once and varpro too
    bool no_orthogonalize = true;   // orthogonalize the radial functions in pre-training only, so a linear fit never raises the loss
    bool riemannian = true;         // the main fit by Riemannian L-BFGS with the radial functions orthonormal (W W^T = I), no radial penalty
    int lbfgs_memory = 1000;        // the pairs riemannian's L-BFGS keeps
    double init_noise = 0.5;        // init_random's noise on each radial coefficient, relative to the diagonal one's 1e-7 scale
    int random_seed = -1;           // init_random's seed; -1 draws one from std::random_device
    bool balance_species = true;    // weigh each force by N / (k n_s), so every species of a structure has an equal share of its force weight
    std::vector<std::string> species;    // the species' names in the error tables
};

class MTPTrainer {
  public:
    // log receives mlp's fit log on the rank given one; checkpoint is called
    // on rank 0 wherever mlp saves the potential during the fit.
    MTPTrainer(MTPTraining& potential, std::vector<TrainingStructure> structures, const TrainerOptions& options, MPI_Comm comm, std::ostream* log = nullptr, std::function<void()> checkpoint = nullptr);

    // Leaves the fitted coefficients in the potential, on every rank, with the
    // species of the training set it lacked added, and logs the errors on the
    // training set before and after.
    void train();

    // BFGS starts from the inverse Hessian in h, n x n for the n coefficients it
    // trains, and leaves its final inverse Hessian there. Every rank gives n;
    // only rank 0 gives h.
    void set_inverse_hessian(double* h, int n) { inverse_hessian = h; inverse_hessian_size = n; }

    // The loss BFGS minimizes, penalty included, and its gradient over every coefficient,
    // at coeffs, summed over the ranks, on every rank. For tests of the gradient.
    double objective(const std::vector<double>& at, std::vector<double>& grad);

  private:
    void add_species();
    void update_min_dist();
    void fit_nonlinear(int max_iter);
    void fit_riemannian(int max_iter);
    void fit_linear(bool orthonormalize = true);
    void assemble_linear();
    void solve_linear(int structure_count);
    void broadcast_linear();
    void rescale();

    // The loss at coeffs, penalty included (the radial one if radial), on rank 0, with its gradient over the first n
    // coefficients in grad and its energy, force, stress and penalty terms in terms, summed over the ranks onto rank 0.
    // Under varpro the linear coefficients are solved for coeffs first.
    double evaluate(std::vector<double>& grad, int n, bool radial, double terms[4]);
    // Loss and dL/dc over this rank's structures, without the penalty.
    double loss_grad(std::vector<double>& grad);
    // Adds the penalty on the radial functions' norms and overlaps, unless radial is false,
    // and on the linear coefficients' size.
    void add_penalty(double& loss, std::vector<double>& grad, bool radial = true);
    void orthogonalize();

    // The radial functions as the rows of W, K x (species pairs * basis size), in the radial block of a coefficient vector.
    // The row product a_k1 . b_k2 of the radial blocks of a and b.
    double row_dot(const double* a, int k1, const double* b, int k2) const;
    // Adds factor times row k2 of the radial block of x to row k1 of that of y.
    void row_axpy(double* y, int k1, double factor, const double* x, int k2) const;
    // Row products a_k . b_k' of the radial blocks of a and b, K x K.
    std::vector<double> radial_overlaps(const double* a, const double* b) const;
    // Projects the radial block of v onto the tangent space at W, the radial block of w: v - sym(v W^T) W.
    void project_tangent(const double* w, double* v) const;
    // QR retraction: Gram-Schmidt over the rows of the radial block of c, in order, a lower-triangular change of the radial functions.
    void retract(std::vector<double>& c) const;

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
    double loss_terms[3] = {};    // the energy, force and stress terms of the last loss_grad

    int n_radial, n_linear;       // coefficients before the species block, and from it on
    std::vector<double> coeffs;   // [n_radial | species | linear], what potential holds

    // Linear least squares for the species and linear coefficients
    std::vector<double> lin_matrix, lin_vector;
    std::vector<double> reg_vector;    // regularization on its diagonal
    bool reg_init = true;              // whether reg_vector is rebuilt at the next solve
    bool reg_built = false;            // whether a linear fit has built reg_vector
    bool orthogonalizing = true;       // whether orthogonalize acts; no_orthogonalize clears it after pre-training

    // Degree of each basis function in the radial functions, and whether the
    // linear system scales with it, so that rescale assembles it once.
    std::vector<int> scalar_degree;    // [alpha_scalar_count]
    bool scalable = true;

    BFGS bfgs;
    double* inverse_hessian = nullptr;    // where BFGS starts from and leaves its inverse Hessian, on rank 0
    int inverse_hessian_size = 0;         // its n, on every rank; 0 starts from the identity

    // Under varpro, each structure's energy, forces and virial as linear in the species and linear coefficients, the
    // rows assemble_linear last built (energy, 3 per atom, the 6 virial components), at the radial coefficients and
    // scaling in linear_rows_at; loss_grad takes the values from them at those coefficients in place of PairMTP::compute
    std::vector<std::vector<double>> linear_rows;    // per structure, [(1 + 3 n_atoms + 6) * n_linear]
    std::vector<double> linear_rows_at;              // [n_radial radial coefficients | scaling]

    // Per-structure scratch, and evaluate's gradient on this rank
    std::vector<double> dloss_dforces, local_grad, linear_values;
    std::vector<double> site_energy_grad, force_grad, virial_grad;
};
