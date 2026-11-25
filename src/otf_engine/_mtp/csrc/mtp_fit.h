/* -*- c++ -*- ----------------------------------------------------------
   Fits an MTP to reference energies, forces and stresses.
   Algorithm of mlip-3's `mlp train` (mlip-3/src/mtpr_trainer.cpp,
   non_linear_regression.cpp, basic_trainer.h, and MLMTPR::AddPenaltyGrad and
   Orthogonalize in mtpr.cpp), on MTP's kernels.
   Copyright (c) 2023, Alexander Shapeev (Skoltech). BSD 2-Clause, see LICENSE.mlip-3.

   SPMD over the ranks of a communicator, as every Fit is (see fit.h).
------------------------------------------------------------------------- */

#pragma once

#include "bfgs.h"
#include "mtp.h"
#include "fit.h"

#include <string>
#include <utility>
#include <vector>

// Options of an MTP fit.
struct MTPFitOptions : FitOptions {
    double penalty_weight = 1e-6;
    double reg_param = 1e-10;       // regularization of the linear coefficients; a linear fit that builds it multiplies it by max(1, A_ii) / N
    double scale_by_force = 0.0;    // > 0 weighs each force by scale_by_force / (|F|^2 + scale_by_force)
    double select_factor = 1.0;     // scales all weights
    int weight_scaling = 1;         // energy weights are divided by n_atoms^weight_scaling
    int weight_scaling_forces = 0;  // force weights by n_atoms^weight_scaling_forces
    int iteration_limit = 1000;
    double tolerance = 1e-3;        // stop when 50 BFGS steps decrease the loss by a relative amount below this
    bool init_random = false;       // randomize the radial coefficients of an untrained potential
    bool skip_preinit = false;      // skip the 75-step pre-training of an untrained potential
    bool bfgs_only = false;         // BFGS over all coefficients of a trained potential, without linear fit or rescale
    bool radial_only = false;       // BFGS over the radial coefficients of a trained potential, without linear fit or rescale; the other coefficients stay fixed
    bool varpro = true;             // variable projection: BFGS steps the radial coefficients, and each evaluation solves the linear ones exactly with the regularization built once
    bool reg_once = true;           // build the linear regularization at the first linear fit only
    bool reg_at_main = true;        // rebuild the linear regularization at the first linear solve of the main fit, also under reg_once and varpro
    bool no_orthogonalize = true;   // orthogonalize the radial functions in pre-training only
    bool riemannian = true;         // run the main fit by Riemannian L-BFGS with orthonormal radial functions (W W^T = I) and no radial penalty
    bool ldlt = true;               // solve the linear system by LDLT with diagonal pivoting; false eliminates without row swaps
    bool reg_follows_scaling = true;    // rescale multiplies the regularization of a linear coefficient by the squared scaling factor of its basis function
    int lbfgs_memory = 1000;        // number of pairs the L-BFGS of riemannian keeps
    double init_noise = 0.5;        // noise of init_random on each radial coefficient, relative to the 1e-7 scale of the diagonal coefficient
    int random_seed = -1;           // seed of init_random; -1 draws one from std::random_device
    // Inverse Hessian [n x n] over the n coefficients BFGS trains, on rank 0. BFGS starts from it and writes its final one back.
    // Every rank gives n. n = 0 starts from the identity.
    double* inverse_hessian = nullptr;
    int inverse_hessian_size = 0;

    // Returns the options as log lines: stopping, loss weights, regularization and the flags set.
    std::vector<std::string> summary() const {
        std::ostringstream stopping, weights, regularization, algorithm;
        stopping << "iteration_limit " << iteration_limit << ", tolerance " << tolerance;
        weights << loss_weights();
        if (scale_by_force > 0) weights << ", scale_by_force " << scale_by_force;
        regularization << "penalty_weight " << penalty_weight << ", reg_param " << reg_param;
        const std::pair<const char*, bool> flags[] = {{"varpro", varpro}, {"riemannian", riemannian}, {"reg_once", reg_once}, {"reg_at_main", reg_at_main}, {"no_orthogonalize", no_orthogonalize}, {"ldlt", ldlt}, {"reg_follows_scaling", reg_follows_scaling}, {"init_random", init_random}, {"skip_preinit", skip_preinit}, {"bfgs_only", bfgs_only}, {"radial_only", radial_only}, {"no_mindist_update", no_mindist_update}};
        for (const auto& [name, set] : flags)
            if (set) algorithm << (algorithm.tellp() > 0 ? ", " : "") << name;
        if (varpro && varpro_fraction < 1.0) algorithm << (algorithm.tellp() > 0 ? ", " : "") << "varpro_fraction " << varpro_fraction;

        std::vector<std::string> lines = {stopping.str(), weights.str(), regularization.str()};
        if (algorithm.tellp() > 0) lines.push_back(algorithm.str());
        return lines;
    }
};

class MTPFit : public Fit<MTP, MTPFitOptions> {
  public:
    // log receives the fit log on the ranks that pass one. Rank 0 calls checkpoint to save the potential during the fit.
    MTPFit(MTP& potential, std::vector<FitStructure> structures, const MTPFitOptions& options, Comm& comm, std::ostream* log = nullptr, std::function<void()> checkpoint = nullptr);

    // Fits the potential and leaves its coefficients on all ranks. Adds the species of the training set the potential lacks.
    // Logs the training errors before and after.
    void run();

    // Returns the loss BFGS minimizes at `at`, penalty included, and writes its gradient over all coefficients into grad.
    // Both are summed over the ranks and returned on all ranks.
    double objective(const std::vector<double>& at, std::vector<double>& grad);

  private:
    friend class MTPFitStages;    // _mtp_testing runs the stages one at a time

    void add_species();
    void fit_nonlinear(int max_iter);
    void fit_riemannian(int max_iter);
    // Fits the linear coefficients over all structures, or over the varpro_fraction draw if subset.
    void fit_linear(bool subset = false);
    void assemble_linear(bool subset = false);
    void solve_linear();
    void broadcast_linear();
    void rescale();

    // Returns the loss at coeffs, penalty included, summed over the ranks onto rank 0. The penalty covers the radial functions if radial.
    // grad receives the gradient over the first n coefficients, and terms the energy, force, stress and penalty terms.
    // Under varpro with solve_linear, it first solves the linear coefficients over the varpro_fraction draw, and the gradient includes the path through the solve.
    double evaluate(std::vector<double>& grad, int n, bool radial, double terms[4], bool solve_linear = true);
    // Logs the loss at coeffs after a linear fit, in the form of the step lines, with the radial penalty if radial.
    void log_linear_fit(bool radial);
    // Returns the loss over the structures of this rank, without the penalty, and writes dL/dc into grad.
    double loss_grad(std::vector<double>& grad);
    // Adds to loss and grad the penalty on the size of the linear coefficients and, if radial, on the norms and overlaps of the radial functions.
    void add_penalty(double& loss, std::vector<double>& grad, bool radial = true);
    void orthogonalize();

    // The radial block of a coefficient vector holds the radial functions as the rows of W, K x (species pairs * basis size).
    // Returns the product a_k1 . b_k2 of rows of the radial blocks of a and b.
    double row_dot(const double* a, int k1, const double* b, int k2) const;
    // Adds factor times row k2 of the radial block of x to row k1 of that of y.
    void row_axpy(double* y, int k1, double factor, const double* x, int k2) const;
    // Returns the K x K row products a_k . b_k' of the radial blocks of a and b.
    std::vector<double> radial_overlaps(const double* a, const double* b) const;
    // Projects the radial block of v onto the tangent space at W, the radial block of w: v - sym(v W^T) W.
    void project_tangent(const double* w, double* v) const;
    // QR retraction: orthonormalizes the rows of the radial block of c in order by Gram-Schmidt, a lower-triangular change of the radial functions.
    void retract(std::vector<double>& c) const;

    double energy_weight(const FitStructure& s) const;
    // Returns the weight of the force on atom i. scale_by_force evaluates it at forces: the reference forces in the linear fit, the predicted ones in the loss.
    double force_weight(const FitStructure& s, int i, const double* forces) const;
    double stress_weight(const FitStructure& s) const;

    int n_radial, n_linear;       // coefficients before the species block, and from it on; coeffs is [radial | species | linear]

    // Linear least squares for the species and linear coefficients
    std::vector<double> lin_matrix, lin_vector;
    std::vector<double> reg_vector;    // regularization on the diagonal of lin_matrix
    std::vector<double> linear_system;    // under varpro_fraction, matrix of the last solve with its regularization, on rank 0
    bool reg_init = true;              // whether reg_vector is rebuilt at the next solve
    bool reg_built = false;            // whether a linear fit has built reg_vector
    bool orthogonalizing = true;       // whether orthogonalize acts; no_orthogonalize clears it after pre-training

    // Degree of each basis function in the radial functions, and whether the linear system scales with it.
    // If scalable, rescale assembles the system once.
    std::vector<int> scalar_degree;    // [alpha_scalar_count]
    bool scalable = true;

    BFGS bfgs;

    // Under varpro, linear_rows holds the rows assemble_linear last built: energy, forces and virial of each structure
    // as linear functions of the species and linear coefficients. A structure has 1 energy row, 3 rows per atom and
    // 6 virial rows, and none outside the varpro_fraction draw. linear_rows_at holds the radial coefficients and
    // scaling of the rows. At those coefficients, loss_grad takes the values from the rows and skips PairMTP::compute.
    std::vector<std::vector<double>> linear_rows;    // per structure, [(1 + 3 n_atoms + 6) * n_linear]
    std::vector<double> linear_rows_at;              // [n_radial radial coefficients | scaling]

    // Scratch per structure, and the gradient of evaluate on this rank
    std::vector<double> local_grad, linear_values;
    std::vector<double> site_energy_grad, force_grad, virial_grad;
};
