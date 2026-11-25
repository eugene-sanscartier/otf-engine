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

// Options of ETNFit, with the defaults of the recipe and force_weight 0.1.
// The loss doubles stress_weight on the off-diagonal virial components.
struct ETNFitOptions : FitOptions {
    int iteration_limit = 1000;
    int max_func_eval = 10000;
    double tolerance = 5e-4;          // stop when 100 steps decrease the loss by a relative amount below this
    bool rescale = true;              // rescale the coefficients at the first step of each fit
    bool skip_preinit = false;        // skip the 75-step fit before the main one
    double wolfe_c1 = 0.1;
    double wolfe_c2 = 0.5;
    int bfgs_max_iters_no_improve = 10;
    bool init_random = true;          // start an untrained potential from init_scale * sin(theta_k) with theta_k uniform, else from sin(10 + k^2)
    double init_scale = 0.3;
    int random_seed = -1;             // seed of init_random and varpro_fraction; -1 seeds init_random from std::random_device
    bool init_species = true;         // before training, fit the site energies by least squares to the reference energies minus those of the tensor train
    bool varpro = true;               // variable projection: solve the site energies and the last core exactly at every evaluation; disables rescale
    double reg_param = 1e-10;         // ridge of varpro on the linear coefficients, times max(1, A_ii) of the first solve
    bool reg_at_main = true;          // rebuild the ridge of varpro at the first solve of the main fit
    int varpro_redraw_every = 50;     // if > 0 under varpro_fraction, draw anew every so many steps of the main fit; the last draw holds over the last 20% of iteration_limit

    ETNFitOptions() { force_weight = 0.1; }

    // Returns the options as log lines: stopping, loss weights, and the algorithm flags that are set.
    std::vector<std::string> summary() const {
        std::ostringstream stopping, algorithm;
        stopping << "iteration_limit " << iteration_limit << ", tolerance " << tolerance << ", max_func_eval " << max_func_eval;
        if (!rescale) algorithm << "rescale false";
        if (skip_preinit) algorithm << (algorithm.tellp() > 0 ? ", " : "") << "skip_preinit";
        if (init_random) algorithm << (algorithm.tellp() > 0 ? ", " : "") << "init_random, init_scale " << init_scale << ", random_seed " << random_seed;
        if (init_species) algorithm << (algorithm.tellp() > 0 ? ", " : "") << "init_species";
        if (varpro) algorithm << (algorithm.tellp() > 0 ? ", " : "") << "varpro, reg_param " << reg_param << (reg_at_main ? ", reg_at_main" : "");
        if (varpro && varpro_fraction < 1.0) algorithm << ", varpro_fraction " << varpro_fraction;
        if (varpro && varpro_fraction < 1.0 && varpro_redraw_every > 0) algorithm << ", varpro_redraw_every " << varpro_redraw_every;
        if (no_mindist_update) algorithm << (algorithm.tellp() > 0 ? ", " : "") << "no_mindist_update";

        std::vector<std::string> lines = {stopping.str(), loss_weights()};
        if (algorithm.tellp() > 0) lines.push_back(algorithm.str());
        return lines;
    }
};

class ETNFit : public Fit<ETN, ETNFitOptions> {
  public:
    // log receives the fit log on the ranks that pass one.
    // checkpoint runs at every step outside a line search, with the current coefficients in the potential.
    ETNFit(ETN& potential, std::vector<FitStructure> structures, const ETNFitOptions& options, Comm& comm, std::ostream* log = nullptr, std::function<void()> checkpoint = nullptr);

    // Fits the potential and leaves the coefficients in it on every rank.
    // Logs the errors on the training set before and after the fit.
    void run();

    // Loss at each evaluation of each step, over all phases of run, on rank 0.
    const std::vector<std::vector<double>>& step_losses() const { return step_losses_; }

    // Returns the loss that BFGS minimizes at `at`, ridge included, and its gradient, zero on the linear coefficients of varpro.
    // Sums both over the ranks and returns them on every rank.
    double objective(const std::vector<double>& at, std::vector<double>& grad);

  private:
    // Adds to the potential the species of the training set that it lacks.
    // An untrained potential restarts from reset_coeffs at the new species count.
    // A trained potential gives each new species a site energy of 0, and each new species pair of each A_l
    // the value 1e-6, with 1e-3 (pair + 1) on its first channel.
    void add_species();
    // Sets the site energies to the least-squares fit, over the species counts, of the reference energies minus those of the tensor train.
    // The normal matrix gets a ridge of 1e-12 times its largest diagonal entry.
    void fit_species_energies();
    // Runs BFGS on the loss divided by the structure count, after a rescale, until a stopping rule of o holds.
    // Under varpro, the gradient on the linear coefficients is zero, and solve_linear sets them before each evaluation.
    void fit_nonlinear(const ETNFitOptions& o);
    // Line-searches alpha in coeffs = sign * alpha * coeffs on the loss summed over the structures.
    void rescale();
    // Returns the loss over the structures of this rank at coeffs, with its gradient in grad and its terms in loss_terms.
    double loss_grad(std::vector<double>& grad);
    // Solves the linear coefficients of varpro, then runs loss_grad.
    // Under varpro_fraction < 1, adds the path of the gradient through the solve. Collective.
    double reduced_loss_grad(std::vector<double>& grad);
    // Loss weights of the energy of s, of its force component i, and of its virial component k (xx yy zz xy xz yz).
    double energy_weight(const FitStructure& s) const;
    double force_weight(const FitStructure& s, int i) const;
    double virial_weight(const FitStructure& s, int k) const;
    // varpro: sets the site energies and the last-core coefficients in coeffs to the minimum of the loss plus the ridge,
    // with the other coefficients fixed. Solves over the varpro_fraction structures if subset, else over all.
    // Builds the ridge when it is empty, and keeps the columns of each structure in linear_rows.
    void solve_linear(bool subset = false);
    // varpro: the columns of the linear coefficients, those of the last core and then the site energies
    std::vector<int> linear_columns() const;
    // Coefficient blocks in the get_coeffs order, as [begin, end): B_l per l, A_l per l, the real part of each core,
    // the imaginary part of each core, the site energies
    std::vector<std::pair<int, int>> blocks() const;
    std::pair<int, int> core_block(int i, bool imaginary) const;

    std::vector<std::vector<double>> step_losses_;
    std::vector<double> ridge;    // ridge of varpro per linear coefficient, empty until the first solve builds it
    // varpro: outputs (energy, virial, forces) of each structure per linear column, [columns * (7 + 3 n_atoms)], from the last solve.
    // A structure outside the solve has none. linear_rows_at holds the coeffs they were built at, with the linear ones 0.
    std::vector<std::vector<double>> linear_rows;
    std::vector<double> linear_rows_at;
    std::vector<double> linear_system;    // varpro: matrix of the last solve, ridge included, [columns * columns]
    std::vector<double> row_outputs;    // outputs of one structure in loss_grad, built from its columns
};
