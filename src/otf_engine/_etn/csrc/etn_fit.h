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
// tolerance, skip_preinit), and the fit's own, whose defaults and force_weight 0.1 are the recipe; train.MLIP4_ALGORITHM
// gives mlip-4's algorithm. The loss doubles stress_weight on the virial's off-diagonal components, as mlip-4's LossCfg
// does.
struct ETNFitOptions : FitOptions {
    int iteration_limit = 1000;
    int max_func_eval = 10000;
    double tolerance = 5e-4;          // stop when 100 steps decrease the loss by less than this, relatively
    bool rescale = true;              // ETN::Rescale at the first step of each fit
    bool skip_preinit = false;        // skip the 75-step fit before the main one
    double wolfe_c1 = 0.1;
    double wolfe_c2 = 0.5;
    int bfgs_max_iters_no_improve = 10;
    bool init_random = true;          // start an untrained potential from init_scale * sin(theta_k), theta_k uniform, in place of sin(10 + k^2)
    double init_scale = 0.3;
    int random_seed = -1;             // init_random's seed and varpro_fraction's; -1 draws init_random's from std::random_device
    bool init_species = true;         // before training, set the site energies to the least-squares fit of the energies the tensor train leaves
    bool varpro = true;               // variable projection: the site energies and the last core solved exactly at every evaluation, no rescale
    double reg_param = 1e-10;         // varpro's ridge on the linear coefficients, times max(1, A_ii) of the first solve
    bool reg_at_main = true;          // rebuild varpro's ridge at the main fit's first solve
    int varpro_redraw_every = 0;      // > 0, under varpro_fraction: in the main fit, a new draw every so many steps, the last one kept over the last 20% of iteration_limit

    ETNFitOptions() { force_weight = 0.1; }

    // The options as the log lists them, one line per group: stopping, loss weights, the algorithm's flags set.
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

        std::vector<std::string> lines = {stopping.str(), loss_weights()};
        if (algorithm.tellp() > 0) lines.push_back(algorithm.str());
        return lines;
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

    // The loss BFGS minimizes at `at`, with varpro's ridge, and its gradient, 0 on varpro's linear coefficients; summed
    // over the ranks, on every rank. For tests of the gradient.
    double objective(const std::vector<double>& at, std::vector<double>& grad);

  private:
    // mlip-3's AddSpecies for an ETN: the potential gains the training set's species it lacks. An untrained one
    // takes mlip-4's initial values at the new count; a trained one gives each new species a site energy of 0,
    // and each new species pair of every A_l 1e-6 with 1e-3 (pair + 1) on its first channel, mlip-3's values for
    // a new MTP pair.
    void add_species();
    // init_species: the site energies that fit, by least squares over the species counts, the reference energies
    // less the tensor train's, with a ridge of 1e-12 times the largest diagonal entry for collinear compositions.
    void fit_species_energies();
    // mlip-4's Train without its pre-training: BFGS steps on the loss divided by the
    // structure count, from a rescale, until a stopping rule of o holds. Under varpro, the linear coefficients'
    // gradient is zeroed, so BFGS keeps them out of its steps, and solve_linear sets them before each evaluation.
    void fit_nonlinear(const ETNFitOptions& o);
    // ETN::Rescale: a line search over alpha in coeffs = sign * alpha * coeffs, on the
    // loss summed over the structures.
    void rescale();
    // Loss and dL/dc over this rank's structures, at coeffs, with its terms in loss_terms.
    double loss_grad(std::vector<double>& grad);
    // loss_grad at coeffs with varpro's linear coefficients solved first, and under varpro_fraction < 1 the
    // gradient's path through the solve added; collective.
    double reduced_loss_grad(std::vector<double>& grad);
    // The loss's weight of s's energy, of its force component i, and of its virial component k (xx yy zz xy xz yz).
    double energy_weight(const FitStructure& s) const;
    double force_weight(const FitStructure& s, int i) const;
    double virial_weight(const FitStructure& s, int k) const;
    // varpro: sets the site energies and the last core's coefficients in coeffs to the minimum of the loss plus
    // the ridge, which the first call builds when it is empty, the other coefficients fixed, over varpro_fraction's
    // structures if subset, else all; keeps each structure's columns in linear_rows.
    void solve_linear(bool subset = false);
    // varpro: the linear coefficients' columns, the last core's then the site energies'
    std::vector<int> linear_columns() const;
    // The coefficient blocks in get_coeffs' order, as [begin, end): each B_l, each A_l, each core's real part,
    // each core's imaginary part, the site energies
    std::vector<std::pair<int, int>> blocks() const;
    std::pair<int, int> core_block(int i, bool imaginary) const;

    std::vector<std::vector<double>> step_losses_;
    std::vector<double> ridge;    // varpro's ridge per linear coefficient, empty until the first solve builds it
    // varpro: each structure's outputs (energy, virial, forces) per linear column, [columns * (7 + 3 n_atoms)], as
    // the last solve built them (empty for a structure outside it), at linear_rows_at, coeffs with the linear ones 0
    std::vector<std::vector<double>> linear_rows;
    std::vector<double> linear_rows_at;
    std::vector<double> linear_system;    // varpro: the last solve's matrix, ridge included, [columns * columns]
    std::vector<double> row_outputs;    // loss_grad's outputs of one structure from its columns
};
