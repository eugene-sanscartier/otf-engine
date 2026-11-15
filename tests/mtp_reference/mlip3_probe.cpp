/* -*- c++ -*- ----------------------------------------------------------
   mlip-3's training stages, one at a time, for tests/test_mtp_trainer.py.

       mlip3_probe POTENTIAL CFG SCRIPT
       mlip3_probe --rosenbrock N STEPS

   The first form loads POTENTIAL and the structures of CFG and runs the
   lines of SCRIPT in order:

       set KEY VALUE        a setting of MTPR_trainer, before `trainer`
       trainer              builds MTPR_trainer from the settings
       add_species          AddSpecies, which also moves min_dist
       orthogonalize        MLMTPR::Orthogonalize
       lin_optimize         LinOptimize
       rescale              Rescale
       objective            the loss and gradient BFGS takes: CalcObjectiveFunctionGrad over K structures
       coeffs N V...        sets the coefficients
       scaling S            sets the scaling
       reg_vector N V...    sets the regularization vector
       print                the coefficients, the scaling, the radial basis' min_dist and the regularization vector

   Each result goes to stdout as "@ NAME N : V..." to 17 digits. The second
   form runs mlip-3's BFGS on the N-dimensional Rosenbrock function from
   x_i = -1.2 (odd i) and 1, printing x, f and whether it is in a line search
   at each of STEPS calls to Iterate.

   Compiled with mlip-3's serial sources (see tests/conftest.py).
------------------------------------------------------------------------- */

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#define private public
#define protected public
#include "mtpr.h"
#include "mtpr_trainer.h"
#include "common/bfgs.h"
#undef private
#undef protected

static void put(const std::string& name, const double* v, int n) {
    std::printf("@ %s %d :", name.c_str(), n);
    for (int i = 0; i < n; i++) std::printf(" %.17g", v[i]);
    std::printf("\n");
}

// The N-dimensional Rosenbrock function and its gradient; _mtp_testing computes it with the same operations.
static double rosenbrock(const std::vector<double>& x, std::vector<double>& g) {
    const int n = (int) x.size();
    double f = 0.0;
    g.assign(n, 0.0);
    for (int i = 0; i + 1 < n; i++) {
        const double a = x[i + 1] - x[i] * x[i];
        const double b = 1.0 - x[i];
        f += 100.0 * a * a + b * b;
        g[i] += -400.0 * a * x[i] - 2.0 * b;
        g[i + 1] += 200.0 * a;
    }
    return f;
}

static int run_rosenbrock(int n, int steps) {
    std::vector<double> x(n), g;
    for (int i = 0; i < n; i++) x[i] = i % 2 == 0 ? -1.2 : 1.0;
    BFGS bfgs;
    bfgs.Set_x(x.data(), n);
    for (int step = 0; step < steps; step++) {
        for (int i = 0; i < n; i++) x[i] = bfgs.x(i);
        const double f = rosenbrock(x, g);
        Array1D grad(g.begin(), g.end());
        const bool decreased = bfgs.Iterate(f, grad);
        const double state[3] = {f, (double) bfgs.is_in_linesearch(), (double) decreased};
        put("x", x.data(), n);
        put("state", state, 3);
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc == 4 && std::string(argv[1]) == "--rosenbrock") return run_rosenbrock(std::stoi(argv[2]), std::stoi(argv[3]));
    if (argc != 4) {
        std::cerr << "usage: mlip3_probe POTENTIAL CFG SCRIPT | --rosenbrock N STEPS\n";
        return 2;
    }

    MLMTPR mtp(argv[1]);
    std::vector<Configuration> structures = LoadCfgs(argv[2]);
    Settings settings;
    MTPR_trainer* trainer = nullptr;

    std::ifstream script(argv[3]);
    for (std::string line; std::getline(script, line);) {
        std::istringstream words(line);
        std::string command;
        if (!(words >> command)) continue;

        if (command == "set") {
            std::string key, value;
            words >> key >> value;
            settings[key] = value;
        } else if (command == "trainer")
            trainer = new MTPR_trainer(&mtp, settings);
        else if (command == "add_species")
            trainer->AddSpecies(structures);
        else if (command == "orthogonalize")
            mtp.Orthogonalize();
        else if (command == "lin_optimize")
            trainer->LinOptimize(structures);
        else if (command == "rescale")
            trainer->Rescale(structures);
        else if (command == "objective") {
            trainer->CalcObjectiveFunctionGrad(structures);
            const double loss = trainer->loss_ / structures.size();
            std::vector<double> grad(trainer->loss_grad_.begin(), trainer->loss_grad_.end());
            for (double& g : grad) g /= structures.size();
            put("loss", &loss, 1);
            put("grad", grad.data(), (int) grad.size());
        } else if (command == "coeffs") {
            int n;
            words >> n;
            mtp.regression_coeffs.resize(n);
            for (int i = 0; i < n; i++) words >> mtp.regression_coeffs[i];
        } else if (command == "scaling")
            words >> mtp.scaling;
        else if (command == "reg_vector") {
            int n;
            words >> n;
            mtp.reg_vector.resize(n);
            for (int i = 0; i < n; i++) words >> mtp.reg_vector[i];
        } else if (command == "print") {
            put("coeffs", mtp.Coeff(), mtp.CoeffCount());
            put("scaling", &mtp.scaling, 1);
            put("min_dist", &mtp.p_RadialBasis->min_val, 1);
            put("reg_vector", mtp.reg_vector.data(), (int) mtp.reg_vector.size());
        } else {
            std::cerr << "mlip3_probe: unknown command " << command << '\n';
            return 2;
        }
    }
    delete trainer;
    return 0;
}
