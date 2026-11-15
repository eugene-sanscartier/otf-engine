/* -*- c++ -*- ----------------------------------------------------------
   mlip-4's Train on an ETN, for check_etn_training.py.

   Reads a training file (written by check_etn_training.py) holding the
   ETN's shape and params, the loss weights, mlip-4's training settings and
   the structures with their reference energy, virial (when given) and
   forces; builds
   mlip-4's LossFunction of LossCfg's on them and runs its Train. Its log,
   with every loss to 17 digits, goes to stdout as mlip-4 writes it, and the
   trained params follow on a line "@ params N : values". Given JIT 1, the
   ETN runs mlip-4's JIT-compiled code in place of its VM.

   Build against a compiled mlip-4 (see check_etn_training.py).
------------------------------------------------------------------------- */

#include "pch.h"
#include <pots/etn/etn.h>
#include <pots/radial_basis/cinf/rb_cinf.h>
#include <simulations/simulation.h>
#include <training/train.h>

#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

int main(int argc, char** argv) {
    if (argc != 2 && argc != 3) {
        std::cerr << "usage: mlip4_train TRAINING_FILE [JIT]\n";
        return 2;
    }
    std::ifstream in(argv[1]);

    // Shape: species_count l_count core_count radial_basis_size min_dist max_dist
    size_t species_count, l_count, core_count, rb_size;
    double min_dist, max_dist;
    in >> species_count >> l_count >> core_count >> rb_size >> min_dist >> max_dist;
    ETN::ETNShape shape;
    for (auto& row : shape) {
        row.resize(l_count);
        for (auto& x : row) in >> x;
    }
    std::vector<std::vector<size_t>> ett_ranks(core_count + 1);
    for (auto& ranks : ett_ranks) {
        size_t n;
        in >> n;
        ranks.resize(n);
        for (auto& x : ranks) in >> x;
    }
    std::vector<Species> species(species_count);
    for (size_t s = 0; s < species_count; s++) species[s] = (Species) s;
    auto basis = std::make_shared<RadialBasisCinf>(rb_size, min_dist, max_dist);
    auto etn = std::make_shared<ETN>(shape, ett_ranks, basis, SpeciesOrder(species), argc == 3 && std::stoi(argv[2]) != 0);

    size_t param_count;
    in >> param_count;
    if (param_count != etn->param_count()) {
        std::cerr << "mlip4_train: " << param_count << " params given, mlip-4's ETN has " << etn->param_count() << '\n';
        return 1;
    }
    for (size_t p = 0; p < param_count; p++) in >> etn->params()[p];

    // Loss weights, then TrainingParams: max_iter_count max_func_eval linstop rescale pretrain
    // wolfe_c1 wolfe_c2 bfgs_max_iters_no_improve
    double weight_energy, weight_forces, weight_stress;
    in >> weight_energy >> weight_forces >> weight_stress;
    int max_iter_count, max_func_eval, rescale, pretrain, no_improve;
    double linstop, wolfe_c1, wolfe_c2;
    in >> max_iter_count >> max_func_eval >> linstop >> rescale >> pretrain >> wolfe_c1 >> wolfe_c2 >> no_improve;
    TrainingParams params(max_iter_count, max_func_eval, linstop, rescale != 0, pretrain != 0, wolfe_c1, wolfe_c2, no_improve);

    // Structures: n_atoms, cell rows, type x y z per atom, then the reference energy, virial
    // (xx, yy, zz, xy, xz, yz) and forces
    LossFunction loss_func;
    size_t structure_count;
    in >> structure_count;
    for (size_t s = 0; s < structure_count; s++) {
        size_t n_atoms;
        in >> n_atoms;
        std::ostringstream json;
        json.precision(17);
        json << "{\"cell\":[";
        for (int a = 0; a < 3; a++) {
            double x, y, z;
            in >> x >> y >> z;
            json << (a ? "," : "") << '[' << x << ',' << y << ',' << z << ']';
        }
        json << "],\"pos\":[";
        std::vector<int> types(n_atoms);
        for (size_t i = 0; i < n_atoms; i++) {
            double x, y, z;
            in >> types[i] >> x >> y >> z;
            json << (i ? "," : "") << '[' << x << ',' << y << ',' << z << ']';
        }
        json << "],\"types\":[";
        for (size_t i = 0; i < n_atoms; i++) json << (i ? "," : "") << types[i];
        json << "]}";

        // The references through CalcCfg's JSON, the virial only when its flag is 1; mlip-4's
        // stress is (xx, yy, zz, yz, xz, xy)
        double energy, virial[6];
        int has_virial;
        in >> energy >> has_virial;
        if (has_virial)
            for (double& v : virial) in >> v;
        std::ostringstream calc_json;
        calc_json.precision(17);
        calc_json << "{\"cfg\":" << json.str() << ",\"energy\":" << energy;
        if (has_virial)
            calc_json << ",\"stress\":[" << virial[0] << ',' << virial[1] << ',' << virial[2] << ',' << virial[5] << ',' << virial[4] << ',' << virial[3] << ']';
        calc_json << ",\"forces\":[";
        for (size_t i = 0; i < n_atoms; i++) {
            double f[3];
            in >> f[0] >> f[1] >> f[2];
            calc_json << (i ? "," : "") << '[' << f[0] << ',' << f[1] << ',' << f[2] << ']';
        }
        calc_json << "]}";
        CalcCfg calc;
        std::stringstream calc_stream(calc_json.str());
        json_io::Reader reader(calc_stream);
        reader >> calc;

        LossCfg loss_cfg(calc, weight_energy, weight_forces, weight_stress);
        loss_func.AddLossTerm(loss_cfg);
    }

    io::log << std::setprecision(17);
    Train(loss_func, etn, params);
    io::log << '\n';

    std::cout << "@ params " << param_count << " :";
    char buffer[32];
    for (size_t p = 0; p < param_count; p++) {
        std::snprintf(buffer, sizeof buffer, " %.17g", etn->params()[p]);
        std::cout << buffer;
    }
    std::cout << '\n';
    return 0;
}
