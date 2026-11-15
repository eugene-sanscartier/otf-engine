/* -*- c++ -*- ----------------------------------------------------------
   Times mlip-4's ETN, for bench_etn.py.

   Reads a training file as mlip4_train does, with the ETN in mlip-4's VM or,
   given jit 1, in its JIT-compiled code, warms it up once (which compiles
   the JIT code), then prints, as "@ name value" lines, the mean time in ms
   of CalcCfg::Run (energy, forces and stress, the neighbor search
   included) per structure and of LossFunction::AccumulateGrads over every
   structure, each over `repeats` runs.

   usage: mlip4_bench TRAINING_FILE JIT REPEATS
------------------------------------------------------------------------- */

#include "pch.h"
#include <pots/etn/etn.h>
#include <pots/radial_basis/cinf/rb_cinf.h>
#include <simulations/simulation.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>

static double since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "usage: mlip4_bench TRAINING_FILE JIT REPEATS\n";
        return 2;
    }
    std::ifstream in(argv[1]);
    const bool jit = std::stoi(argv[2]) != 0;
    const int repeats = std::stoi(argv[3]);

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
    auto etn = std::make_shared<ETN>(shape, ett_ranks, basis, SpeciesOrder(species), jit);

    size_t param_count;
    in >> param_count;
    for (size_t p = 0; p < param_count; p++) in >> etn->params()[p];

    double weight_energy, weight_forces, weight_stress;
    in >> weight_energy >> weight_forces >> weight_stress;
    std::string settings;
    std::getline(in >> std::ws, settings);    // the training settings, unused here

    LossFunction loss_func;
    std::vector<CalcCfg> calcs;
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
        calcs.push_back(calc);
    }

    // Warm-up: prepares the programs and compiles the JIT code
    auto start = std::chrono::steady_clock::now();
    loss_func.AttachPot(etn);
    std::vector<double> grads(loss_func.grads_count());
    loss_func.AccumulateGrads(grads.data());
    for (auto& calc : calcs) {
        calc.AttachPot(etn);
        calc.Run();
    }
    std::cout << "@ warmup_ms " << since(start) << '\n';

    start = std::chrono::steady_clock::now();
    for (int r = 0; r < repeats; r++)
        for (auto& calc : calcs) calc.Run();
    std::cout << "@ eval_ms " << since(start) / (repeats * calcs.size()) << '\n';

    start = std::chrono::steady_clock::now();
    for (int r = 0; r < repeats; r++) {
        std::fill(grads.begin(), grads.end(), 0.0);
        loss_func.AccumulateGrads(grads.data());
    }
    std::cout << "@ loss_grad_ms " << since(start) / repeats << '\n';
    return 0;
}
