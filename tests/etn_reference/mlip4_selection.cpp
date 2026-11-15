/* -*- c++ -*- ----------------------------------------------------------
   mlip-4's active learning on an ETN, for check_etn_selection.py.

   Reads a selection file (written by check_etn_selection.py) holding the
   ETN's shape and params, the selection mode (0 cfg, 1 nbh), the training
   structures and the candidates; then, as mlip-4's CfgCalc does:
     - builds the active set on the training structures (UpdateActiveSet);
     - grades each candidate (Grade, and GradePerAtom in nbh mode);
     - selects among the candidates (UpdateActiveSet).
   Prints, on lines starting with "@ ":
     active_set N : structure equation, per row of A, after the training
     log_det : mlip-4's log |det A| in base n, after the training
     grade c : the grade of candidate c, then its per-atom grades in nbh mode
     selected K : the selected candidates' indices
     log_det_after : log |det A| in base n, after the selection

   Build against a compiled mlip-4 (see check_etn_kernels.py).
------------------------------------------------------------------------- */

#include "pch.h"
#include <active_learning/cfg_calculator.h>
#include <pots/etn/etn.h>
#include <pots/radial_basis/cinf/rb_cinf.h>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>

// CfgCalculator with its links, which say which structure and equation each row of A holds.
struct ProbeCalculator : CfgCalculator {
    using CfgCalculator::CfgCalculator;

    void print_links(const std::map<const Cfg*, int>& index) const {
        std::cout << "@ active_set " << eqn2cfg_links.size() << " :";
        for (const auto& link : eqn2cfg_links) {
            const int s = link.p_cfg ? index.at(link.p_cfg.get()) : -1;
            std::cout << ' ' << s << ' ' << (link.p_cfg ? link.eqn_ind : -1);
        }
        std::cout << '\n';
    }
};

static std::shared_ptr<const Cfg> read_cfg(std::istream& in) {
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

    auto cfg = std::make_shared<Cfg>();
    std::stringstream stream(json.str());
    json_io::Reader reader(stream);
    reader >> *cfg;
    return cfg;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: mlip4_selection SELECTION_FILE\n";
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
    auto etn = std::make_shared<ETN>(shape, ett_ranks, basis, SpeciesOrder(species), false);

    size_t param_count;
    in >> param_count;
    if (param_count != etn->param_count()) {
        std::cerr << "mlip4_selection: " << param_count << " params given, mlip-4's ETN has " << etn->param_count() << '\n';
        return 1;
    }
    for (size_t p = 0; p < param_count; p++) in >> etn->params()[p];

    int nbh;
    in >> nbh;
    std::map<const Cfg*, int> index;
    std::vector<std::shared_ptr<const Cfg>> training, candidates;
    size_t count;
    in >> count;
    for (size_t s = 0; s < count; s++) training.push_back(read_cfg(in));
    in >> count;
    for (size_t s = 0; s < count; s++) candidates.push_back(read_cfg(in));
    for (size_t s = 0; s < training.size(); s++) index[training[s].get()] = (int) s;
    for (size_t c = 0; c < candidates.size(); c++) index[candidates[c].get()] = (int) (training.size() + c);

    ProbeCalculator calc(etn, nbh != 0);
    calc.UpdateActiveSet(training);
    calc.print_links(index);
    char buffer[32];
    std::snprintf(buffer, sizeof buffer, "%.17g", calc.GetLogVol());
    std::cout << "@ log_det " << buffer << '\n';

    for (size_t c = 0; c < candidates.size(); c++) {
        std::snprintf(buffer, sizeof buffer, "%.17g", calc.Grade(*candidates[c]));
        std::cout << "@ grade " << c << " : " << buffer;
        if (nbh)
            for (double g : calc.GradePerAtom(*candidates[c])) {
                std::snprintf(buffer, sizeof buffer, " %.17g", g);
                std::cout << buffer;
            }
        std::cout << '\n';
    }

    const auto selected = calc.UpdateActiveSet(candidates);
    std::cout << "@ selected " << selected.size() << " :";
    for (const auto& cfg : selected) std::cout << ' ' << index.at(cfg.get()) - (int) training.size();
    std::cout << '\n';
    std::snprintf(buffer, sizeof buffer, "%.17g", calc.GetLogVol());
    std::cout << "@ log_det_after " << buffer << '\n';
    return 0;
}
