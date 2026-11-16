/* -*- c++ -*- ----------------------------------------------------------
   mlip-4's MTP for tests/test_mtp_basis.py --recapture.

   Reads a probe file (written by test_mtp_basis.py) holding a level, the
   species count, an RBCinf radial basis, the params and structures, and
   prints mlip-4's energy, forces and stress (CalcCfg) and each atom's site
   energy for the MTP of that level, unorthogonalized as mlip-4's Python MTP
   builds it by default.

   Build against a compiled mlip-4 (see tests/etn_potentials.py).
------------------------------------------------------------------------- */

#include "pch.h"
#include <pots/mtp/mtp.h>
#include <pots/mtp/mtp_levels.h>
#include <pots/radial_basis/cinf/rb_cinf.h>
#include <simulations/simulation.h>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>

static void print(const std::string& name, const std::vector<size_t>& shape, const std::vector<double>& values) {
    std::cout << name;
    for (size_t s : shape) std::cout << ' ' << s;
    std::cout << " :";
    char buffer[32];
    for (double x : values) {
        std::snprintf(buffer, sizeof buffer, " %.17g", x);
        std::cout << buffer;
    }
    std::cout << '\n';
}

// p_MTP's switch in mtp_pyinterface.cpp, with the basis unorthogonalized.
static MTP_Basis level_basis(int level) {
    const auto off = MTP_Basis::Orthogonalize::ORTH_OFF;
    switch (level) {
        case 6: return MTP_Basis(MTP6_array, off);
        case 8: return MTP_Basis(MTP8_array, off);
        case 10: return MTP_Basis(MTP10_array, off);
        case 12: return MTP_Basis(MTP12_array, off);
        case 14: return MTP_Basis(MTP14_array, off);
        case 16: return MTP_Basis(MTP16_array, off);
        case 18: return MTP_Basis(MTP18_array, off);
        case 20: return MTP_Basis(MTP20_array, off);
        case 22: return MTP_Basis(MTP22_array, off);
        case 24: return MTP_Basis(MTP24_array, off);
        case 26: return MTP_Basis(MTP26_array, off);
        case 28: return MTP_Basis(MTP28_array, off);
    }
    throw std::runtime_error("mlip4_mtp_probe: no level " + std::to_string(level));
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: mlip4_mtp_probe PROBE_FILE\n";
        return 2;
    }
    std::ifstream in(argv[1]);

    // level species_count radial_basis_size min_dist max_dist, then the params
    int level;
    size_t species_count, rb_size, param_count;
    double min_dist, max_dist;
    in >> level >> species_count >> rb_size >> min_dist >> max_dist >> param_count;
    std::vector<Species> species(species_count);
    for (size_t s = 0; s < species_count; s++) species[s] = (Species) s;

    auto basis = std::make_shared<RadialBasisCinf>(rb_size, min_dist, max_dist);
    auto mtp = std::make_shared<MTP>(level_basis(level), basis, SpeciesOrder(species), false);
    if (param_count != mtp->param_count()) {
        std::cerr << "mlip4_mtp_probe: " << param_count << " params given, mlip-4's MTP has " << mtp->param_count() << '\n';
        return 1;
    }
    for (size_t p = 0; p < param_count; p++) in >> mtp->params()[p];

    // Structures: n_atoms, cell rows, type x y z per atom
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
        Cfg cfg;
        cfg.FromString(json.str());

        CalcCfg calc(cfg, true, true);
        calc.AttachPot(mtp);
        calc.Run();
        const auto& stress = calc.get_stress();
        std::vector<double> forces(n_atoms * 3);
        for (size_t i = 0; i < n_atoms; i++)
            for (int a = 0; a < 3; a++) forces[i * 3 + a] = calc.get_forces()[i][a];

        std::vector<double> eatom(n_atoms, 0.0);
        ExtCfg ext(cfg, max_dist);
        for (const auto& nbh : ext.nbhs()) mtp->AccumulateSiteEnergy(nbh, eatom[nbh.my_ind()]);

        // mlip-4's stress is (xx, yy, zz, yz, xz, xy) of sum_i F_i (x) r_i, symmetrized
        const std::string prefix = "s" + std::to_string(s) + "/";
        print(prefix + "energy", {}, {calc.get_energy()});
        print(prefix + "eatom", {n_atoms}, eatom);
        print(prefix + "forces", {n_atoms, 3}, forces);
        print(prefix + "virial", {6}, {stress[0], stress[1], stress[2], stress[5], stress[4], stress[3]});
    }
    return 0;
}
