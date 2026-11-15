/* -*- c++ -*- ----------------------------------------------------------
   mlip-4's ETN tensors for check_etn_kernels.py --capture.

   Reads a probe file (written by check_etn_kernels.py) holding the ETN's
   shape and params, probe distances and structures, and prints every tensor
   mlip-4 computes for them: the radial basis, the real 3-j tables, and per
   atom the input vector, the bond vectors and the site energy, then forces
   and virial. The input and bond vectors come from mlip-4's own program,
   recomposed by ETNProbe below as ETN::ComposeETTProgram composes it, with
   the bond vectors kept as outputs.

   Build against a compiled mlip-4 (see check_etn_kernels.py).
------------------------------------------------------------------------- */

#include "pch.h"
#include <pots/etn/etn.h>
#include <pots/radial_basis/cinf/rb_cinf.h>
#include <variables/var_equiv_tensor.h>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>

class ETNProbe : public ETN {
  public:
    using ETN::ETN;

    std::shared_ptr<VmTwoLevelExecutable> exe_tensors;
    std::vector<Variable> bond_re, bond_im;  // [core_count + 1]; empty Variables where absent
    std::vector<Variable> in_vars;

    double* param_data() { return &params_[0]; }
    size_t input_size() { return var_ett_input_gvector.n_elem(); }
    AnyTwoLevelExecutable& energy_ders_executable() { return *exe_energy_ders; }

    // ETN::PrepareForCalc's program, composed once, with ComposeETTProgram's bond vectors kept.
    void build() {
        AnyPairDescriptorPot::PrepareForCalc();
        std::vector<Variable> radial_basis_vars, vars_angular;
        ComposeBasisProgram(radial_basis_vars, vars_angular);
        ComposeFeatureProgram(radial_basis_vars, vars_angular, var_nbhr_species, var_ett_input_gvector);
        compose_ett();
        MakeExecutables();

        in_vars.clear();
        in_vars.push_back(var_radial_scaling);
        in_vars.insert(in_vars.end(), vars_params_radial.begin(), vars_params_radial.end());
        in_vars.insert(in_vars.end(), vars_params_species.begin(), vars_params_species.end());
        in_vars.insert(in_vars.end(), vars_params_ett.begin(), vars_params_ett.end());
        in_vars.push_back(var_params_self_species);
        in_vars.push_back(var_self_species);
        in_vars.push_back(var_pair_x);
        in_vars.push_back(var_pair_y);
        in_vars.push_back(var_pair_z);
        in_vars.push_back(var_nbhr_species);

        std::vector<Variable> outs = {var_ett_input_gvector, var_energy};
        for (auto& v : bond_re) if (!v.empty()) outs.push_back(v);
        for (auto& v : bond_im) if (!v.empty()) outs.push_back(v);
        exe_tensors = std::make_shared<VmTwoLevelExecutable>();
        exe_tensors->Construct(in_vars, outs);
    }

    // ETN::ComposeETTProgram, verbatim but for keeping u_re and u_im in members.
    void compose_ett() {
        size_t n_cores = ett_ranks.size() - 1;
        std::vector<Variable> vars_u_re(n_cores + 1);
        std::vector<Variable> vars_T_re(n_cores);
        std::vector<Variable> vars_S_re(n_cores);

        vars_u_re[0] = Variable::New<VarTensor>("u_re_0", {1}, VarFlags::AGGREGATE);
        Variable var_const_0 = Variable::New<VarFloat>("zero", VarFlags::AGGREGATE);
        Variable var_const_1 = Variable::New<VarFloat>("one", VarFlags::AGGREGATE);
        var_const_1 += 1.;
        vars_u_re[0] += MakeOp<OpToArray>(var_const_1, var_const_0);

        Variable& v = var_ett_input_gvector;
        v += MakeOp<OpToArray>(var_const_1, var_const_0);
        std::vector<size_t> shape_ett_input_gvector = etn_shape[2];
        shape_ett_input_gvector[0] += 1;

        std::vector<Variable> vars_u_im(n_cores + 1);
        if ((n_cores <= 3) || (wigner_im.vals.empty())) {
            for (size_t i = 0; i < n_cores; ++i) {
                std::string i_str = std::to_string(i);
                Variable& c = vars_params_ett[i];
                Variable& u = vars_u_re[i];
                Variable& w = vars_u_re[i + 1];
                Variable& T = vars_T_re[i];
                Variable& S = vars_S_re[i];
                EquivIndex shape_T({{ett_ranks[i], shape_ett_input_gvector, ett_ranks[i + 1]}}, true);
                T = Variable::New<VarEquivTensor>("T_re_" + i_str, shape_T, &select_wigner_rsph_re, !sparse, VarFlags::AGGREGATE);
                w = Variable::New<VarTensor>("u_re_" + std::to_string(i + 1), {T.shape()[2]}, VarFlags::AGGREGATE);
                S = Variable::New<VarTensor>("S_re_" + i_str, {T.shape()[1], T.shape()[2]}, VarFlags::AGGREGATE);
                T += MakeOp<OpConstructEquivTensor>(c, wigner_re, !sparse);
                S += multiply(u, T, {{0, 0}});
                w += multiply(v, S, {{0, 0}});
            }
        } else {
            std::vector<Variable>& u_re = vars_u_re;
            std::vector<Variable>& u_im = vars_u_im;
            std::vector<Variable>& T_re = vars_T_re;
            std::vector<Variable> T_im(n_cores);
            std::vector<Variable>& S_re = vars_S_re;
            std::vector<Variable> S_im(n_cores);
            std::vector<Variable>& c = vars_params_ett;

            {
                EquivIndex shape_T({{ett_ranks[0], shape_ett_input_gvector, ett_ranks[1]}}, true);
                T_re[0] = Variable::New<VarEquivTensor>("T_re_0", shape_T, &select_wigner_rsph_re, !sparse, VarFlags::AGGREGATE);
                u_re[1] = Variable::New<VarTensor>("u_re_1", {T_re[0].shape()[2]}, VarFlags::AGGREGATE);
                S_re[0] = Variable::New<VarTensor>("S_re_0", {T_re[0].shape()[1], T_re[0].shape()[2]}, VarFlags::AGGREGATE);
            }
            for (size_t i = 1; i < n_cores - 1; ++i) {
                std::string i_str = std::to_string(i);
                std::string i_plus_one_str = std::to_string(i + 1);
                EquivIndex shape_T({{ett_ranks[i], shape_ett_input_gvector, ett_ranks[i + 1]}}, true);
                T_re[i] = Variable::New<VarEquivTensor>("T_re_" + i_str, shape_T, &select_wigner_rsph_re, !sparse, VarFlags::AGGREGATE);
                T_im[i] = Variable::New<VarEquivTensor>("T_im_" + i_str, shape_T, &select_wigner_rsph_im, !sparse, VarFlags::AGGREGATE);
                u_re[i + 1] = Variable::New<VarTensor>("u_re_" + i_plus_one_str, {T_re[i].shape()[2]}, VarFlags::AGGREGATE);
                u_im[i + 1] = Variable::New<VarTensor>("u_im_" + i_plus_one_str, {T_im[i].shape()[2]}, VarFlags::AGGREGATE);
                S_re[i] = Variable::New<VarTensor>("S_re_" + i_str, {T_re[i].shape()[1], T_re[i].shape()[2]}, VarFlags::AGGREGATE);
                S_im[i] = Variable::New<VarTensor>("S_im_" + i_str, {T_im[i].shape()[1], T_im[i].shape()[2]}, VarFlags::AGGREGATE);
            }
            {
                EquivIndex shape_T({{ett_ranks[n_cores - 1], shape_ett_input_gvector, ett_ranks[n_cores]}}, true);
                T_re[n_cores - 1] = Variable::New<VarEquivTensor>("T_re_" + std::to_string(n_cores - 1), shape_T, &select_wigner_rsph_re, !sparse, VarFlags::AGGREGATE);
                u_re[n_cores] = Variable::New<VarTensor>("u_re_" + std::to_string(n_cores), {T_re[n_cores - 1].shape()[2]}, VarFlags::AGGREGATE);
                S_re[n_cores - 1] = Variable::New<VarTensor>("S_re_" + std::to_string(n_cores - 1), {T_re[n_cores - 1].shape()[1], T_re[n_cores - 1].shape()[2]}, VarFlags::AGGREGATE);
            }

            T_re[0] += MakeOp<OpConstructEquivTensor>(c[0], wigner_re, !sparse);
            S_re[0] += multiply(u_re[0], T_re[0], {{0, 0}});
            u_re[1] += multiply(v, S_re[0], {{0, 0}});

            T_re[1] += MakeOp<OpConstructEquivTensor>(c[1], wigner_re, !sparse);
            T_im[1] += MakeOp<OpConstructEquivTensor>(c[n_cores], wigner_im, !sparse);
            S_re[1] += multiply(u_re[1], T_re[1], {{0, 0}});
            u_re[2] += multiply(v, S_re[1], {{0, 0}});
            S_im[1] += multiply(u_re[1], T_im[1], {{0, 0}});
            u_im[2] += multiply(v, S_im[1], {{0, 0}});

            for (size_t i = 2; i < n_cores - 2; ++i) {
                T_re[i] += MakeOp<OpConstructEquivTensor>(c[i], wigner_re, !sparse);
                T_im[i] += MakeOp<OpConstructEquivTensor>(c[n_cores + i - 1], wigner_im, !sparse);
                S_re[i] += multiply(u_re[i], T_re[i], {{0, 0}});
                S_re[i] += multiply(u_im[i], T_im[i], {{0, 0}});
                u_re[i + 1] += multiply(v, S_re[i], {{0, 0}});
                S_im[i] += multiply(u_im[i], T_re[i], {{0, 0}});
                S_im[i] += multiply(u_re[i], T_im[i], {{0, 0}});
                u_im[i + 1] += multiply(v, S_im[i], {{0, 0}});
            }

            T_re[n_cores - 2] += MakeOp<OpConstructEquivTensor>(c[n_cores - 2], wigner_re, !sparse);
            T_im[n_cores - 2] += MakeOp<OpConstructEquivTensor>(c[2 * n_cores - 3], wigner_im, !sparse);
            S_re[n_cores - 2] += multiply(u_re[n_cores - 2], T_re[n_cores - 2], {{0, 0}});
            S_re[n_cores - 2] += multiply(u_im[n_cores - 2], T_im[n_cores - 2], {{0, 0}});
            u_re[n_cores - 1] += multiply(v, S_re[n_cores - 2], {{0, 0}});

            T_re[n_cores - 1] += MakeOp<OpConstructEquivTensor>(c[n_cores - 1], wigner_re, !sparse);
            S_re[n_cores - 1] += multiply(u_re[n_cores - 1], T_re[n_cores - 1], {{0, 0}});
            u_re[n_cores] += multiply(v, S_re[n_cores - 1], {{0, 0}});
        }

        var_energy = Variable::New<VarFloat>("var_energy", VarFlags::AGGREGATE);
        var_energy += vars_u_re[n_cores][var_const_0];
        var_energy += var_params_self_species[var_self_species];

        bond_re.assign(vars_u_re.begin(), vars_u_re.end());
        bond_re[0] = Variable::NONE;
        bond_im.assign(vars_u_im.begin(), vars_u_im.end());

        std::vector<Variable> keep = {var_energy, var_ett_input_gvector};
        for (auto& b : bond_re) if (!b.empty()) keep.push_back(b);
        for (auto& b : bond_im) if (!b.empty()) keep.push_back(b);
        Simplify(keep);
    }

    // Fills exe's inputs as AnyPairDescriptorPot::AccumulateSiteEnergy_template does; returns the pair count.
    int load(AnyTwoLevelExecutable& exe, const ExtCfg::NeighborhoodIterator& nbh) {
        const double cutoff_sq = Cutoff() * Cutoff();
        int nbhr_count = 0;
        for (int j = 0; j < (int) nbh.size(); j++)
            if (norm_sq(nbh.rel_pos(j)) < cutoff_sq) nbhr_count++;
        exe.Prepare(nbhr_count);
        double* p_in = (double*) exe.in_aggregate_vars();
        *p_in++ = radial_scaling();
        std::memcpy(p_in, params_.begin(), param_count() * sizeof(double));
        p_in += param_count();
        *p_in++ = nbh.self_species();
        int k = 0;
        for (int j = 0; j < (int) nbh.size(); j++)
            if (norm_sq(nbh.rel_pos(j)) < cutoff_sq) {
                for (int a = 0; a < 3; a++) ((double*) exe.in_component_vars(k))[a] = nbh.rel_pos(j)[a];
                ((double*) exe.in_component_vars(k))[3] = nbh.nbhr_species(j);
                k++;
            }
        return nbhr_count;
    }
};

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

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: mlip4_probe PROBE_FILE\n";
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
    ETNProbe etn(shape, ett_ranks, basis, SpeciesOrder(species), false);

    size_t param_count;
    in >> param_count;
    if (param_count != etn.param_count()) {
        std::cerr << "mlip4_probe: " << param_count << " params given, mlip-4's ETN has " << etn.param_count() << '\n';
        return 1;
    }
    for (size_t p = 0; p < param_count; p++) in >> etn.param_data()[p];
    etn.build();

    // The radial basis at probe distances, as RadialBasisCinf::CalcDer gives it (derivatives in r^2)
    size_t dist_count;
    in >> dist_count;
    std::vector<double> dists(dist_count), vals, ders;
    for (auto& d : dists) in >> d;
    for (double d : dists) {
        std::vector<double> v(rb_size), dv(rb_size);
        basis->CalcDer(d * d, 0, 0, v.data(), dv.data());
        for (size_t n = 0; n < rb_size; n++) {
            vals.push_back(v[n]);
            ders.push_back(dv[n] * 2 * d);
        }
    }
    print("rb_vals", {dist_count, rb_size}, vals);
    print("rb_ders", {dist_count, rb_size}, ders);

    // Dense real 3-j tables over l1, l2, l3 < wigner_l_count, indexed (l m) as l * l + l + m
    size_t wigner_l_count;
    in >> wigner_l_count;
    const size_t lm = wigner_l_count * wigner_l_count;
    std::vector<double> w_re(lm * lm * lm), w_im(lm * lm * lm);
    for (int l1 = 0; l1 < (int) wigner_l_count; l1++)
        for (int m1 = -l1; m1 <= l1; m1++)
            for (int l2 = 0; l2 < (int) wigner_l_count; l2++)
                for (int m2 = -l2; m2 <= l2; m2++)
                    for (int l3 = 0; l3 < (int) wigner_l_count; l3++)
                        for (int m3 = -l3; m3 <= l3; m3++) {
                            const size_t idx = ((l1 * l1 + l1 + m1) * lm + (l2 * l2 + l2 + m2)) * lm + (l3 * l3 + l3 + m3);
                            w_re[idx] = wigner_rsph_re(l1, m1, l2, m2, l3, m3);
                            w_im[idx] = wigner_rsph_im(l1, m1, l2, m2, l3, m3);
                        }
    print("wigner_re", {lm, lm, lm}, w_re);
    print("wigner_im", {lm, lm, lm}, w_im);

    // Structures: n_atoms, cell rows, then type x y z per atom
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
        ExtCfg ext(cfg, max_dist);

        std::vector<double> inputs, eatom, energy_total(1, 0.0);
        std::vector<std::vector<double>> bonds_re(core_count + 1), bonds_im(core_count + 1);
        std::vector<Vec3> forces(ext.size_with_ghost(), Vec3::ZERO());
        std::vector<double> virial(6, 0.0);
        auto& exe = *etn.exe_tensors;
        const double cutoff_sq = max_dist * max_dist;

        for (const auto& nbh : ext.nbhs()) {
            etn.load(exe, nbh);
            exe.Run();
            // Outputs in build()'s order: input vector, energy, bonds re, bonds im
            size_t out = 0;
            const size_t input_size = etn.input_size();
            for (size_t k = 0; k < input_size; k++) inputs.push_back(exe.out_aggregate_vars(out, k));
            out++;
            eatom.push_back(exe.out_aggregate_vars(out++));
            bonds_re[0].push_back(1.0);
            for (size_t b = 1; b <= core_count; b++) {
                const size_t size = etn.bond_re[b].n_elem();
                for (size_t k = 0; k < size; k++) bonds_re[b].push_back(exe.out_aggregate_vars(out, k));
                out++;
            }
            for (size_t b = 0; b <= core_count; b++) {
                const size_t size = etn.bond_re[b].empty() ? 1 : etn.bond_re[b].n_elem();
                if (etn.bond_im[b].empty()) {
                    for (size_t k = 0; k < size; k++) bonds_im[b].push_back(0.0);
                    continue;
                }
                for (size_t k = 0; k < size; k++) bonds_im[b].push_back(exe.out_aggregate_vars(out, k));
                out++;
            }

            // Forces and virial from mlip-4's own derivative program, tallied as PairMTP tallies them
            double energy = 0.0;
            etn.AccumulateSiteEnergyDers(nbh, energy, forces);
            energy_total[0] += energy;
            auto& ders = etn.energy_ders_executable();
            etn.load(ders, nbh);
            ders.Run();
            int k = 0;
            for (int j = 0; j < (int) nbh.size(); j++) {
                const Vec3 r = nbh.rel_pos(j);
                if (norm_sq(r) >= cutoff_sq) continue;
                const double* g = (double*) ders.out_component_vars(k++);
                virial[0] -= r[0] * g[0];
                virial[1] -= r[1] * g[1];
                virial[2] -= r[2] * g[2];
                virial[3] -= r[0] * g[1];
                virial[4] -= r[0] * g[2];
                virial[5] -= r[1] * g[2];
            }
        }

        std::vector<double> folded(n_atoms * 3, 0.0);
        for (size_t a = 0; a < ext.size_with_ghost(); a++)
            for (int c = 0; c < 3; c++) folded[ext.ghost_atom_ind(a) * 3 + c] += forces[a][c];

        const std::string prefix = "s" + std::to_string(s) + "/";
        print(prefix + "energy", {}, energy_total);
        print(prefix + "eatom", {n_atoms}, eatom);
        print(prefix + "forces", {n_atoms, 3}, folded);
        print(prefix + "virials", {6}, virial);
        print(prefix + "inputs", {n_atoms, inputs.size() / n_atoms}, inputs);
        for (size_t b = 0; b <= core_count; b++) {
            print(prefix + "bonds/" + std::to_string(b), {n_atoms, bonds_re[b].size() / n_atoms}, bonds_re[b]);
            print(prefix + "bonds_im/" + std::to_string(b), {n_atoms, bonds_im[b].size() / n_atoms}, bonds_im[b]);
        }
    }
    return 0;
}
