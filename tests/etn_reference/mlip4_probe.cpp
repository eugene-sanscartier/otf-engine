/* -*- c++ -*- ----------------------------------------------------------
   mlip-4's ETN tensors for check_etn_kernels.py --capture.

   Reads a probe file (written by check_etn_kernels.py) holding the ETN's
   shape and params, probe distances, directions and structures, and prints
   every tensor mlip-4 computes for them: the radial basis, the real 3-j
   tables, the real harmonics and their gradients, and per structure the
   energy, forces and stress of mlip-4's CalcCfg and its gradient of a loss
   with given derivatives (CalcCfg::AccumulateGrads), each atom's site energy,
   the first atoms' site-energy gradients (AccumulateSiteEnergyGrads), and
   the first atoms' input and bond vectors. Those come from mlip-4's own
   program, recomposed by ETNProbe below as ETN::ComposeETTProgram composes
   it with the bond vectors kept as outputs; the probe fails if a site
   energy of the recomposed program departs from the original's.

   Build against a compiled mlip-4 (see check_etn_kernels.py).
------------------------------------------------------------------------- */

#include "pch.h"
#include <pots/etn/etn.h>
#include <pots/radial_basis/cinf/rb_cinf.h>
#include <simulations/simulation.h>
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

// Y_lm and their gradients at each unit vector, from mlip-4's AngularBasis program run by its VM;
// at a unit vector the gradient of Y_lm(r / |r|) is tangent to the sphere.
static void print_harmonics(size_t l_count, const std::vector<std::array<double, 3>>& directions) {
    const size_t count = l_count * l_count;
    Variable x = Variable::New<VarFloat>("x", VarFlags::EXTERN);
    Variable y = Variable::New<VarFloat>("y", VarFlags::EXTERN);
    Variable z = Variable::New<VarFloat>("z", VarFlags::EXTERN);
    Variable r_sq = Variable::New<VarFloat>("r_sq");
    r_sq += x * x;
    r_sq += y * y;
    r_sq += z * z;
    Variable r_inv = Variable::New<VarFloat>("r_inv");
    r_inv += pow(r_sq, -0.5);
    AngularBasis angular(l_count);
    std::vector<Variable> ylm = angular.ComposeProgram(x, y, z, r_inv);

    std::vector<Variable> weights(count), grads = {Variable::New<VarFloat>("dx"), Variable::New<VarFloat>("dy"), Variable::New<VarFloat>("dz")};
    for (size_t k = 0; k < count; k++) weights[k] = Variable::New<VarFloat>("weight", VarFlags::EXTERN | VarFlags::AGGREGATE);
    DifferentiateLincomb({x, y, z}, ylm, weights, grads);

    std::vector<Variable> outs = ylm;
    outs.insert(outs.end(), grads.begin(), grads.end());
    Simplify(outs);
    std::vector<Variable> in_vars = weights;
    in_vars.insert(in_vars.end(), {x, y, z});
    VmTwoLevelExecutable exe;
    exe.Construct(in_vars, outs);

    // Output k sits at index k among the outputs of its level
    std::vector<size_t> out_index(outs.size());
    size_t n_aggregate = 0, n_component = 0;
    for (size_t k = 0; k < outs.size(); k++)
        out_index[k] = outs[k].level() == AnyTwoLevelExecutable::Level::Aggregate ? n_aggregate++ : n_component++;
    auto value = [&](size_t k) {
        return outs[k].level() == AnyTwoLevelExecutable::Level::Aggregate ? exe.out_aggregate_vars(out_index[k]) : exe.out_component_vars(0, out_index[k]);
    };

    for (size_t d = 0; d < directions.size(); d++) {
        std::vector<double> vals(count), grad_vals(count * 3);
        for (size_t k = 0; k <= count; k++) {
            exe.Prepare(1);
            for (size_t w = 0; w < count; w++) exe.in_aggregate_vars(w) = (w == k) ? 1.0 : 0.0;
            for (int a = 0; a < 3; a++) exe.in_component_vars(0, a) = directions[d][a];
            exe.Run();
            if (k == count)
                for (size_t q = 0; q < count; q++) vals[q] = value(q);
            else
                for (int a = 0; a < 3; a++) grad_vals[k * 3 + a] = value(count + a);
        }
        print("ylm/" + std::to_string(d) + "/vals", {count}, vals);
        print("ylm/" + std::to_string(d) + "/grads", {count, 3}, grad_vals);
    }
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
    auto plain = std::make_shared<ETN>(shape, ett_ranks, basis, SpeciesOrder(species), false);

    size_t param_count;
    in >> param_count;
    if (param_count != etn.param_count()) {
        std::cerr << "mlip4_probe: " << param_count << " params given, mlip-4's ETN has " << etn.param_count() << '\n';
        return 1;
    }
    for (size_t p = 0; p < param_count; p++) in >> etn.param_data()[p];
    std::copy(etn.param_data(), etn.param_data() + param_count, plain->params().begin());
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

    // Harmonics: l_count, then the unit vectors
    size_t harmonics_l_count, direction_count;
    in >> harmonics_l_count >> direction_count;
    std::vector<std::array<double, 3>> directions(direction_count);
    for (auto& d : directions) in >> d[0] >> d[1] >> d[2];
    print_harmonics(harmonics_l_count, directions);

    // Structures: tensors are printed for the first tensor_atoms atoms and site-energy gradients
    // for the first grad_atoms; then n_atoms, cell rows, type x y z per atom, and the loss
    // derivatives
    size_t tensor_atoms, grad_atoms, structure_count;
    in >> tensor_atoms >> grad_atoms >> structure_count;
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

        // The loss derivatives: energy, virial (xx, yy, zz, xy, xz, yz), then forces per atom
        std::vector<double> loss_seeds(1 + 6 + 3 * n_atoms);
        for (auto& x : loss_seeds) in >> x;

        // Energy, forces and stress from mlip-4's own CalcCfg
        CalcCfg calc(cfg, true, true);
        calc.AttachPot(plain);
        calc.Run();
        const auto& stress = calc.get_stress();
        std::vector<double> folded(n_atoms * 3);
        for (size_t i = 0; i < n_atoms; i++)
            for (int a = 0; a < 3; a++) folded[i * 3 + a] = calc.get_forces()[i][a];

        // The loss gradient from CalcCfg::AccumulateGrads, whose results are the energy, the
        // stress (xx, yy, zz, yz, xz, xy) and the forces
        std::vector<double> grad_over_results = {loss_seeds[0], loss_seeds[1], loss_seeds[2], loss_seeds[3], loss_seeds[6], loss_seeds[5], loss_seeds[4]};
        grad_over_results.insert(grad_over_results.end(), loss_seeds.begin() + 7, loss_seeds.end());
        std::vector<double> loss_grad(param_count, 0.0);
        calc.AccumulateGrads(grad_over_results.data(), loss_grad.data());

        // Per atom: the input and bond vectors from the recomposed program, and the site energy,
        // checked against the original program's
        ExtCfg ext(cfg, max_dist);
        std::vector<double> inputs, eatom, site_grads;
        std::vector<std::vector<double>> bonds_re(core_count + 1), bonds_im(core_count + 1);
        auto& exe = *etn.exe_tensors;
        for (const auto& nbh : ext.nbhs()) {
            etn.load(exe, nbh);
            exe.Run();
            size_t out = 0;
            std::vector<double> input(etn.input_size());
            for (size_t k = 0; k < input.size(); k++) input[k] = exe.out_aggregate_vars(out, k);
            out++;
            const double site_energy = exe.out_aggregate_vars(out++);
            eatom.push_back(site_energy);

            double original = 0.0;
            plain->AccumulateSiteEnergy(nbh, original);
            if (std::abs(original - site_energy) > 1e-12 * std::max(1.0, std::abs(original))) {
                std::cerr << "mlip4_probe: the recomposed program gives site energy " << site_energy << ", mlip-4's " << original << '\n';
                return 1;
            }

            std::vector<std::vector<double>> re(core_count + 1), im(core_count + 1);
            re[0] = {1.0};
            for (size_t b = 1; b <= core_count; b++) {
                for (size_t k = 0; k < etn.bond_re[b].n_elem(); k++) re[b].push_back(exe.out_aggregate_vars(out, k));
                out++;
            }
            for (size_t b = 0; b <= core_count; b++) {
                if (etn.bond_im[b].empty()) {
                    im[b].assign(re[b].size(), 0.0);
                    continue;
                }
                for (size_t k = 0; k < etn.bond_im[b].n_elem(); k++) im[b].push_back(exe.out_aggregate_vars(out, k));
                out++;
            }

            if (nbh.my_ind() >= tensor_atoms) continue;
            if (nbh.my_ind() < grad_atoms) {
                std::vector<double> row(param_count, 0.0);
                plain->AccumulateSiteEnergyGrads(nbh, row.data());
                site_grads.insert(site_grads.end(), row.begin(), row.end());
            }
            inputs.insert(inputs.end(), input.begin(), input.end());
            for (size_t b = 0; b <= core_count; b++) {
                bonds_re[b].insert(bonds_re[b].end(), re[b].begin(), re[b].end());
                bonds_im[b].insert(bonds_im[b].end(), im[b].begin(), im[b].end());
            }
        }

        // mlip-4's stress is (xx, yy, zz, yz, xz, xy) of sum_i F_i (x) r_i, symmetrized
        const size_t kept = std::min(tensor_atoms, n_atoms);
        const std::string prefix = "s" + std::to_string(s) + "/";
        print(prefix + "energy", {}, {calc.get_energy()});
        print(prefix + "eatom", {n_atoms}, eatom);
        print(prefix + "forces", {n_atoms, 3}, folded);
        print(prefix + "virials", {6}, {stress[0], stress[1], stress[2], stress[5], stress[4], stress[3]});
        print(prefix + "inputs", {kept, inputs.size() / kept}, inputs);
        for (size_t b = 0; b <= core_count; b++) {
            print(prefix + "bonds/" + std::to_string(b), {kept, bonds_re[b].size() / kept}, bonds_re[b]);
            print(prefix + "bonds_im/" + std::to_string(b), {kept, bonds_im[b].size() / kept}, bonds_im[b]);
        }
        print(prefix + "loss_grad", {param_count}, loss_grad);
        print(prefix + "grad", {std::min(grad_atoms, n_atoms), param_count}, site_grads);
    }
    return 0;
}
