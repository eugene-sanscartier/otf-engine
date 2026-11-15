/* -*- c++ -*- ----------------------------------------------------------
   Standalone ETN potential — no LAMMPS dependency.
   Computes mlip-4's ETN (src/pots/etn/etn.cpp at 79a39bf), which has no
   LAMMPS pair style; hand-written passes replace mlip-4's program.
   Original author of mlip-4's ETN: Max Hodapp
------------------------------------------------------------------------- */

#include "etn_potential.h"

#include "etn_radial_basis.h"
#include "text_file_reader.h"

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

CovariantShape::CovariantShape(std::vector<int> ranks) : ranks(std::move(ranks)) {
    offsets.resize(this->ranks.size());
    for (size_t l = 0; l < this->ranks.size(); l++) {
        offsets[l] = size;
        size += (2 * (int) l + 1) * this->ranks[l];
    }
}

PairETN::PairETN(const std::string& filename) {
    std::ifstream f(filename);
    if (!f.is_open())
        throw std::runtime_error("PairETN: cannot open file '" + filename + "'");
    PairETN::read_file(f);
}

PairETN::~PairETN() {
    delete radial_basis;
}

/* ----------------------------------------------------------------------
   Main Computation Function
   ---------------------------------------------------------------------- */
void PairETN::compute(const NeighList& list) {
    const int l_count = (int) species_ranks.size();
    const int core_count = get_core_count();
    const int harmonic_count = l_count * l_count;
    const int mixed_size = mixed_offsets[l_count];
    int nbr_offset = 0;

    energy = 0.0;
    eatom.assign(list.n_atoms, 0.0);
    forces.assign((size_t) list.n_atoms * 3, 0.0);
    std::fill(virial, virial + 6, 0.0);
    inputs.assign((size_t) list.inum * input_shape.size, 0.0);
    bonds.assign((size_t) list.inum * bonds_size, 0.0);

    // Loop over all provided neighbourhoods
    for (int ii = 0; ii < list.inum; ii++) {
        const int i = list.ilist[ii];
        const int itype = list.types[i];
        double* v = inputs.data() + (size_t) ii * input_shape.size;
        const int valid_count = site_input(list, ii, nbr_offset, v);
        nbr_offset += list.numneigh[ii];

        // ------------ Tensor Train, Forwards ------------
        double* bond = bonds.data() + (size_t) ii * bonds_size;
        train_forward(v, bond);

        const double nbh_energy = bond[bond_offsets[core_count]] + species_coeffs[itype];
        eatom[i] += nbh_energy;
        energy += nbh_energy;

        // ------------ Tensor Train, Backwards ------------
        std::fill(bond_adj.begin(), bond_adj.end(), 0.0);
        std::fill(input_adj.begin(), input_adj.end(), 0.0);
        bond_adj[bond_offsets[core_count]] = 1.0;
        train_backward(v, bond, bond_adj.data(), input_adj.data(), nullptr, nullptr);

        // ------------ Forces Through the Input Vector ------------
        for (int jj = 0; jj < valid_count; jj++) {
            const int j = cached_j[jj];
            const double* r = valid_dr[jj].data();
            const double* cache = neighbor_cache.data() + (size_t) jj * cache_stride;
            const double inv_dist = cache[0];
            const double* ylm = cache + 1;
            const double* ylm_grads = ylm + harmonic_count;
            const double* mixed = ylm_grads + 3 * harmonic_count;
            const double* mixed_ders = mixed + mixed_size;

            double angular[3] = {0, 0, 0};
            double radial_force = 0;
            for (int l = 0; l < l_count; l++) {
                const int rank_r = radial_ranks[l];
                const double* b = mixed + mixed_offsets[l];
                const double* db = mixed_ders + mixed_offsets[l];
                const int shift = l == 0 ? 1 : 0;
                for (int m = -l; m <= l; m++) {
                    const int index = l * l + l + m;
                    const double* adj = input_adj.data() + input_shape.index(l, m) + shift;
                    double ylm_adj = 0, mixed_adj = 0;
                    for (int n = 0; n < rank_r; n++) {
                        ylm_adj += b[n] * adj[n];
                        mixed_adj += db[n] * adj[n];
                    }
                    for (int a = 0; a < 3; a++) angular[a] += ylm_adj * ylm_grads[index * 3 + a];
                    radial_force += ylm[index] * mixed_adj;
                }
            }

            // dE_i / dr_ij, with r_ij = r_j - r_i
            double temp_force[3];
            for (int a = 0; a < 3; a++) temp_force[a] = inv_dist * (angular[a] + radial_force * r[a]);

            forces[i * 3 + 0] += temp_force[0];
            forces[i * 3 + 1] += temp_force[1];
            forces[i * 3 + 2] += temp_force[2];

            forces[j * 3 + 0] -= temp_force[0];
            forces[j * 3 + 1] -= temp_force[1];
            forces[j * 3 + 2] -= temp_force[2];

            // Accumulate virial stress as PairMTP does
            virial[0] -= r[0] * temp_force[0];    //xx
            virial[1] -= r[1] * temp_force[1];    //yy
            virial[2] -= r[2] * temp_force[2];    //zz
            virial[3] -= r[0] * temp_force[1];    //xy
            virial[4] -= r[0] * temp_force[2];    //xz
            virial[5] -= r[1] * temp_force[2];    //yz
        }
    }
}

/* ----------------------------------------------------------------------
   The input vector of one central atom, and the tensor train each way.
------------------------------------------------------------------------- */
int PairETN::site_input(const NeighList& list, int ii, int nbr_offset, double* v) {
    const int l_count = (int) species_ranks.size();
    const int harmonic_count = l_count * l_count;
    const int mixed_size = mixed_offsets[l_count];
    const int i = list.ilist[ii];
    const int itype = list.types[i];
    const int jnum = list.numneigh[ii];
    const int* nbrs = list.firstneigh + nbr_offset;
    const double* dr = list.displacements + nbr_offset * 3;
    int valid_count = 0;

    // Resize per neighbor arrays
    if (cache_size < jnum) {
        neighbor_cache.resize((size_t) jnum * cache_stride);
        cached_j.resize(jnum);
        valid_dr.resize(jnum);
        cache_size = jnum;
    }

    std::fill(v, v + input_shape.size, 0.0);
    v[0] = 1.0;
    for (int jj = 0; jj < jnum; jj++) {
        const int j = nbrs[jj];
        const int jtype = list.types[j];
        const double r[3] = {dr[jj * 3 + 0], dr[jj * 3 + 1], dr[jj * 3 + 2]};
        const double rsq = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];

        if (rsq >= max_cutoff_sq) continue;
        cached_j[valid_count] = j;
        valid_dr[valid_count] = {r[0], r[1], r[2]};

        const double dist = std::sqrt(rsq);
        const double inv_dist = 1.0 / dist;
        const double u[3] = {r[0] * inv_dist, r[1] * inv_dist, r[2] * inv_dist};
        double* cache = neighbor_cache.data() + (size_t) valid_count * cache_stride;
        cache[0] = inv_dist;
        double* ylm = cache + 1;
        double* ylm_grads = ylm + harmonic_count;
        double* mixed = ylm_grads + 3 * harmonic_count;
        double* mixed_ders = mixed + mixed_size;
        valid_count++;

        radial_basis->calc_radial_basis_ders(dist);
        const double* basis_vals = radial_basis->radial_basis_vals.data();
        const double* basis_ders = radial_basis->radial_basis_ders.data();
        harmonics.compute(u);
        std::copy(harmonics.vals.begin(), harmonics.vals.end(), ylm);
        std::copy(harmonics.grads.begin(), harmonics.grads.end(), ylm_grads);

        for (int l = 0; l < l_count; l++) {
            const int n_rad = radial_func_counts[l], rank_r = radial_ranks[l];
            const double* s = radial_mixing[l].data() + (size_t) (itype * species_count + jtype) * n_rad * rank_r;
            double* b = mixed + mixed_offsets[l];
            double* db = mixed_ders + mixed_offsets[l];
            for (int n = 0; n < rank_r; n++) {
                double val = 0, der = 0;
                for (int np = 0; np < n_rad; np++) {
                    val += basis_vals[np] * s[np * rank_r + n];
                    der += basis_ders[np] * s[np * rank_r + n];
                }
                b[n] = val;
                db[n] = der;
            }
            const int shift = l == 0 ? 1 : 0;
            for (int m = -l; m <= l; m++) {
                const double y = ylm[l * l + l + m];
                double* pv = v + input_shape.index(l, m) + shift;
                for (int n = 0; n < rank_r; n++) pv[n] += y * b[n];
            }
        }
    }
    return valid_count;
}

void PairETN::train_forward(const double* v, double* bond) const {
    std::fill(bond, bond + bonds_size, 0.0);
    bond[bond_offsets[0]] = 1.0;
    for (int c = 0; c < get_core_count(); c++) {
        const CovariantShape& left = bond_shapes[c];
        const CovariantShape& right = bond_shapes[c + 1];
        for (const CoreTerm& t : core_terms[c]) {
            const Core& core = t.imaginary ? cores_im[c] : cores[c];
            const double* coeffs = t.imaginary ? core_coeffs_im[c].data() : core_coeffs[c].data();
            contract(core, coeffs, left, right, bond + bond_offsets[c] + t.in_part * left.size, v, bond + bond_offsets[c + 1] + t.out_part * right.size);
        }
    }
}

void PairETN::train_backward(const double* v, const double* bond, double* bond_adj, double* v_adj, double* const* core_adj, double* const* core_im_adj) const {
    for (int c = get_core_count() - 1; c >= 0; c--) {
        const CovariantShape& left = bond_shapes[c];
        const CovariantShape& right = bond_shapes[c + 1];
        for (const CoreTerm& t : core_terms[c]) {
            const Core& core = t.imaginary ? cores_im[c] : cores[c];
            const double* coeffs = t.imaginary ? core_coeffs_im[c].data() : core_coeffs[c].data();
            double* c_adj = core_adj ? (t.imaginary ? core_im_adj[c] : core_adj[c]) : nullptr;
            const int in = bond_offsets[c] + t.in_part * left.size;
            const int out = bond_offsets[c + 1] + t.out_part * right.size;
            contract_adjoint(core, coeffs, left, right, bond + in, v, bond_adj + out, bond_adj + in, v_adj, c_adj);
        }
    }
}

void PairETN::contract(const Core& core, const double* c, const CovariantShape& left, const CovariantShape& right, const double* u, const double* v, double* out) const {
    for (const CoreBlock& block : core.blocks) {
        const int n1_count = left.ranks[block.l1], n2_count = input_shape.ranks[block.l2], n3_count = right.ranks[block.l3];
        for (int e = block.w_begin; e < block.w_end; e++) {
            const WignerEntry& w = wigner[e];
            const double* pu = u + left.index(block.l1, w.m1);
            const double* pv = v + input_shape.index(block.l2, w.m2);
            double* po = out + right.index(block.l3, w.m3);
            for (int n3 = 0; n3 < n3_count; n3++) {
                const double* c3 = c + block.base + n3 * block.stride3;
                double sum = 0;
                for (int n2 = 0; n2 < n2_count; n2++) {
                    const double* c2 = c3 + n2 * block.stride2;
                    double t = 0;
                    for (int n1 = 0; n1 < n1_count; n1++) t += c2[n1] * pu[n1];
                    sum += t * pv[n2];
                }
                po[n3] += w.w * sum;
            }
        }
    }
}

void PairETN::contract_adjoint(const Core& core, const double* c, const CovariantShape& left, const CovariantShape& right, const double* u, const double* v, const double* out_adj, double* u_adj, double* v_adj, double* c_adj) const {
    for (const CoreBlock& block : core.blocks) {
        const int n1_count = left.ranks[block.l1], n2_count = input_shape.ranks[block.l2], n3_count = right.ranks[block.l3];
        for (int e = block.w_begin; e < block.w_end; e++) {
            const WignerEntry& w = wigner[e];
            const int iu = left.index(block.l1, w.m1), iv = input_shape.index(block.l2, w.m2);
            const double* po = out_adj + right.index(block.l3, w.m3);
            for (int n3 = 0; n3 < n3_count; n3++) {
                const double g = w.w * po[n3];
                if (g == 0.0) continue;
                const int c3 = block.base + n3 * block.stride3;
                for (int n2 = 0; n2 < n2_count; n2++) {
                    const double* c2 = c + c3 + n2 * block.stride2;
                    const double gv = g * v[iv + n2];
                    double t = 0;
                    for (int n1 = 0; n1 < n1_count; n1++) {
                        t += c2[n1] * u[iu + n1];
                        u_adj[iu + n1] += gv * c2[n1];
                    }
                    v_adj[iv + n2] += g * t;
                    if (c_adj) {
                        double* a2 = c_adj + c3 + n2 * block.stride2;
                        for (int n1 = 0; n1 < n1_count; n1++) a2[n1] += gv * u[iu + n1];
                    }
                }
            }
        }
    }
}

void PairETN::eval_radial_basis(double dist, double* vals_out, double* ders_out) {
    radial_basis->calc_radial_basis_ders(dist);
    std::copy(radial_basis->radial_basis_vals.begin(), radial_basis->radial_basis_vals.end(), vals_out);
    std::copy(radial_basis->radial_basis_ders.begin(), radial_basis->radial_basis_ders.end(), ders_out);
}

int PairETN::coeff_count() const {
    int count = species_count;
    for (const auto* blocks : {&radial_coeffs, &species_pair_coeffs, &core_coeffs, &core_coeffs_im})
        for (const auto& block : *blocks) count += (int) block.size();
    return count;
}

void PairETN::set_radial_coeffs(int l, const double* c) {
    std::copy(c, c + radial_coeffs[l].size(), radial_coeffs[l].begin());
    prepare_radial_mixing();
}

void PairETN::set_species_pair_coeffs(int l, const double* c) {
    std::copy(c, c + species_pair_coeffs[l].size(), species_pair_coeffs[l].begin());
    prepare_radial_mixing();
}

void PairETN::set_core_coeffs(int i, bool imaginary, const double* c) {
    std::vector<double>& block = imaginary ? core_coeffs_im[i] : core_coeffs[i];
    std::copy(c, c + block.size(), block.begin());
}

/* ----------------------------------------------------------------------
   ETN file parsing. Includes allocation. Excludes the radial basis
   hyperparameters, which the radial basis constructor reads.
------------------------------------------------------------------------- */
template <class T>
static std::vector<T> parse_values(std::istringstream ss) {
    std::vector<T> values;
    T x;
    while (ss >> x) values.push_back(x);
    return values;
}

void PairETN::read_file(std::istream& is) {
    TextFileReader tfr(is);

    tfr.expect("ETN");
    if (tfr.next_string("version") != "1.0.0")
        throw std::runtime_error("PairETN: ETN file must have version \"1.0.0\"");

    tfr.advance();

    // Read the potential name (optional field)
    if (tfr.keyword() == "potential_name") {
        tfr.rest() >> potential_name;
        tfr.advance();
    }

    // Check the scaling
    if (tfr.keyword() == "scaling") {
        tfr.rest() >> scaling;
        tfr.advance();
    }

    if (tfr.keyword() != "species_count")
        throw std::runtime_error("PairETN: no species count is specified");
    tfr.rest() >> species_count;
    if (species_count < 1)
        throw std::runtime_error("PairETN: ETN species count must be positive");

    tfr.advance();
    if (tfr.keyword() != "radial_basis_type")
        throw std::runtime_error("PairETN: no radial basis set type is specified");
    tfr.rest() >> radial_basis_type;

    // Set the type of radial basis.
    if (radial_basis_type == "RBCinf")
        radial_basis = new RBCinf(tfr);
    else
        throw std::runtime_error("PairETN: the radial basis type '" + radial_basis_type + "' was not found/available");

    radial_basis->scaling = scaling;
    radial_basis_size = radial_basis->size;
    min_cutoff = radial_basis->min_cutoff;
    max_cutoff = radial_basis->max_cutoff;
    max_cutoff_sq = max_cutoff * max_cutoff;

    // Shape of the input and of the tensor train
    tfr.expect("species_ranks");
    species_ranks = parse_values<int>(tfr.rest());
    tfr.expect("radial_funcs_counts");
    radial_func_counts = parse_values<int>(tfr.rest());
    tfr.expect("radial_ranks");
    radial_ranks = parse_values<int>(tfr.rest());

    const int core_count = tfr.next_int("core_count");
    if (core_count < 1)
        throw std::runtime_error("PairETN: ETN core count must be positive");
    tfr.expect("bond_ranks");
    for (int b = 0; b <= core_count; b++) {
        tfr.advance();
        bond_ranks.push_back(parse_values<int>(std::istringstream(tfr.line())));
    }

    prepare_cores();

    // An untrained potential may omit every coefficient; mlip-4's ETN::PostConstruct
    // gives the k-th of its params sin(10 + k^2).
    if (!tfr.next_line() || tfr.keyword() != "radial_coeffs") {
        trained = false;
        int k = 0;
        for (auto* blocks : {&radial_coeffs, &species_pair_coeffs, &core_coeffs, &core_coeffs_im})
            for (auto& block : *blocks)
                for (double& x : block) x = std::sin(10 + k * k), k++;
        for (double& x : species_coeffs) x = std::sin(10 + k * k), k++;
        prepare_radial_mixing();
        return;
    }

    // One line of values per block, each of the length the shape gives it
    auto read_block = [&](std::vector<double>& block, const std::string& what) {
        tfr.advance();
        const std::vector<double> values = parse_values<double>(std::istringstream(tfr.line()));
        if (values.size() != block.size())
            throw std::runtime_error("PairETN: " + what + " has " + std::to_string(values.size()) + " values, the shape gives " + std::to_string(block.size()));
        block = values;
    };

    const int l_count = (int) species_ranks.size();
    for (int l = 0; l < l_count; l++)
        read_block(radial_coeffs[l], "radial_coeffs of l = " + std::to_string(l));
    tfr.expect("species_pair_coeffs");
    for (int l = 0; l < l_count; l++)
        read_block(species_pair_coeffs[l], "species_pair_coeffs of l = " + std::to_string(l));
    tfr.expect("core_coeffs");
    for (int c = 0; c < core_count; c++)
        read_block(core_coeffs[c], "core_coeffs of core " + std::to_string(c));
    if (has_imaginary_cores()) {
        tfr.expect("core_coeffs_im");
        for (int c = 1; c < core_count - 1; c++)
            read_block(core_coeffs_im[c], "core_coeffs_im of core " + std::to_string(c));
    }

    tfr.expect("species_coeffs");
    std::istringstream species_ss = tfr.rest();
    for (int s = 0; s < species_count; s++)
        if (!(species_ss >> species_coeffs[s]))
            throw std::runtime_error("PairETN: not enough species coefficients");

    prepare_radial_mixing();
}

/* ----------------------------------------------------------------------
   Layouts of the input, the bonds and the cores, and the coefficient
   blocks and working buffers they size.
------------------------------------------------------------------------- */
void PairETN::prepare_cores() {
    const int l_count = (int) species_ranks.size();
    if (l_count == 0 || (int) radial_func_counts.size() != l_count || (int) radial_ranks.size() != l_count)
        throw std::runtime_error("PairETN: species_ranks, radial_funcs_counts and radial_ranks must have the same positive length");
    for (const auto* ranks : {&species_ranks, &radial_func_counts, &radial_ranks})
        if (std::all_of(ranks->begin(), ranks->end(), [](int r) { return r == 0; }) || std::any_of(ranks->begin(), ranks->end(), [](int r) { return r < 0; }))
            throw std::runtime_error("PairETN: a size as a function of l must be non-negative and not all zero");
    for (int n : radial_func_counts)
        if (n > radial_basis_size)
            throw std::runtime_error("PairETN: radial_funcs_counts exceeds radial_basis_size");
    if (bond_ranks.front() != std::vector<int>{1} || bond_ranks.back() != std::vector<int>{1})
        throw std::runtime_error("PairETN: the first and last bond ranks must be {1}");
    for (const auto& ranks : bond_ranks)
        if (std::all_of(ranks.begin(), ranks.end(), [](int r) { return r == 0; }) || std::any_of(ranks.begin(), ranks.end(), [](int r) { return r < 0; }))
            throw std::runtime_error("PairETN: a bond rank as a function of l must be non-negative and not all zero");

    harmonics = RealHarmonics(l_count);

    // The input carries the constant 1 ahead of the l = 0 radial channels
    std::vector<int> input_ranks = radial_ranks;
    input_ranks[0] += 1;
    input_shape = CovariantShape(input_ranks);

    mixed_offsets.assign(l_count + 1, 0);
    for (int l = 0; l < l_count; l++) mixed_offsets[l + 1] = mixed_offsets[l] + radial_ranks[l];

    for (const auto& ranks : bond_ranks) {
        bond_shapes.emplace_back(ranks);
        bond_offsets.push_back(bonds_size);
        bonds_size += 2 * bond_shapes.back().size;
    }

    radial_coeffs.resize(l_count);
    species_pair_coeffs.resize(l_count);
    for (int l = 0; l < l_count; l++) {
        radial_coeffs[l].resize(species_ranks[l] * radial_func_counts[l] * radial_ranks[l]);
        species_pair_coeffs[l].resize(species_count * species_count * species_ranks[l]);
    }

    const int core_count = get_core_count();
    cores_im.assign(core_count, Core());
    for (int c = 0; c < core_count; c++)
        cores.push_back(prepare_core(c, false));
    if (has_imaginary_cores())
        for (int c = 1; c < core_count - 1; c++)
            cores_im[c] = prepare_core(c, true);

    // With more than 3 cores, bond b has an imaginary part for 2 <= b <= d-2 and core c for
    // 1 <= c <= d-2; they combine as mlip-4's ComposeETTProgram combines them.
    auto bond_has_im = [&](int b) { return has_imaginary_cores() && b >= 2 && b <= core_count - 2; };
    auto core_has_im = [&](int c) { return has_imaginary_cores() && c >= 1 && c <= core_count - 2; };
    core_terms.assign(core_count, {});
    for (int c = 0; c < core_count; c++) {
        core_terms[c].push_back({false, 0, 0});
        if (bond_has_im(c) && bond_has_im(c + 1)) core_terms[c].push_back({false, 1, 1});
        if (core_has_im(c) && bond_has_im(c)) core_terms[c].push_back({true, 1, 0});
        if (core_has_im(c) && bond_has_im(c + 1)) core_terms[c].push_back({true, 0, 1});
    }

    core_coeffs.resize(core_count);
    core_coeffs_im.resize(core_count);
    for (int c = 0; c < core_count; c++) {
        core_coeffs[c].resize(cores[c].coeff_count);
        core_coeffs_im[c].resize(cores_im[c].coeff_count);
    }
    species_coeffs.resize(species_count);

    cache_stride = 1 + 4 * l_count * l_count + 2 * mixed_offsets[l_count];
    input_adj.resize(input_shape.size);
    bond_adj.resize(bonds_size);
}

// The coefficients of core i ordered l3, n3, l2, n2, l1, n1 with n1 fastest, over the
// (l1, l2, l3) that pass the triangle rule with an even sum (real part) or an odd one.
Core PairETN::prepare_core(int i, bool imaginary) {
    const std::vector<int>& left = bond_shapes[i].ranks;
    const std::vector<int>& middle = input_shape.ranks;
    const std::vector<int>& right = bond_shapes[i + 1].ranks;
    const int parity = imaginary ? 1 : 0;
    auto allowed = [parity](int l1, int l2, int l3) {
        const int sum = l1 + l2 + l3;
        return 2 * std::max({l1, l2, l3}) <= sum && sum % 2 == parity;
    };

    Core core;
    int base3 = 0;
    for (int l3 = 0; l3 < (int) right.size(); l3++) {
        int stride3 = 0;
        for (int l2 = 0; l2 < (int) middle.size(); l2++)
            for (int l1 = 0; l1 < (int) left.size(); l1++)
                if (allowed(l1, l2, l3)) stride3 += middle[l2] * left[l1];

        int base2 = 0;
        for (int l2 = 0; l2 < (int) middle.size(); l2++) {
            int stride2 = 0;
            for (int l1 = 0; l1 < (int) left.size(); l1++)
                if (allowed(l1, l2, l3)) stride2 += left[l1];

            int base1 = 0;
            for (int l1 = 0; l1 < (int) left.size(); l1++) {
                if (!allowed(l1, l2, l3)) continue;
                if (left[l1] > 0 && middle[l2] > 0 && right[l3] > 0) {
                    const std::vector<WignerEntry> entries = real_wigner_entries(l1, l2, l3);
                    const int w_begin = (int) wigner.size();
                    wigner.insert(wigner.end(), entries.begin(), entries.end());
                    core.blocks.push_back({l1, l2, l3, base3 + base2 + base1, stride2, stride3, w_begin, (int) wigner.size()});
                }
                base1 += left[l1];
            }
            base2 += middle[l2] * stride2;
        }
        base3 += right[l3] * stride3;
    }
    core.coeff_count = base3;
    return core;
}

// S_l[s_i, s_j](n', n) = sum_k A_l[s_i, s_j, k] B_l[k, n', n], as mlip-4 contracts the species first.
void PairETN::prepare_radial_mixing() {
    const int l_count = (int) species_ranks.size();
    radial_mixing.resize(l_count);
    for (int l = 0; l < l_count; l++) {
        const int rank_s = species_ranks[l], n_rad = radial_func_counts[l], rank_r = radial_ranks[l];
        const double* radial = radial_coeffs[l].data();
        std::vector<double>& mixing = radial_mixing[l];
        mixing.assign((size_t) species_count * species_count * n_rad * rank_r, 0.0);

        for (int pair = 0; pair < species_count * species_count; pair++) {
            const double* a = species_pair_coeffs[l].data() + pair * rank_s;
            double* s = mixing.data() + (size_t) pair * n_rad * rank_r;
            for (int n = 0; n < rank_r; n++)
                for (int np = 0; np < n_rad; np++) {
                    double sum = 0;
                    for (int k = 0; k < rank_s; k++) sum += a[k] * radial[k + rank_s * (np + n_rad * n)];
                    s[np * rank_r + n] = sum;
                }
        }
    }
}
