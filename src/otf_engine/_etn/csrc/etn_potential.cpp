/* -*- c++ -*- ----------------------------------------------------------
   Equivariant tensor network potential: the computation of mlip-4's ETN.
------------------------------------------------------------------------- */

#include "etn_potential.h"
#include "text_file_reader.h"

#include <algorithm>
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

ETN::ETN(const std::string& filename) {
    std::ifstream f(filename);
    if (!f)
        throw std::runtime_error("ETN: cannot open potential file " + filename);
    read_file(f);
}

ETN::ETN(int species_count, std::vector<int> species_ranks, std::vector<int> radial_func_counts, std::vector<int> radial_ranks, std::vector<std::vector<int>> bond_ranks, int radial_basis_size, double min_dist, double max_dist)
    : species_count(species_count), radial_basis(radial_basis_size, min_dist, max_dist), species_ranks(std::move(species_ranks)), radial_func_counts(std::move(radial_func_counts)),
      radial_ranks(std::move(radial_ranks)), bond_ranks(std::move(bond_ranks)) {
    build();
}

void ETN::set_coeffs(const double* c) {
    std::copy(c, c + coeffs.size(), coeffs.begin());
    prepare_radial_mixing();
}

/* ----------------------------------------------------------------------
   Layout of the coefficients, the bonds and the scratch, from the shape.
   The coefficients are mlip-4's params: B_l, A_l, the real cores, the
   imaginary cores 1 .. d-2 when d > 3, the species energies.
------------------------------------------------------------------------- */
void ETN::build() {
    const size_t l_count = species_ranks.size();
    if (species_count < 1)
        throw std::runtime_error("ETN: species_count must be positive");
    if (l_count == 0 || radial_func_counts.size() != l_count || radial_ranks.size() != l_count)
        throw std::runtime_error("ETN: species_ranks, radial_funcs_counts and radial_ranks must have the same positive length");
    for (const auto* ranks : {&species_ranks, &radial_func_counts, &radial_ranks})
        if (std::all_of(ranks->begin(), ranks->end(), [](int r) { return r == 0; }) || std::any_of(ranks->begin(), ranks->end(), [](int r) { return r < 0; }))
            throw std::runtime_error("ETN: a size as a function of l must be non-negative and not all zero");
    for (int n : radial_func_counts)
        if (n > radial_basis.size)
            throw std::runtime_error("ETN: radial_funcs_counts exceeds radial_basis_size");
    if (bond_ranks.size() < 2)
        throw std::runtime_error("ETN: a tensor train needs at least one core");
    if (bond_ranks.front() != std::vector<int>{1} || bond_ranks.back() != std::vector<int>{1})
        throw std::runtime_error("ETN: the first and last bond ranks must be {1}");
    for (const auto& ranks : bond_ranks)
        if (ranks.empty() || std::all_of(ranks.begin(), ranks.end(), [](int r) { return r == 0; }) || std::any_of(ranks.begin(), ranks.end(), [](int r) { return r < 0; }))
            throw std::runtime_error("ETN: a bond rank as a function of l must be non-negative and not all zero");

    harmonics = RealHarmonics((int) l_count);

    // The input carries the constant 1 ahead of the l = 0 radial channels
    std::vector<int> input_ranks = radial_ranks;
    input_ranks[0] += 1;
    input_shape = CovariantShape(input_ranks);

    bond_shapes.clear();
    bond_offsets.clear();
    bonds_size = 0;
    for (const auto& ranks : bond_ranks) {
        bond_shapes.emplace_back(ranks);
        bond_offsets.push_back(bonds_size);
        bonds_size += 2 * bond_shapes.back().size;
    }

    int offset = 0;
    radial_offsets.resize(l_count);
    for (size_t l = 0; l < l_count; l++) {
        radial_offsets[l] = offset;
        offset += species_ranks[l] * radial_func_counts[l] * radial_ranks[l];
    }
    species_pair_offsets.resize(l_count);
    for (size_t l = 0; l < l_count; l++) {
        species_pair_offsets[l] = offset;
        offset += species_count * species_count * species_ranks[l];
    }

    const int core_count = get_core_count();
    wigner.clear();
    cores_re.clear();
    cores_im.assign(core_count, Core());
    for (int i = 0; i < core_count; i++)
        cores_re.push_back(build_core(i, false, offset));
    if (has_imaginary_cores())
        for (int i = 1; i < core_count - 1; i++)
            cores_im[i] = build_core(i, true, offset);

    offset += species_count;
    coeffs.assign(offset, 0.0);

    radial_vals.resize(radial_basis.size);
    radial_ders.resize(radial_basis.size);
    input_adj.resize(input_shape.size);
    bond_adj.resize(bonds_size);
    int mixed_size = 0;
    for (int r : radial_ranks) mixed_size += r;
    mixed_adj.resize(mixed_size);

    prepare_radial_mixing();
}

// The coefficients of core i ordered l3, n3, l2, n2, l1, n1 with n1 fastest, over the
// (l1, l2, l3) that pass the triangle rule with an even sum (real part) or an odd one.
Core ETN::build_core(int i, bool imaginary, int& offset) {
    const std::vector<int>& left = bond_shapes[i].ranks;
    const std::vector<int>& middle = input_shape.ranks;
    const std::vector<int>& right = bond_shapes[i + 1].ranks;
    const int parity = imaginary ? 1 : 0;
    auto allowed = [parity](int l1, int l2, int l3) {
        const int sum = l1 + l2 + l3;
        return 2 * std::max({l1, l2, l3}) <= sum && sum % 2 == parity;
    };

    Core core;
    core.coeff_offset = offset;
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
    offset += base3;
    return core;
}

// S_l[s_i, s_j](n', n) = sum_k A_l[s_i, s_j, k] B_l[k, n', n], as mlip-4 contracts the species first.
void ETN::prepare_radial_mixing() {
    const int l_count = (int) species_ranks.size();
    radial_mixing.resize(l_count);
    for (int l = 0; l < l_count; l++) {
        const int rank_s = species_ranks[l], n_rad = radial_func_counts[l], rank_r = radial_ranks[l];
        const double* species_pair = coeffs.data() + species_pair_offsets[l];
        const double* radial = coeffs.data() + radial_offsets[l];
        std::vector<double>& mixing = radial_mixing[l];
        mixing.assign((size_t) species_count * species_count * n_rad * rank_r, 0.0);

        for (int pair = 0; pair < species_count * species_count; pair++) {
            const double* a = species_pair + pair * rank_s;
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

/* ----------------------------------------------------------------------
   One core applied to (u, v), added into out, and its adjoint.
------------------------------------------------------------------------- */
void ETN::contract(const Core& core, const CovariantShape& left, const CovariantShape& right, const double* u, const double* v, double* out) const {
    const double* c = coeffs.data() + core.coeff_offset;
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

void ETN::contract_adjoint(const Core& core, const CovariantShape& left, const CovariantShape& right, const double* u, const double* v, const double* out_adj, double* u_adj, double* v_adj) const {
    const double* c = coeffs.data() + core.coeff_offset;
    for (const CoreBlock& block : core.blocks) {
        const int n1_count = left.ranks[block.l1], n2_count = input_shape.ranks[block.l2], n3_count = right.ranks[block.l3];
        for (int e = block.w_begin; e < block.w_end; e++) {
            const WignerEntry& w = wigner[e];
            const int iu = left.index(block.l1, w.m1), iv = input_shape.index(block.l2, w.m2);
            const double* po = out_adj + right.index(block.l3, w.m3);
            for (int n3 = 0; n3 < n3_count; n3++) {
                const double g = w.w * po[n3];
                if (g == 0.0) continue;
                const double* c3 = c + block.base + n3 * block.stride3;
                for (int n2 = 0; n2 < n2_count; n2++) {
                    const double* c2 = c3 + n2 * block.stride2;
                    const double gv = g * v[iv + n2];
                    double t = 0;
                    for (int n1 = 0; n1 < n1_count; n1++) {
                        t += c2[n1] * u[iu + n1];
                        u_adj[iu + n1] += gv * c2[n1];
                    }
                    v_adj[iv + n2] += g * t;
                }
            }
        }
    }
}

/* ----------------------------------------------------------------------
   Main computation: per central atom the input vector, the tensor train
   forwards, then backwards to the input and through it to the neighbors.
------------------------------------------------------------------------- */
void ETN::compute(const NeighList& list) {
    const int l_count = (int) species_ranks.size();
    const int core_count = get_core_count();
    const double max_dist_sq = radial_basis.max_dist * radial_basis.max_dist;
    const double* species_energies = coeffs.data() + species_coeffs_offset();

    std::vector<int> mixed_offsets(l_count + 1, 0);
    for (int l = 0; l < l_count; l++) mixed_offsets[l + 1] = mixed_offsets[l] + radial_ranks[l];
    const int mixed_size = mixed_offsets[l_count];
    const int harmonic_count = l_count * l_count;
    const int stride = 4 + 4 * harmonic_count + 2 * mixed_size;

    // The imaginary part of bond b exists for 2 <= b <= d-2, of core i for 1 <= i <= d-2, when d > 3
    auto bond_has_im = [&](int b) { return has_imaginary_cores() && b >= 2 && b <= core_count - 2; };
    auto core_has_im = [&](int i) { return has_imaginary_cores() && i >= 1 && i <= core_count - 2; };

    energy = 0.0;
    eatom.assign(list.n_atoms, 0.0);
    forces.assign((size_t) list.n_atoms * 3, 0.0);
    std::fill(virial, virial + 6, 0.0);
    inputs.assign((size_t) list.inum * input_shape.size, 0.0);
    bonds.assign((size_t) list.inum * bonds_size, 0.0);

    int nbr_offset = 0;
    for (int ii = 0; ii < list.inum; ii++) {
        const int i = list.ilist[ii];
        const int itype = list.types[i];
        const int jnum = list.numneigh[ii];
        const int* nbrs = list.firstneigh + nbr_offset;
        const double* dr = list.displacements + nbr_offset * 3;
        nbr_offset += jnum;

        if (neighbor_cache.size() < (size_t) jnum * stride) neighbor_cache.resize((size_t) jnum * stride);
        std::vector<int> cached_j;
        cached_j.reserve(jnum);

        // ------------ Input vector ------------
        double* v = inputs.data() + (size_t) ii * input_shape.size;
        v[0] = 1.0;
        for (int jj = 0; jj < jnum; jj++) {
            const double r[3] = {dr[jj * 3 + 0], dr[jj * 3 + 1], dr[jj * 3 + 2]};
            const double rsq = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
            if (rsq >= max_dist_sq) continue;
            const int jtype = list.types[nbrs[jj]];

            double* cache = neighbor_cache.data() + cached_j.size() * stride;
            cached_j.push_back(nbrs[jj]);
            const double dist = std::sqrt(rsq);
            const double inv_dist = 1.0 / dist;
            const double u[3] = {r[0] * inv_dist, r[1] * inv_dist, r[2] * inv_dist};
            radial_basis.calc(dist, radial_vals.data(), radial_ders.data());
            harmonics.compute(u);

            std::copy(r, r + 3, cache);
            cache[3] = inv_dist;
            double* ylm = cache + 4;
            double* ylm_grads = ylm + harmonic_count;
            double* mixed = ylm_grads + 3 * harmonic_count;
            double* mixed_ders = mixed + mixed_size;
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
                        val += radial_vals[np] * s[np * rank_r + n];
                        der += radial_ders[np] * s[np * rank_r + n];
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

        // ------------ Tensor train, forwards ------------
        double* bond = bonds.data() + (size_t) ii * bonds_size;
        bond[bond_offsets[0]] = 1.0;
        for (int c = 0; c < core_count; c++) {
            const CovariantShape& left = bond_shapes[c];
            const CovariantShape& right = bond_shapes[c + 1];
            const double* in_re = bond + bond_offsets[c];
            const double* in_im = in_re + left.size;
            double* out_re = bond + bond_offsets[c + 1];
            double* out_im = out_re + right.size;

            contract(cores_re[c], left, right, in_re, v, out_re);
            if (bond_has_im(c) && bond_has_im(c + 1)) contract(cores_re[c], left, right, in_im, v, out_im);
            if (core_has_im(c)) {
                if (bond_has_im(c)) contract(cores_im[c], left, right, in_im, v, out_re);
                if (bond_has_im(c + 1)) contract(cores_im[c], left, right, in_re, v, out_im);
            }
        }

        const double site_energy = bond[bond_offsets[core_count]] + species_energies[itype];
        eatom[i] += site_energy;
        energy += site_energy;

        // ------------ Tensor train, backwards ------------
        std::fill(bond_adj.begin(), bond_adj.end(), 0.0);
        std::fill(input_adj.begin(), input_adj.end(), 0.0);
        bond_adj[bond_offsets[core_count]] = 1.0;
        for (int c = core_count - 1; c >= 0; c--) {
            const CovariantShape& left = bond_shapes[c];
            const CovariantShape& right = bond_shapes[c + 1];
            const double* in_re = bond + bond_offsets[c];
            const double* in_im = in_re + left.size;
            double* in_re_adj = bond_adj.data() + bond_offsets[c];
            double* in_im_adj = in_re_adj + left.size;
            const double* out_re_adj = bond_adj.data() + bond_offsets[c + 1];
            const double* out_im_adj = out_re_adj + right.size;

            contract_adjoint(cores_re[c], left, right, in_re, v, out_re_adj, in_re_adj, input_adj.data());
            if (bond_has_im(c) && bond_has_im(c + 1)) contract_adjoint(cores_re[c], left, right, in_im, v, out_im_adj, in_im_adj, input_adj.data());
            if (core_has_im(c)) {
                if (bond_has_im(c)) contract_adjoint(cores_im[c], left, right, in_im, v, out_re_adj, in_im_adj, input_adj.data());
                if (bond_has_im(c + 1)) contract_adjoint(cores_im[c], left, right, in_re, v, out_im_adj, in_re_adj, input_adj.data());
            }
        }

        // ------------ Through the input to the neighbors ------------
        for (size_t jj = 0; jj < cached_j.size(); jj++) {
            const int j = cached_j[jj];
            const double* cache = neighbor_cache.data() + jj * stride;
            const double* r = cache;
            const double inv_dist = cache[3];
            const double* ylm = cache + 4;
            const double* ylm_grads = ylm + harmonic_count;
            const double* mixed = ylm_grads + 3 * harmonic_count;
            const double* mixed_ders = mixed + mixed_size;

            double angular[3] = {0, 0, 0};
            double radial = 0;
            for (int l = 0; l < l_count; l++) {
                const int rank_r = radial_ranks[l];
                const double* b = mixed + mixed_offsets[l];
                const double* db = mixed_ders + mixed_offsets[l];
                const int shift = l == 0 ? 1 : 0;
                for (int m = -l; m <= l; m++) {
                    const int index = l * l + l + m;
                    const double* adj = input_adj.data() + input_shape.index(l, m) + shift;
                    double y_adj = 0, b_adj = 0;
                    for (int n = 0; n < rank_r; n++) {
                        y_adj += b[n] * adj[n];
                        b_adj += db[n] * adj[n];
                    }
                    for (int a = 0; a < 3; a++) angular[a] += y_adj * ylm_grads[index * 3 + a];
                    radial += ylm[index] * b_adj;
                }
            }

            // dE_i / dr_ij, with r_ij = r_j - r_i
            double temp_force[3];
            for (int a = 0; a < 3; a++) temp_force[a] = inv_dist * angular[a] + radial * r[a] * inv_dist;

            forces[i * 3 + 0] += temp_force[0];
            forces[i * 3 + 1] += temp_force[1];
            forces[i * 3 + 2] += temp_force[2];

            forces[j * 3 + 0] -= temp_force[0];
            forces[j * 3 + 1] -= temp_force[1];
            forces[j * 3 + 2] -= temp_force[2];

            // Virial tallied as PairMTP tallies it
            virial[0] -= r[0] * temp_force[0];
            virial[1] -= r[1] * temp_force[1];
            virial[2] -= r[2] * temp_force[2];
            virial[3] -= r[0] * temp_force[1];
            virial[4] -= r[0] * temp_force[2];
            virial[5] -= r[1] * temp_force[2];
        }
    }
}

/* ----------------------------------------------------------------------
   .etn file parsing: the shape, then the coefficients block by block.
------------------------------------------------------------------------- */
template <class T>
static std::vector<T> parse_values(std::istringstream ss) {
    std::vector<T> values;
    T x;
    while (ss >> x) values.push_back(x);
    return values;
}

void ETN::read_file(std::istream& is) {
    TextFileReader tfr(is);

    tfr.expect("ETN");
    if (tfr.next_string("version") != "1.0.0")
        throw std::runtime_error("ETN: ETN file must have version \"1.0.0\"");

    tfr.advance();
    if (tfr.keyword() == "potential_name") {
        tfr.rest() >> potential_name;
        tfr.advance();
    }
    if (tfr.keyword() != "species_count")
        throw std::runtime_error("ETN: expected 'species_count', got '" + tfr.keyword() + "'");
    tfr.rest() >> species_count;

    const double scaling = tfr.next_double("scaling");
    if (tfr.next_string("radial_basis_type") != "RadialBasisCinf")
        throw std::runtime_error("ETN: the radial basis type must be RadialBasisCinf");
    const double min_dist = tfr.next_double("min_dist");
    const double max_dist = tfr.next_double("max_dist");
    const int radial_basis_size = tfr.next_int("radial_basis_size");
    radial_basis = RadialBasisCinf(radial_basis_size, min_dist, max_dist);
    radial_basis.scaling = scaling;

    tfr.expect("species_ranks");
    species_ranks = parse_values<int>(tfr.rest());
    tfr.expect("radial_funcs_counts");
    radial_func_counts = parse_values<int>(tfr.rest());
    tfr.expect("radial_ranks");
    radial_ranks = parse_values<int>(tfr.rest());

    const int core_count = tfr.next_int("core_count");
    if (core_count < 1)
        throw std::runtime_error("ETN: core_count must be positive");
    tfr.expect("bond_ranks");
    bond_ranks.clear();
    for (int b = 0; b <= core_count; b++) {
        tfr.advance();
        bond_ranks.push_back(parse_values<int>(std::istringstream(tfr.line())));
    }

    build();

    // One line of values per block, each of the length the shape gives it
    auto read_block = [&](int offset, int count, const std::string& what) {
        tfr.advance();
        const std::vector<double> values = parse_values<double>(std::istringstream(tfr.line()));
        if ((int) values.size() != count)
            throw std::runtime_error("ETN: " + what + " has " + std::to_string(values.size()) + " values, the shape gives " + std::to_string(count));
        std::copy(values.begin(), values.end(), coeffs.begin() + offset);
    };

    const int l_count = (int) species_ranks.size();
    tfr.expect("radial_coeffs");
    for (int l = 0; l < l_count; l++)
        read_block(radial_offsets[l], species_ranks[l] * radial_func_counts[l] * radial_ranks[l], "radial_coeffs of l = " + std::to_string(l));
    tfr.expect("species_pair_coeffs");
    for (int l = 0; l < l_count; l++)
        read_block(species_pair_offsets[l], species_count * species_count * species_ranks[l], "species_pair_coeffs of l = " + std::to_string(l));
    tfr.expect("core_coeffs");
    for (int i = 0; i < core_count; i++)
        read_block(cores_re[i].coeff_offset, cores_re[i].coeff_count, "core_coeffs of core " + std::to_string(i));
    if (has_imaginary_cores()) {
        tfr.expect("core_coeffs_im");
        for (int i = 1; i < core_count - 1; i++)
            read_block(cores_im[i].coeff_offset, cores_im[i].coeff_count, "core_coeffs_im of core " + std::to_string(i));
    }

    tfr.expect("species_coeffs");
    const std::vector<double> energies = parse_values<double>(tfr.rest());
    if ((int) energies.size() != species_count)
        throw std::runtime_error("ETN: species_coeffs must hold species_count values");
    std::copy(energies.begin(), energies.end(), coeffs.begin() + species_coeffs_offset());

    prepare_radial_mixing();
}
