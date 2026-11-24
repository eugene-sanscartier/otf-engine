/* -*- c++ -*- ----------------------------------------------------------
   Standalone ETN potential — no LAMMPS dependency.
   Computes mlip-4's ETN (src/pots/etn/etn.cpp at 79a39bf), which has no
   LAMMPS pair style; hand-written passes replace mlip-4's program.
   Original author of mlip-4's ETN: Max Hodapp
------------------------------------------------------------------------- */

#include "pair_etn.h"

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

PairETN::PairETN(std::vector<int> species_ranks, std::vector<int> radial_func_counts, std::vector<int> radial_ranks, std::vector<std::vector<int>> bond_ranks, int species_count, const std::string& radial_basis_type, double min_dist, double max_dist, int radial_basis_size) : Pair(species_count, radial_basis_type, min_dist, max_dist, radial_basis_size), species_ranks(std::move(species_ranks)), radial_func_counts(std::move(radial_func_counts)), radial_ranks(std::move(radial_ranks)), bond_ranks(std::move(bond_ranks)) {
    prepare_cores();
    trained = false;
    reset_coeffs();
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
    virial.fill(0.0);

    grow_chunk_cache(list);

    // Loop over all provided neighbourhoods, a chunk of them at a time
    for (int ii_begin = 0; ii_begin < list.inum; ii_begin += chunk_size) {
        const int ii_end = std::min(ii_begin + chunk_size, list.inum);

        // Clear the input and bond vectors and their derivatives
        std::fill(input_vals.begin(), input_vals.end(), Lanes());
        std::fill(bond_vals.begin(), bond_vals.end(), Lanes());
        std::fill(input_energy_ders.begin(), input_energy_ders.end(), Lanes());
        std::fill(bond_energy_ders.begin(), bond_energy_ders.end(), Lanes());
        input_vals[0] = 1.0;
        bond_vals[bond_offsets[0]] = 1.0;

        for (int ii = ii_begin; ii < ii_end; ii++) {
            int valid_count = 0;
            const int i = list.ilist[ii];
            const int itype = list.types[i];
            const int jnum = list.numneigh[ii];
            const int* nbrs = list.firstneigh + nbr_offset;
            const double* dr = list.displacements + nbr_offset * 3;
            nbr_offset += jnum;

            select_atom_cache(ii - ii_begin);
            calc_neighbor_rows(jnum, dr);

            // ------------ Calculate the Input Vector ------------
            for (int jj = 0; jj < jnum; jj++) {
                const int j = nbrs[jj];
                const int jtype = list.types[j];
                const double r[3] = {dr[jj * 3 + 0], dr[jj * 3 + 1], dr[jj * 3 + 2]};
                const double rsq = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];

                if (rsq > max_cutoff_sq) continue;
                cached_j[valid_count] = j;
                valid_dr[valid_count] = {r[0], r[1], r[2]};

                const double dist = std::sqrt(rsq);
                const double inv_dist = 1.0 / dist;
                double* cache = neighbor_cache + (size_t) valid_count * cache_stride;
                cache[0] = inv_dist;
                double* ylm = cache + 1;
                double* ylm_grads = ylm + harmonic_count;
                double* mixed = ylm_grads + 3 * harmonic_count;
                double* mixed_ders = mixed + mixed_size;
                radial_basis->select_row(valid_count);
                const double* basis_vals = radial_basis->radial_basis_vals.data();
                const double* basis_ders = radial_basis->radial_basis_ders.data();
                harmonics.copy_row(valid_count, ylm, ylm_grads);

                // The radial channels of the species pair, and their products with the harmonics
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
                        Lanes* pv = input_vals.data() + input_shape.index(l, m) + shift;
                        for (int n = 0; n < rank_r; n++) pv[n][ii - ii_begin] += y * b[n];
                    }
                }
                valid_count++;
            }
            cached_counts[ii - ii_begin] = valid_count;
        }

        // ------------ Tensor Train, Forwards ------------
        for (int c = 0; c < core_count; c++) {
            const CovariantShape& left = bond_shapes[c];
            const CovariantShape& right = bond_shapes[c + 1];
            for (const CoreTerm& t : core_terms[c]) {
                const Core& core = t.imaginary ? cores_im[c] : cores[c];
                const Lanes* u = bond_vals.data() + bond_offsets[c] + t.in_part * left.size;
                Lanes* out = bond_vals.data() + bond_offsets[c + 1] + t.out_part * right.size;
                for (const CoreBlock& block : core.blocks) {
                    const int n1_count = left.ranks[block.l1], n2_count = input_shape.ranks[block.l2], n3_count = right.ranks[block.l3];
                    const int m1_stride = n3_count * n2_count;
                    Lanes* block_products = core_products.data() + t.products_offset + block.products_offset;

                    // The block contracted with the left bond, once per m1
                    for (int m1 = -block.l1; m1 <= block.l1; m1++) {
                        const Lanes* pu = u + left.index(block.l1, m1);
                        Lanes* products = block_products + (m1 + block.l1) * m1_stride;
                        for (int n3 = 0; n3 < n3_count; n3++)
                            for (int n2 = 0; n2 < n2_count; n2++) {
                                const double* c2 = core.coeffs.data() + block.base + n3 * block.stride3 + n2 * block.stride2;
                                Lanes prod;
                                for (int n1 = 0; n1 < n1_count; n1++) prod += c2[n1] * pu[n1];
                                products[n3 * n2_count + n2] = prod;
                            }
                    }

                    for (int e = block.w_begin; e < block.w_end; e++) {
                        const WignerEntry& w = wigner[e];
                        const Lanes* products = block_products + (w.m1 + block.l1) * m1_stride;
                        const Lanes* pv = input_vals.data() + input_shape.index(block.l2, w.m2);
                        Lanes* po = out + right.index(block.l3, w.m3);
                        for (int n3 = 0; n3 < n3_count; n3++) {
                            Lanes sum;
                            for (int n2 = 0; n2 < n2_count; n2++) sum += products[n3 * n2_count + n2] * pv[n2];
                            po[n3] += w.w * sum;
                        }
                    }
                }
            }
        }

        for (int ii = ii_begin; ii < ii_end; ii++) {
            const double nbh_energy = bond_vals[bond_offsets[core_count]][ii - ii_begin] + species_coeffs[list.types[list.ilist[ii]]];
            eatom[list.ilist[ii]] = nbh_energy;
            energy += nbh_energy;
        }

        // =========== Begin Backpropagation ===========
        bond_energy_ders[bond_offsets[core_count]] = 1.0;
        for (int c = core_count - 1; c >= 0; c--) {
            const CovariantShape& left = bond_shapes[c];
            const CovariantShape& right = bond_shapes[c + 1];
            for (const CoreTerm& t : core_terms[c]) {
                const Core& core = t.imaginary ? cores_im[c] : cores[c];
                const int in = bond_offsets[c] + t.in_part * left.size;
                const int out = bond_offsets[c + 1] + t.out_part * right.size;
                Lanes* u_ders = bond_energy_ders.data() + in;
                for (const CoreBlock& block : core.blocks) {
                    const int n1_count = left.ranks[block.l1], n2_count = input_shape.ranks[block.l2], n3_count = right.ranks[block.l3];
                    const int m1_stride = n3_count * n2_count;
                    const int m1_count = 2 * block.l1 + 1;
                    const Lanes* block_products = core_products.data() + t.products_offset + block.products_offset;

                    // The input's derivatives, and the products' summed per m1
                    std::fill(core_product_ders.begin(), core_product_ders.begin() + m1_count * m1_stride, Lanes());
                    for (int e = block.w_begin; e < block.w_end; e++) {
                        const WignerEntry& w = wigner[e];
                        const int iv = input_shape.index(block.l2, w.m2);
                        const Lanes* products = block_products + (w.m1 + block.l1) * m1_stride;
                        Lanes* product_ders = core_product_ders.data() + (w.m1 + block.l1) * m1_stride;
                        const Lanes* po = bond_energy_ders.data() + out + right.index(block.l3, w.m3);
                        for (int n3 = 0; n3 < n3_count; n3++) {
                            const Lanes g = w.w * po[n3];
                            for (int n2 = 0; n2 < n2_count; n2++) {
                                input_energy_ders[iv + n2] += g * products[n3 * n2_count + n2];
                                product_ders[n3 * n2_count + n2] += g * input_vals[iv + n2];
                            }
                        }
                    }

                    // The left bond's derivatives, once per m1
                    for (int m1 = -block.l1; m1 <= block.l1; m1++) {
                        const int iu = left.index(block.l1, m1);
                        const Lanes* product_ders = core_product_ders.data() + (m1 + block.l1) * m1_stride;
                        for (int n3 = 0; n3 < n3_count; n3++)
                            for (int n2 = 0; n2 < n2_count; n2++) {
                                const Lanes& d = product_ders[n3 * n2_count + n2];
                                const double* c2 = core.coeffs.data() + block.base + n3 * block.stride3 + n2 * block.stride2;
                                for (int n1 = 0; n1 < n1_count; n1++) u_ders[iu + n1] += d * c2[n1];
                            }
                    }
                }
            }
        }

        for (int ii = ii_begin; ii < ii_end; ii++) {
            const int i = list.ilist[ii];
            const int valid_count = cached_counts[ii - ii_begin];

            select_atom_cache(ii - ii_begin);

            //------------ Compute forces from the input vector's derivatives ------------
            for (int jj = 0; jj < valid_count; jj++) {
                const int j = cached_j[jj];
                const double* r = valid_dr[jj].data();
                const double* cache = neighbor_cache + (size_t) jj * cache_stride;
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
                        const Lanes* ders = input_energy_ders.data() + input_shape.index(l, m) + shift;
                        double ylm_der = 0, mixed_der = 0;
                        for (int n = 0; n < rank_r; n++) {
                            ylm_der += b[n] * ders[n][ii - ii_begin];
                            mixed_der += db[n] * ders[n][ii - ii_begin];
                        }
                        for (int a = 0; a < 3; a++) angular[a] += ylm_der * ylm_grads[index * 3 + a];
                        radial_force += ylm[index] * mixed_der;
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
}

void PairETN::get_coeffs(double* c) const {
    for (const auto* blocks : {&radial_basis_coeffs, &species_pair_coeffs})
        for (const auto& block : *blocks) c = std::copy(block.begin(), block.end(), c);
    for (const auto* part : {&cores, &cores_im})
        for (const Core& core : *part) c = std::copy(core.coeffs.begin(), core.coeffs.end(), c);
    std::copy(species_coeffs.begin(), species_coeffs.end(), c);
}

void PairETN::set_coeffs(const double* c) {
    for (auto* blocks : {&radial_basis_coeffs, &species_pair_coeffs})
        for (auto& block : *blocks) {
            std::copy(c, c + block.size(), block.begin());
            c += block.size();
        }
    for (auto* part : {&cores, &cores_im})
        for (Core& core : *part) {
            std::copy(c, c + core.coeffs.size(), core.coeffs.begin());
            c += core.coeffs.size();
        }
    std::copy(c, c + species_count, species_coeffs.begin());
    prepare_radial_mixing();
}

int PairETN::coeff_count() const {
    return species_columns + species_count;
}

void PairETN::set_radial_basis_coeffs(int l, const double* c) {
    std::copy(c, c + radial_basis_coeffs[l].size(), radial_basis_coeffs[l].begin());
    prepare_radial_mixing();
}

void PairETN::set_species_count(int n) {
    const int kept = std::min(n, species_count);
    for (size_t l = 0; l < species_pair_coeffs.size(); l++) {
        const int rank = species_ranks[l];
        std::vector<double> sized((size_t) n * n * rank, 0.0);
        for (int si = 0; si < kept; si++)
            for (int sj = 0; sj < kept; sj++)
                std::copy_n(species_pair_coeffs[l].begin() + (si * species_count + sj) * rank, rank, sized.begin() + (si * n + sj) * rank);
        species_pair_coeffs[l] = std::move(sized);
    }
    species_coeffs.resize(n, 0.0);
    species_count = n;
    prepare_columns();
    prepare_radial_mixing();
}

void PairETN::reset_coeffs() {
    std::vector<double> c(coeff_count());
    for (int k = 0; k < (int) c.size(); k++) c[k] = std::sin(10 + k * k);
    set_coeffs(c.data());
}

void PairETN::set_species_pair_coeffs(int l, const double* c) {
    std::copy(c, c + species_pair_coeffs[l].size(), species_pair_coeffs[l].begin());
    prepare_radial_mixing();
}

void PairETN::set_core_coeffs(int i, bool imaginary, const double* c) {
    std::vector<double>& block = (imaginary ? cores_im[i] : cores[i]).coeffs;
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

    // An untrained potential may omit the species count, which training then sets
    read_header(tfr, "ETN", "1.0.0");

    read_radial_basis(tfr);

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

    // An untrained potential may omit every coefficient, and takes reset_coeffs' values
    if (!tfr.next_line() || tfr.keyword() != "radial_coeffs") {
        trained = false;
        reset_coeffs();
        return;
    }
    if (species_count == 0)
        throw std::runtime_error("PairETN: a trained ETN file must give its species count");

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
        read_block(radial_basis_coeffs[l], "radial_coeffs of l = " + std::to_string(l));
    tfr.expect("species_pair_coeffs");
    for (int l = 0; l < l_count; l++)
        read_block(species_pair_coeffs[l], "species_pair_coeffs of l = " + std::to_string(l));
    tfr.expect("core_coeffs");
    for (int c = 0; c < core_count; c++)
        read_block(cores[c].coeffs, "core_coeffs of core " + std::to_string(c));
    if (has_imaginary_cores()) {
        tfr.expect("core_coeffs_im");
        for (int c = 1; c < core_count - 1; c++)
            read_block(cores_im[c].coeffs, "core_coeffs_im of core " + std::to_string(c));
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
    if (bond_ranks.size() < 2)
        throw std::runtime_error("PairETN: an ETN needs at least one core, two bond ranks");
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

    radial_basis_coeffs.resize(l_count);
    species_pair_coeffs.resize(l_count);
    for (int l = 0; l < l_count; l++) {
        radial_basis_coeffs[l].resize(species_ranks[l] * radial_func_counts[l] * radial_ranks[l]);
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
    int products_size = 0;
    for (int c = 0; c < core_count; c++) {
        core_terms[c].push_back({false, 0, 0, 0});
        if (bond_has_im(c) && bond_has_im(c + 1)) core_terms[c].push_back({false, 1, 1, 0});
        if (core_has_im(c) && bond_has_im(c)) core_terms[c].push_back({true, 1, 0, 0});
        if (core_has_im(c) && bond_has_im(c + 1)) core_terms[c].push_back({true, 0, 1, 0});
        for (CoreTerm& t : core_terms[c]) {
            t.products_offset = products_size;
            products_size += (t.imaginary ? cores_im[c] : cores[c]).products_size;
        }
    }

    species_coeffs.resize(species_count);
    prepare_columns();

    cache_stride = 1 + 4 * l_count * l_count + 2 * mixed_offsets[l_count];
    size_t block_products_size = 0;
    for (int c = 0; c < core_count; c++)
        for (const Core* core : {&cores[c], &cores_im[c]})
            for (const CoreBlock& block : core->blocks)
                block_products_size = std::max(block_products_size, (size_t) (2 * block.l1 + 1) * bond_shapes[c + 1].ranks[block.l3] * input_shape.ranks[block.l2]);

    input_vals.resize(input_shape.size);
    bond_vals.resize(bonds_size);
    input_energy_ders.resize(input_shape.size);
    bond_energy_ders.resize(bonds_size);
    core_products.resize(products_size);
    core_product_ders.resize(block_products_size);

    atom_input_vals.resize(input_shape.size);
    atom_bond_vals.resize(bonds_size);
    atom_input_energy_ders.resize(input_shape.size);
    atom_bond_energy_ders.resize(bonds_size);
    atom_core_products.resize(products_size);
    atom_core_product_ders.resize(block_products_size);
}

// The first column of each coefficient block in get_coeffs' order, for the current shape and species count.
void PairETN::prepare_columns() {
    const int l_count = (int) species_ranks.size();
    const int core_count = get_core_count();
    int column = 0;

    radial_columns.resize(l_count);
    for (int l = 0; l < l_count; l++) radial_columns[l] = column, column += (int) radial_basis_coeffs[l].size();
    species_pair_columns.resize(l_count);
    for (int l = 0; l < l_count; l++) species_pair_columns[l] = column, column += (int) species_pair_coeffs[l].size();
    core_columns.resize(core_count);
    for (int c = 0; c < core_count; c++) core_columns[c] = column, column += (int) cores[c].coeffs.size();
    core_im_columns.resize(core_count);
    for (int c = 0; c < core_count; c++) core_im_columns[c] = column, column += (int) cores_im[c].coeffs.size();
    species_columns = column;
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
                    core.blocks.push_back({l1, l2, l3, base3 + base2 + base1, stride2, stride3, w_begin, (int) wigner.size(), core.products_size});
                    core.products_size += (2 * l1 + 1) * right[l3] * middle[l2];
                }
                base1 += left[l1];
            }
            base2 += middle[l2] * stride2;
        }
        base3 += right[l3] * stride3;
    }
    core.coeffs.assign(base3, 0.0);
    return core;
}

// S_l[s_i, s_j](n', n) = sum_k A_l[s_i, s_j, k] B_l[k, n', n], as mlip-4 contracts the species first.
void PairETN::prepare_radial_mixing() {
    const int l_count = (int) species_ranks.size();
    radial_mixing.resize(l_count);
    for (int l = 0; l < l_count; l++) {
        const int rank_s = species_ranks[l], n_rad = radial_func_counts[l], rank_r = radial_ranks[l];
        const double* radial = radial_basis_coeffs[l].data();
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
