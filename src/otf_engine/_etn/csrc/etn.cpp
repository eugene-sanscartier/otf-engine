/* -*- c++ -*- ----------------------------------------------------------
   Input and bond vectors of an ETN, and derivatives of its site energies,
   and of a loss built on its energy, forces and virial, with respect to
   its coefficients.
------------------------------------------------------------------------- */

#include "etn.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

/* ----------------------------------------------------------------------
   Input and bond vectors per central atom
------------------------------------------------------------------------- */
void ETN::eval_bonds(const NeighList& list, double* inputs_out, double* bonds_out) {
    const int l_count = (int) species_ranks.size();
    const int core_count = get_core_count();
    int nbr_offset = 0;

    for (int ii = 0; ii < list.inum; ii++) {
        const int i = list.ilist[ii];
        const int itype = list.types[i];
        const int jnum = list.numneigh[ii];
        const int* nbrs = list.firstneigh + nbr_offset;
        const double* dr = list.displacements + nbr_offset * 3;
        nbr_offset += jnum;

        std::fill(atom_input_vals.begin(), atom_input_vals.end(), 0.0);
        std::fill(atom_bond_vals.begin(), atom_bond_vals.end(), 0.0);
        atom_input_vals[0] = 1.0;
        atom_bond_vals[bond_offsets[0]] = 1.0;
        int valid_count = 0;
        calc_neighbor_rows(jnum, dr);

        // ------------ Calculate the Input Vector ------------
        for (int jj = 0; jj < jnum; jj++) {
            const int j = nbrs[jj];
            const int jtype = list.types[j];
            const double r[3] = {dr[jj * 3 + 0], dr[jj * 3 + 1], dr[jj * 3 + 2]};
            const double rsq = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];

            if (rsq > max_cutoff_sq) continue;

            radial_basis->select_row(valid_count);
            const double* basis_vals = radial_basis->radial_basis_vals.data();
            harmonics.select_row(valid_count);
            const double* ylm = harmonics.vals.data();
            valid_count++;

            for (int l = 0; l < l_count; l++) {
                const int n_rad = radial_func_counts[l], rank_r = radial_ranks[l];
                const double* s = radial_mixing[l].data() + (size_t) (itype * species_count + jtype) * n_rad * rank_r;
                const int shift = l == 0 ? 1 : 0;
                for (int n = 0; n < rank_r; n++) {
                    double val = 0;
                    for (int np = 0; np < n_rad; np++) val += basis_vals[np] * s[np * rank_r + n];
                    for (int m = -l; m <= l; m++) atom_input_vals[input_shape.index(l, m) + shift + n] += ylm[l * l + l + m] * val;
                }
            }
        }

        // ------------ Tensor Train, Forwards ------------
        for (int c = 0; c < core_count; c++) {
            const CovariantShape& left = bond_shapes[c];
            const CovariantShape& right = bond_shapes[c + 1];
            for (const CoreTerm& t : core_terms[c]) {
                const Core& core = t.imaginary ? cores_im[c] : cores[c];
                const double* u = atom_bond_vals.data() + bond_offsets[c] + t.in_part * left.size;
                double* out = atom_bond_vals.data() + bond_offsets[c + 1] + t.out_part * right.size;
                for (const CoreBlock& block : core.blocks) {
                    const int n1_count = left.ranks[block.l1], n2_count = input_shape.ranks[block.l2], n3_count = right.ranks[block.l3];
                    const int m1_stride = n3_count * n2_count;
                    double* block_products = atom_core_products.data() + t.products_offset + block.products_offset;

                    // The block contracted with the left bond, once per m1
                    for (int m1 = -block.l1; m1 <= block.l1; m1++) {
                        const double* pu = u + left.index(block.l1, m1);
                        double* products = block_products + (m1 + block.l1) * m1_stride;
                        for (int n3 = 0; n3 < n3_count; n3++)
                            for (int n2 = 0; n2 < n2_count; n2++) {
                                const double* c2 = core.coeffs.data() + block.base + n3 * block.stride3 + n2 * block.stride2;
                                double prod = 0;
                                for (int n1 = 0; n1 < n1_count; n1++) prod += c2[n1] * pu[n1];
                                products[n3 * n2_count + n2] = prod;
                            }
                    }

                    for (int e = block.w_begin; e < block.w_end; e++) {
                        const WignerEntry& w = wigner[e];
                        const double* products = block_products + (w.m1 + block.l1) * m1_stride;
                        const double* pv = atom_input_vals.data() + input_shape.index(block.l2, w.m2);
                        double* po = out + right.index(block.l3, w.m3);
                        for (int n3 = 0; n3 < n3_count; n3++) {
                            double sum = 0;
                            for (int n2 = 0; n2 < n2_count; n2++) sum += products[n3 * n2_count + n2] * pv[n2];
                            po[n3] += w.w * sum;
                        }
                    }
                }
            }
        }

        std::copy(atom_input_vals.begin(), atom_input_vals.end(), inputs_out + (size_t) ii * input_shape.size);
        std::copy(atom_bond_vals.begin(), atom_bond_vals.end(), bonds_out + (size_t) ii * bonds_size);
    }
}

/* ---------------------------------------------------------------------- */
void ETN::grow_neighbor_rows(int jnum) {
    if (rows_size >= jnum) return;
    rows_size = jnum;
    const size_t harmonic_count = species_ranks.size() * species_ranks.size();
    neighbor_harmonics.resize(rows_size * harmonic_count);
    neighbor_radial_vals.resize((size_t) rows_size * radial_basis_size);
    neighbor_radial_ders.resize((size_t) rows_size * radial_basis_size);
    neighbor_harmonic_tangents.resize(rows_size * harmonic_count);
    neighbor_dist_tangents.resize(rows_size);
}

/* ----------------------------------------------------------------------
   The site-energy gradient takes the cores' columns from the backwards
   pass through the tensor train, and the radial and species-pair columns
   from dE_i/dv: each neighbor's channel b_n = sum_n' basis_n' S(n', n),
   with S_l = A_l B_l for its species pair.

   Without the nonlinear columns, a last-core column's site energy is
   bilinear in the last bond and the input vector, and its force gradient
   is PairETN::compute's force pass on its derivative w.r.t. the input
   vector. One backwards pass carries every column's seed, a value per
   seed, as compute's carries a chunk's atoms, and the passes of A atoms
   run together, the atom's lane innermost. The last core has no
   imaginary part.
------------------------------------------------------------------------- */
template <int A>
void ETN::eval_grad_lanes(const NeighList& list, double* site_e_grad, double* force_grad, double* virial_grad, bool nonlinear) {
    if (nonlinear && force_grad)
        throw std::runtime_error("eval_grad: an ETN has force gradients for its linear columns only");

    const int width = coeff_count();
    const int l_count = (int) species_ranks.size();
    const int core_count = get_core_count();
    const int harmonic_count = l_count * l_count;
    const int mixed_size = mixed_offsets[l_count];
    const int last = core_count - 1;
    const int column_count = (int) cores[last].coeffs.size();
    const bool forces = force_grad || virial_grad;

    std::fill(site_e_grad, site_e_grad + (size_t) list.inum * width, 0.0);
    if (force_grad) std::fill(force_grad, force_grad + (size_t) list.n_atoms * 3 * width, 0.0);
    if (virial_grad) std::fill(virial_grad, virial_grad + 6 * width, 0.0);
    // The last core's columns, numbered by block, n3, n1 then n2, and the backwards passes that carry them, numbered by
    // block, n1 then pass, so that a channel (l1, n1) of the last bond seeds contiguous passes. A column has a pass of its
    // own, unless its block has fewer unit seeds, one per m1, than columns per n1: then it combines its n1's unit passes.
    // The last core has one term, real on the real last bond, since the last bond has no imaginary part.
    struct LastBlock {
        int first_column, first_pass, first_unit_coef;
        bool unit;
    };
    std::vector<LastBlock> last_blocks;
    std::vector<int> column_coeffs;                  // each column's coefficient within the core
    std::vector<int> column_pass;                    // each column's pass, or -1 when it combines unit passes
    std::vector<std::pair<int, int>> entry_passes;   // [first, end) of the passes each entry of the last bond seeds, from its offset
    std::vector<double> unit_coefs;                  // per column of a unit block and m1, its seed on that m1's unit pass
    int pass_count = 0;
    if (!nonlinear) {
        const CovariantShape& left = bond_shapes[last];
        const CovariantShape& right = bond_shapes[core_count];
        entry_passes.assign(left.size, {std::numeric_limits<int>::max(), 0});
        int unit_coef_count = 0;
        for (const CoreBlock& block : cores[last].blocks) {
            const int n1_count = left.ranks[block.l1], n2_count = input_shape.ranks[block.l2], n3_count = right.ranks[block.l3];
            const int m1_count = 2 * block.l1 + 1;
            const bool unit = m1_count < n3_count * n2_count;
            const int passes_per_n1 = unit ? m1_count : n3_count * n2_count;
            last_blocks.push_back({(int) column_coeffs.size(), pass_count, unit_coef_count, unit});
            for (int n3 = 0; n3 < n3_count; n3++)
                for (int n1 = 0; n1 < n1_count; n1++)
                    for (int n2 = 0; n2 < n2_count; n2++) {
                        column_coeffs.push_back(block.base + n3 * block.stride3 + n2 * block.stride2 + n1);
                        column_pass.push_back(unit ? -1 : pass_count + n1 * passes_per_n1 + n3 * n2_count + n2);
                    }
            for (int m1 = -block.l1; m1 <= block.l1; m1++)
                for (int n1 = 0; n1 < n1_count; n1++) {
                    std::pair<int, int>& range = entry_passes[left.index(block.l1, m1) + n1];
                    const int begin = pass_count + n1 * passes_per_n1;
                    range = {std::min(range.first, begin), std::max(range.second, begin + passes_per_n1)};
                }
            pass_count += n1_count * passes_per_n1;
            if (unit) unit_coef_count += n3_count * n1_count * n2_count * m1_count;
        }
        unit_coefs.resize((size_t) unit_coef_count * A);
        input_pass_ders.resize((size_t) input_shape.size * pass_count * A);
        bond_pass_ders.resize((size_t) bonds_size * pass_count * A);
        product_pass_ders.resize(atom_core_product_ders.size() * pass_count * A);
        input_column_ders.resize((size_t) input_shape.size * column_count);
        column_pair_ders.resize(4 * (size_t) column_count);
    }

    int nbr_offset = 0;

    // A atoms per chunk: the linear path runs their backwards passes on a lane per atom
    grow_chunk_cache(list);
    if (!nonlinear) {
        lane_input_vals.resize((size_t) input_shape.size * A);
        lane_core_products.resize(atom_core_products.size() * A);
        lane_column_ders.resize((size_t) input_shape.size * column_count * A);
    }

    for (int ii_begin = 0; ii_begin < list.inum; ii_begin += A) {
        const int ii_end = std::min(ii_begin + A, list.inum);
        if (!nonlinear) {
            std::fill(input_pass_ders.begin(), input_pass_ders.end(), 0.0);
            std::fill(bond_pass_ders.begin(), bond_pass_ders.end(), 0.0);
            std::fill(lane_column_ders.begin(), lane_column_ders.end(), 0.0);
            std::fill(unit_coefs.begin(), unit_coefs.end(), 0.0);
            std::fill(lane_input_vals.begin(), lane_input_vals.end(), 0.0);
            std::fill(lane_core_products.begin(), lane_core_products.end(), 0.0);
        }

        for (int ii = ii_begin; ii < ii_end; ii++) {
            const int a = ii - ii_begin;
            const int i = list.ilist[ii];
            const int itype = list.types[i];
            const int jnum = list.numneigh[ii];
            const int* nbrs = list.firstneigh + nbr_offset;
            const double* dr = list.displacements + nbr_offset * 3;
            nbr_offset += jnum;

            int valid_count = 0;

            select_atom_cache(a);
            grow_neighbor_rows(jnum);
            calc_neighbor_rows(jnum, dr);

            std::fill(atom_input_vals.begin(), atom_input_vals.end(), 0.0);
            std::fill(atom_bond_vals.begin(), atom_bond_vals.end(), 0.0);
            std::fill(atom_input_energy_ders.begin(), atom_input_energy_ders.end(), 0.0);
            std::fill(atom_bond_energy_ders.begin(), atom_bond_energy_ders.end(), 0.0);
            atom_input_vals[0] = 1.0;
            atom_bond_vals[bond_offsets[0]] = 1.0;

            // ------------ The neighbors within the cutoff: harmonics, radial basis and the input vector ------------
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
                radial_basis->select_row(valid_count);
                double* basis_vals = neighbor_radial_vals.data() + (size_t) valid_count * radial_basis_size;
                std::copy(radial_basis->radial_basis_vals.begin(), radial_basis->radial_basis_vals.end(), basis_vals);
                std::copy(radial_basis->radial_basis_ders.begin(), radial_basis->radial_basis_ders.end(), neighbor_radial_ders.data() + (size_t) valid_count * radial_basis_size);
                harmonics.select_row(valid_count);
                double* ylm = neighbor_harmonics.data() + (size_t) valid_count * harmonic_count;
                std::copy(harmonics.vals.begin(), harmonics.vals.end(), ylm);

                // The force pass's values, laid out as PairETN::compute caches them
                double* cache = neighbor_cache + (size_t) valid_count * cache_stride;
                double* ylm_grads = cache + 1 + harmonic_count;
                double* mixed = ylm_grads + 3 * harmonic_count;
                double* mixed_ders = mixed + mixed_size;
                if (forces) {
                    cache[0] = inv_dist;
                    std::copy(harmonics.vals.begin(), harmonics.vals.end(), cache + 1);
                    std::copy(harmonics.grads.begin(), harmonics.grads.end(), ylm_grads);
                }

                for (int l = 0; l < l_count; l++) {
                    const int n_rad = radial_func_counts[l], rank_r = radial_ranks[l];
                    const double* s = radial_mixing[l].data() + (size_t) (itype * species_count + jtype) * n_rad * rank_r;
                    const int shift = l == 0 ? 1 : 0;
                    for (int n = 0; n < rank_r; n++) {
                        double val = 0;
                        for (int np = 0; np < n_rad; np++) val += basis_vals[np] * s[np * rank_r + n];
                        for (int m = -l; m <= l; m++) atom_input_vals[input_shape.index(l, m) + shift + n] += ylm[l * l + l + m] * val;
                        if (forces) {
                            const double* basis_ders = neighbor_radial_ders.data() + (size_t) valid_count * radial_basis_size;
                            double der = 0;
                            for (int np = 0; np < n_rad; np++) der += basis_ders[np] * s[np * rank_r + n];
                            mixed[mixed_offsets[l] + n] = val;
                            mixed_ders[mixed_offsets[l] + n] = der;
                        }
                    }
                }
                valid_count++;
            }

            // ------------ Tensor Train, Forwards ------------
            for (int c = 0; c < core_count; c++) {
                const CovariantShape& left = bond_shapes[c];
                const CovariantShape& right = bond_shapes[c + 1];
                for (const CoreTerm& t : core_terms[c]) {
                    const Core& core = t.imaginary ? cores_im[c] : cores[c];
                    const double* u = atom_bond_vals.data() + bond_offsets[c] + t.in_part * left.size;
                    double* out = atom_bond_vals.data() + bond_offsets[c + 1] + t.out_part * right.size;
                    for (const CoreBlock& block : core.blocks) {
                        const int n1_count = left.ranks[block.l1], n2_count = input_shape.ranks[block.l2], n3_count = right.ranks[block.l3];
                        const int m1_stride = n3_count * n2_count;
                        double* block_products = atom_core_products.data() + t.products_offset + block.products_offset;

                        // The block contracted with the left bond, once per m1
                        for (int m1 = -block.l1; m1 <= block.l1; m1++) {
                            const double* pu = u + left.index(block.l1, m1);
                            double* products = block_products + (m1 + block.l1) * m1_stride;
                            for (int n3 = 0; n3 < n3_count; n3++)
                                for (int n2 = 0; n2 < n2_count; n2++) {
                                    const double* c2 = core.coeffs.data() + block.base + n3 * block.stride3 + n2 * block.stride2;
                                    double prod = 0;
                                    for (int n1 = 0; n1 < n1_count; n1++) prod += c2[n1] * pu[n1];
                                    products[n3 * n2_count + n2] = prod;
                                }
                        }

                        for (int e = block.w_begin; e < block.w_end; e++) {
                            const WignerEntry& w = wigner[e];
                            const double* products = block_products + (w.m1 + block.l1) * m1_stride;
                            const double* pv = atom_input_vals.data() + input_shape.index(block.l2, w.m2);
                            double* po = out + right.index(block.l3, w.m3);
                            for (int n3 = 0; n3 < n3_count; n3++) {
                                double sum = 0;
                                for (int n2 = 0; n2 < n2_count; n2++) sum += products[n3 * n2_count + n2] * pv[n2];
                                po[n3] += w.w * sum;
                            }
                        }
                    }
                }
            }

            double* row = site_e_grad + (size_t) ii * width;
            atom_bond_energy_ders[bond_offsets[core_count]] = 1.0;

            if (!nonlinear) {
                const CovariantShape& last_bond = bond_shapes[last];
                const CovariantShape& energy_bond = bond_shapes[core_count];
                for (int v = 0; v < input_shape.size; v++) lane_input_vals[(size_t) v * A + a] = atom_input_vals[v];
                for (size_t q = 0; q < atom_core_products.size(); q++) lane_core_products[q * A + a] = atom_core_products[q];

                // =========== The last core's columns: their site energies, their derivatives w.r.t. the input, and the passes' seeds ===========
                double* core_ders = row + core_columns[last];
                for (size_t b = 0; b < cores[last].blocks.size(); b++) {
                    const CoreBlock& block = cores[last].blocks[b];
                    const LastBlock& lb = last_blocks[b];
                    const int n1_count = last_bond.ranks[block.l1], n2_count = input_shape.ranks[block.l2], n3_count = energy_bond.ranks[block.l3];
                    const int m1_count = 2 * block.l1 + 1;
                    if (lb.unit)
                        for (int m1 = -block.l1; m1 <= block.l1; m1++)
                            for (int n1 = 0; n1 < n1_count; n1++)
                                bond_pass_ders[((size_t) (bond_offsets[last] + last_bond.index(block.l1, m1) + n1) * pass_count + lb.first_pass + n1 * m1_count + m1 + block.l1) * A + a] = 1.0;
                    for (int e = block.w_begin; e < block.w_end; e++) {
                        const WignerEntry& w = wigner[e];
                        const int iu = bond_offsets[last] + last_bond.index(block.l1, w.m1), iv = input_shape.index(block.l2, w.m2);
                        const double* po = atom_bond_energy_ders.data() + bond_offsets[core_count] + energy_bond.index(block.l3, w.m3);
                        for (int n3 = 0; n3 < n3_count; n3++) {
                            const double g = w.w * po[n3];
                            if (g == 0.0) continue;
                            for (int n1 = 0; n1 < n1_count; n1++)
                                for (int n2 = 0; n2 < n2_count; n2++) {
                                    const int k = lb.first_column + (n3 * n1_count + n1) * n2_count + n2;
                                    core_ders[column_coeffs[k]] += g * atom_bond_vals[iu + n1] * atom_input_vals[iv + n2];
                                    lane_column_ders[((size_t) (iv + n2) * column_count + k) * A + a] += g * atom_bond_vals[iu + n1];
                                    if (lb.unit)
                                        unit_coefs[((size_t) lb.first_unit_coef + (k - lb.first_column) * m1_count + w.m1 + block.l1) * A + a] += g * atom_input_vals[iv + n2];
                                    else
                                        bond_pass_ders[((size_t) (iu + n1) * pass_count + column_pass[k]) * A + a] += g * atom_input_vals[iv + n2];
                                }
                        }
                    }
                }

                row[species_columns + itype] = 1.0;
                cached_counts[a] = valid_count;
                continue;
            }

            // =========== Begin Backpropagation, with the cores' columns ===========
            for (int c = core_count - 1; c >= 0; c--) {
                const CovariantShape& left = bond_shapes[c];
                const CovariantShape& right = bond_shapes[c + 1];
                for (const CoreTerm& t : core_terms[c]) {
                    const Core& core = t.imaginary ? cores_im[c] : cores[c];
                    const int in = bond_offsets[c] + t.in_part * left.size;
                    const int out = bond_offsets[c + 1] + t.out_part * right.size;
                    const double* u = atom_bond_vals.data() + in;
                    double* u_ders = atom_bond_energy_ders.data() + in;
                    double* core_ders = row + (t.imaginary ? core_im_columns[c] : core_columns[c]);
                    for (const CoreBlock& block : core.blocks) {
                        const int n1_count = left.ranks[block.l1], n2_count = input_shape.ranks[block.l2], n3_count = right.ranks[block.l3];
                        const int m1_stride = n3_count * n2_count;
                        const int m1_count = 2 * block.l1 + 1;
                        const double* block_products = atom_core_products.data() + t.products_offset + block.products_offset;

                        // The input's derivatives, and the products' summed per m1
                        std::fill(atom_core_product_ders.begin(), atom_core_product_ders.begin() + m1_count * m1_stride, 0.0);
                        for (int e = block.w_begin; e < block.w_end; e++) {
                            const WignerEntry& w = wigner[e];
                            const int iv = input_shape.index(block.l2, w.m2);
                            const double* products = block_products + (w.m1 + block.l1) * m1_stride;
                            double* product_ders = atom_core_product_ders.data() + (w.m1 + block.l1) * m1_stride;
                            const double* po = atom_bond_energy_ders.data() + out + right.index(block.l3, w.m3);
                            for (int n3 = 0; n3 < n3_count; n3++) {
                                const double g = w.w * po[n3];
                                if (g == 0.0) continue;
                                for (int n2 = 0; n2 < n2_count; n2++) {
                                    atom_input_energy_ders[iv + n2] += g * products[n3 * n2_count + n2];
                                    product_ders[n3 * n2_count + n2] += g * atom_input_vals[iv + n2];
                                }
                            }
                        }

                        // The left bond's and the core's derivatives, once per m1
                        for (int m1 = -block.l1; m1 <= block.l1; m1++) {
                            const int iu = left.index(block.l1, m1);
                            const double* product_ders = atom_core_product_ders.data() + (m1 + block.l1) * m1_stride;
                            for (int n3 = 0; n3 < n3_count; n3++)
                                for (int n2 = 0; n2 < n2_count; n2++) {
                                    const double d = product_ders[n3 * n2_count + n2];
                                    if (d == 0.0) continue;
                                    const int c2_offset = block.base + n3 * block.stride3 + n2 * block.stride2;
                                    const double* c2 = core.coeffs.data() + c2_offset;
                                    double* c2_ders = core_ders + c2_offset;
                                    for (int n1 = 0; n1 < n1_count; n1++) {
                                        u_ders[iu + n1] += d * c2[n1];
                                        c2_ders[n1] += d * u[iu + n1];
                                    }
                                }
                        }
                    }
                }
            }

            // ---- Species coefficients ----
            row[species_columns + itype] = 1.0;

            // ---- Radial and species-pair coefficients ----
            for (int jj = 0; jj < valid_count; jj++) {
                const int pair = itype * species_count + list.types[cached_j[jj]];
                const double* ylm = neighbor_harmonics.data() + (size_t) jj * harmonic_count;
                const double* basis_vals = neighbor_radial_vals.data() + (size_t) jj * radial_basis_size;
                for (int l = 0; l < l_count; l++) {
                    const int rank_s = species_ranks[l], n_rad = radial_func_counts[l], rank_r = radial_ranks[l];
                    const int shift = l == 0 ? 1 : 0;
                    const double* pair_coeffs = species_pair_coeffs[l].data() + pair * rank_s;
                    const double* radial = radial_basis_coeffs[l].data();
                    double* pair_ders = row + species_pair_columns[l] + pair * rank_s;
                    double* radial_ders = row + radial_columns[l];
                    for (int n = 0; n < rank_r; n++) {
                        // dE_i/db_n, the derivative of the channel's value
                        double value_der = 0;
                        for (int m = -l; m <= l; m++)
                            value_der += ylm[l * l + l + m] * atom_input_energy_ders[input_shape.index(l, m) + shift + n];
                        for (int np = 0; np < n_rad; np++) {
                            const double s = value_der * basis_vals[np];
                            if (s == 0.0) continue;
                            for (int k = 0; k < rank_s; k++) {
                                const int index = k + rank_s * (np + n_rad * n);
                                pair_ders[k] += s * radial[index];
                                radial_ders[index] += s * pair_coeffs[k];
                            }
                        }
                    }
                }
            }
        }

        if (nonlinear) continue;
        {
            const CovariantShape& last_bond = bond_shapes[last];
            const CovariantShape& energy_bond = bond_shapes[core_count];

            // =========== Backpropagation of every pass through the other cores ===========
            // An entry of the last bond carries only the passes it seeds; the other bonds carry them all
            for (int c = last - 1; c >= 0; c--) {
                const CovariantShape& left = bond_shapes[c];
                const CovariantShape& right = bond_shapes[c + 1];
                for (const CoreTerm& t : core_terms[c]) {
                    const Core& core = t.imaginary ? cores_im[c] : cores[c];
                    const int in = bond_offsets[c] + t.in_part * left.size;
                    const int out = bond_offsets[c + 1] + t.out_part * right.size;
                    auto passes_of = [&](int io) {
                        if (c < last - 1) return std::pair<int, int>(0, pass_count);
                        const std::pair<int, int> range = entry_passes[io - bond_offsets[last]];
                        return std::pair<int, int>(range.first, std::max(range.first, range.second));
                    };
                    for (const CoreBlock& block : core.blocks) {
                        const int n1_count = left.ranks[block.l1], n2_count = input_shape.ranks[block.l2], n3_count = right.ranks[block.l3];
                        const int m1_stride = n3_count * n2_count;
                        const int m1_count = 2 * block.l1 + 1;
                        const double* block_products = lane_core_products.data() + (size_t) (t.products_offset + block.products_offset) * A;

                        // The input's derivatives, and the products' summed per m1, which carry the passes of channel (l3, n3)
                        for (int n3 = 0; n3 < n3_count; n3++) {
                            const auto [p_begin, p_end] = passes_of(out + right.index(block.l3, -block.l3) + n3);
                            for (int m1 = 0; m1 < m1_count; m1++)
                                for (int n2 = 0; n2 < n2_count; n2++) {
                                    double* pd = product_pass_ders.data() + (size_t) (m1 * m1_stride + n3 * n2_count + n2) * pass_count * A;
                                    std::fill(pd + (size_t) p_begin * A, pd + (size_t) p_end * A, 0.0);
                                }
                        }
                        for (int e = block.w_begin; e < block.w_end; e++) {
                            const WignerEntry& w = wigner[e];
                            const int iv = input_shape.index(block.l2, w.m2), io = out + right.index(block.l3, w.m3);
                            const double* products = block_products + (size_t) (w.m1 + block.l1) * m1_stride * A;
                            double* product_ders = product_pass_ders.data() + (size_t) (w.m1 + block.l1) * m1_stride * pass_count * A;
                            for (int n3 = 0; n3 < n3_count; n3++) {
                                const auto [p_begin, p_end] = passes_of(io + n3);
                                const double* po = bond_pass_ders.data() + (size_t) (io + n3) * pass_count * A;
                                for (int n2 = 0; n2 < n2_count; n2++) {
                                    double gp[A], gv[A];
                                    for (int b = 0; b < A; b++) {
                                        gp[b] = w.w * products[(size_t) (n3 * n2_count + n2) * A + b];
                                        gv[b] = w.w * lane_input_vals[(size_t) (iv + n2) * A + b];
                                    }
                                    double* pi = input_pass_ders.data() + (size_t) (iv + n2) * pass_count * A;
                                    double* pd = product_ders + (size_t) (n3 * n2_count + n2) * pass_count * A;
                                    for (int p = p_begin; p < p_end; p++)
                                        for (int b = 0; b < A; b++) {
                                            pi[p * A + b] += gp[b] * po[p * A + b];
                                            pd[p * A + b] += gv[b] * po[p * A + b];
                                        }
                                }
                            }
                        }

                        // The left bond's derivatives, once per m1
                        for (int m1 = -block.l1; m1 <= block.l1; m1++) {
                            const int iu = in + left.index(block.l1, m1);
                            const double* product_ders = product_pass_ders.data() + (size_t) (m1 + block.l1) * m1_stride * pass_count * A;
                            for (int n3 = 0; n3 < n3_count; n3++) {
                                // the passes of channel (l3, n3), which every m3 of it carries
                                const auto [p_begin, p_end] = passes_of(out + right.index(block.l3, -block.l3) + n3);
                                for (int n2 = 0; n2 < n2_count; n2++) {
                                    const double* d = product_ders + (size_t) (n3 * n2_count + n2) * pass_count * A;
                                    const double* c2 = core.coeffs.data() + block.base + n3 * block.stride3 + n2 * block.stride2;
                                    for (int n1 = 0; n1 < n1_count; n1++) {
                                        double* pu = bond_pass_ders.data() + (size_t) (iu + n1) * pass_count * A;
                                        for (int q = p_begin * A; q < p_end * A; q++) pu[q] += c2[n1] * d[q];
                                    }
                                }
                            }
                        }
                    }
                }
            }

            for (int ii = ii_begin; ii < ii_end; ii++) {
                const int a = ii - ii_begin;
                const int i = list.ilist[ii];
                const int valid_count = cached_counts[a];
                select_atom_cache(a);
                for (size_t q = 0; q < input_column_ders.size(); q++) input_column_ders[q] = lane_column_ders[q * A + a];

                // ---- Each column's derivative w.r.t. the input: its own pass, or its seeds on its n1's unit passes ----
                for (size_t b = 0; b < cores[last].blocks.size(); b++) {
                    const CoreBlock& block = cores[last].blocks[b];
                    const LastBlock& lb = last_blocks[b];
                    const int n1_count = last_bond.ranks[block.l1], n2_count = input_shape.ranks[block.l2], n3_count = energy_bond.ranks[block.l3];
                    const int m1_count = 2 * block.l1 + 1;
                    for (int n3 = 0; n3 < n3_count; n3++)
                        for (int n1 = 0; n1 < n1_count; n1++)
                            for (int n2 = 0; n2 < n2_count; n2++) {
                                const int k = lb.first_column + (n3 * n1_count + n1) * n2_count + n2;
                                for (int m = 0; m < (lb.unit ? m1_count : 1); m++) {
                                    const double coef = lb.unit ? unit_coefs[((size_t) lb.first_unit_coef + (k - lb.first_column) * m1_count + m) * A + a] : 1.0;
                                    const int p = lb.unit ? lb.first_pass + n1 * m1_count + m : column_pass[k];
                                    if (coef == 0.0) continue;
                                    for (int v = 0; v < input_shape.size; v++)
                                        input_column_ders[(size_t) v * column_count + k] += coef * input_pass_ders[((size_t) v * pass_count + p) * A + a];
                                }
                            }
                }

                //------------ Each column's forces and virial from its input vector's derivatives ------------
                if (forces)
                    for (int jj = 0; jj < valid_count; jj++) {
                        const int j = cached_j[jj];
                        const double* r = valid_dr[jj].data();
                        const double* cache = neighbor_cache + (size_t) jj * cache_stride;
                        const double inv_dist = cache[0];
                        const double* ylm = cache + 1;
                        const double* ylm_grads = ylm + harmonic_count;
                        const double* mixed = ylm_grads + 3 * harmonic_count;
                        const double* mixed_ders = mixed + mixed_size;

                        // per column, the angular part (3) and the radial part of dE_k/dr_ij
                        std::fill(column_pair_ders.begin(), column_pair_ders.end(), 0.0);
                        double* angular = column_pair_ders.data();
                        double* radial_force = angular + 3 * column_count;
                        for (int l = 0; l < l_count; l++) {
                            const int rank_r = radial_ranks[l];
                            const double* b = mixed + mixed_offsets[l];
                            const double* db = mixed_ders + mixed_offsets[l];
                            const int shift = l == 0 ? 1 : 0;
                            for (int m = -l; m <= l; m++) {
                                const int index = l * l + l + m;
                                for (int n = 0; n < rank_r; n++) {
                                    const double* ders = input_column_ders.data() + (size_t) (input_shape.index(l, m) + shift + n) * column_count;
                                    const double ga[3] = {b[n] * ylm_grads[index * 3 + 0], b[n] * ylm_grads[index * 3 + 1], b[n] * ylm_grads[index * 3 + 2]};
                                    const double gr = ylm[index] * db[n];
                                    for (int k = 0; k < column_count; k++) {
                                        angular[k] += ga[0] * ders[k];
                                        angular[column_count + k] += ga[1] * ders[k];
                                        angular[2 * column_count + k] += ga[2] * ders[k];
                                        radial_force[k] += gr * ders[k];
                                    }
                                }
                            }
                        }

                        // dE_k / dr_ij, with r_ij = r_j - r_i, into the columns' forces and virial as PairETN::compute tallies them
                        for (int a = 0; a < 3; a++)
                            for (int k = 0; k < column_count; k++) angular[a * column_count + k] = inv_dist * (angular[a * column_count + k] + radial_force[k] * r[a]);
                        const double* temp_force[3] = {angular, angular + column_count, angular + 2 * column_count};
                        if (force_grad)
                            for (int a = 0; a < 3; a++) {
                                double* fi = force_grad + (size_t) (i * 3 + a) * width + core_columns[last];
                                double* fj = force_grad + (size_t) (j * 3 + a) * width + core_columns[last];
                                for (int k = 0; k < column_count; k++) {
                                    fi[column_coeffs[k]] += temp_force[a][k];
                                    fj[column_coeffs[k]] -= temp_force[a][k];
                                }
                            }
                        if (virial_grad) {
                            static constexpr int VIRIAL_PAIRS[6][2] = {{0, 0}, {1, 1}, {2, 2}, {0, 1}, {0, 2}, {1, 2}};    // xx yy zz xy xz yz
                            for (int v = 0; v < 6; v++) {
                                double* pv = virial_grad + (size_t) v * width + core_columns[last];
                                const double ra = r[VIRIAL_PAIRS[v][0]];
                                const double* f = temp_force[VIRIAL_PAIRS[v][1]];
                                for (int k = 0; k < column_count; k++) pv[column_coeffs[k]] -= ra * f[k];
                            }
                        }
                    }
            }
        }
    }
}

// The linear columns of a structure of jacobian_chunk atoms or more in chunks of that many, the rest one atom at a time.
void ETN::eval_grad(const NeighList& list, double* site_e_grad, double* force_grad, double* virial_grad, bool nonlinear) {
    if (!nonlinear && list.inum >= jacobian_chunk)
        eval_grad_lanes<jacobian_chunk>(list, site_e_grad, force_grad, virial_grad, nonlinear);
    else
        eval_grad_lanes<1>(list, site_e_grad, force_grad, virial_grad, nonlinear);
}

/* ----------------------------------------------------------------------
   Per site, the loss is a E_i + sum_j w_j . dE_i/dr_ij with a = dL/dE and
   w_j = dL/dF_i - dL/dF_j - S r_ij, where S[b][a] = dL/dvirial_ab, since
   virial_ab sums -r_a f_b over the pairs. The forward pass carries each
   entry of the input and bond vectors' derivative along w (its tangent)
   beside its value. Backward, the tangents' adjoints are the energy
   derivatives input_energy_ders and bond_energy_ders, and the values'
   adjoints pick up the tangents through the cores. A harmonic's tangent is
   grad Y . w / |r|, the distance's w . r / |r|. The central atoms go in
   chunks, as in PairETN::compute, and a core coefficient's gradient sums
   over the chunk's atoms.
------------------------------------------------------------------------- */
void ETN::eval_loss_grad(const NeighList& list, double dloss_denergy, const double* dloss_dforces, const double* dloss_dvirial, double* loss_grad) {
    const int l_count = (int) species_ranks.size();
    const int core_count = get_core_count();
    const int harmonic_count = l_count * l_count;
    const double a = dloss_denergy;

    // dL/dvirial as a matrix over the components PairETN::compute tallies
    double S[3][3] = {};
    if (dloss_dvirial) {
        S[0][0] = dloss_dvirial[0];
        S[1][1] = dloss_dvirial[1];
        S[2][2] = dloss_dvirial[2];
        S[1][0] = dloss_dvirial[3];
        S[2][0] = dloss_dvirial[4];
        S[2][1] = dloss_dvirial[5];
    }

    input_tangents.resize(input_shape.size);
    bond_tangents.resize(bonds_size);
    input_adjoints.resize(input_shape.size);
    bond_adjoints.resize(bonds_size);
    core_product_tangents.resize(core_products.size());
    core_product_adjoints.resize(core_product_ders.size());
    radial_mixing_grad.resize(l_count);
    for (int l = 0; l < l_count; l++) radial_mixing_grad[l].assign(radial_mixing[l].size(), 0.0);

    grow_chunk_cache(list);
    grow_neighbor_rows(chunk_size * atom_cache_size);

    int nbr_offset = 0;

    // Loop over all provided neighbourhoods, a chunk of them at a time
    for (int ii_begin = 0; ii_begin < list.inum; ii_begin += chunk_size) {
        const int ii_end = std::min(ii_begin + chunk_size, list.inum);

        // Clear the input and bond vectors, their tangents, derivatives and adjoints
        for (std::vector<Lanes>* lanes : {&input_vals, &input_tangents, &bond_vals, &bond_tangents, &input_energy_ders, &input_adjoints, &bond_energy_ders, &bond_adjoints})
            std::fill(lanes->begin(), lanes->end(), Lanes());
        input_vals[0] = 1.0;
        bond_vals[bond_offsets[0]] = 1.0;

        // ------------ The neighbors within the cutoff: the input vector, its tangent and the radial basis ------------
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
            const size_t first_row = (size_t) (ii - ii_begin) * atom_cache_size;

            for (int jj = 0; jj < jnum; jj++) {
                const int j = nbrs[jj];
                const int jtype = list.types[j];
                const double r[3] = {dr[jj * 3 + 0], dr[jj * 3 + 1], dr[jj * 3 + 2]};
                const double rsq = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];

                if (rsq > max_cutoff_sq) continue;
                cached_j[valid_count] = j;
                valid_dr[valid_count] = {r[0], r[1], r[2]};

                const size_t row = first_row + valid_count;
                const double dist = std::sqrt(rsq);
                const double inv_dist = 1.0 / dist;
                radial_basis->select_row(valid_count);
                double* basis_vals = neighbor_radial_vals.data() + row * radial_basis_size;
                double* basis_ders = neighbor_radial_ders.data() + row * radial_basis_size;
                std::copy(radial_basis->radial_basis_vals.begin(), radial_basis->radial_basis_vals.end(), basis_vals);
                std::copy(radial_basis->radial_basis_ders.begin(), radial_basis->radial_basis_ders.end(), basis_ders);
                harmonics.select_row(valid_count);
                double* ylm = neighbor_harmonics.data() + row * harmonic_count;
                std::copy(harmonics.vals.begin(), harmonics.vals.end(), ylm);

                // The loss weight of this displacement, and the tangents of the harmonics and of the distance along it
                double w[3] = {0.0, 0.0, 0.0};
                if (dloss_dforces)
                    for (int d = 0; d < 3; d++) w[d] = dloss_dforces[i * 3 + d] - dloss_dforces[j * 3 + d];
                for (int d = 0; d < 3; d++) w[d] -= S[d][0] * r[0] + S[d][1] * r[1] + S[d][2] * r[2];
                const double dist_tangent = (w[0] * r[0] + w[1] * r[1] + w[2] * r[2]) * inv_dist;
                neighbor_dist_tangents[row] = dist_tangent;
                double* ylm_tangents = neighbor_harmonic_tangents.data() + row * harmonic_count;
                for (int index = 0; index < harmonic_count; index++) {
                    const double* g = harmonics.grads.data() + index * 3;
                    ylm_tangents[index] = (g[0] * w[0] + g[1] * w[1] + g[2] * w[2]) * inv_dist;
                }

                for (int l = 0; l < l_count; l++) {
                    const int n_rad = radial_func_counts[l], rank_r = radial_ranks[l];
                    const double* s = radial_mixing[l].data() + (size_t) (itype * species_count + jtype) * n_rad * rank_r;
                    const int shift = l == 0 ? 1 : 0;
                    for (int n = 0; n < rank_r; n++) {
                        double val = 0, der = 0;
                        for (int np = 0; np < n_rad; np++) {
                            val += basis_vals[np] * s[np * rank_r + n];
                            der += basis_ders[np] * s[np * rank_r + n];
                        }
                        for (int m = -l; m <= l; m++) {
                            const int index = l * l + l + m;
                            const int k = input_shape.index(l, m) + shift + n;
                            input_vals[k][ii - ii_begin] += ylm[index] * val;
                            input_tangents[k][ii - ii_begin] += ylm_tangents[index] * val + ylm[index] * dist_tangent * der;
                        }
                    }
                }
                valid_count++;
            }
            cached_counts[ii - ii_begin] = valid_count;
        }

        // ------------ Tensor Train, Forwards: values and tangents ------------
        for (int c = 0; c < core_count; c++) {
            const CovariantShape& left = bond_shapes[c];
            const CovariantShape& right = bond_shapes[c + 1];
            for (const CoreTerm& t : core_terms[c]) {
                const Core& core = t.imaginary ? cores_im[c] : cores[c];
                const int in = bond_offsets[c] + t.in_part * left.size;
                const int out = bond_offsets[c + 1] + t.out_part * right.size;
                for (const CoreBlock& block : core.blocks) {
                    const int n1_count = left.ranks[block.l1], n2_count = input_shape.ranks[block.l2], n3_count = right.ranks[block.l3];
                    const int m1_stride = n3_count * n2_count;
                    Lanes* block_products = core_products.data() + t.products_offset + block.products_offset;
                    Lanes* block_product_tangents = core_product_tangents.data() + t.products_offset + block.products_offset;

                    // The block contracted with the left bond and its tangent, once per m1
                    for (int m1 = -block.l1; m1 <= block.l1; m1++) {
                        const int iu = in + left.index(block.l1, m1);
                        Lanes* products = block_products + (m1 + block.l1) * m1_stride;
                        Lanes* product_tangents = block_product_tangents + (m1 + block.l1) * m1_stride;
                        for (int n3 = 0; n3 < n3_count; n3++)
                            for (int n2 = 0; n2 < n2_count; n2++) {
                                const double* c2 = core.coeffs.data() + block.base + n3 * block.stride3 + n2 * block.stride2;
                                Lanes prod, prod_tangent;
                                for (int n1 = 0; n1 < n1_count; n1++) {
                                    prod += c2[n1] * bond_vals[iu + n1];
                                    prod_tangent += c2[n1] * bond_tangents[iu + n1];
                                }
                                products[n3 * n2_count + n2] = prod;
                                product_tangents[n3 * n2_count + n2] = prod_tangent;
                            }
                    }

                    for (int e = block.w_begin; e < block.w_end; e++) {
                        const WignerEntry& w = wigner[e];
                        const int iv = input_shape.index(block.l2, w.m2), io = out + right.index(block.l3, w.m3);
                        const Lanes* products = block_products + (w.m1 + block.l1) * m1_stride;
                        const Lanes* product_tangents = block_product_tangents + (w.m1 + block.l1) * m1_stride;
                        for (int n3 = 0; n3 < n3_count; n3++) {
                            Lanes sum, sum_tangent;
                            for (int n2 = 0; n2 < n2_count; n2++) {
                                sum += products[n3 * n2_count + n2] * input_vals[iv + n2];
                                sum_tangent += product_tangents[n3 * n2_count + n2] * input_vals[iv + n2];
                                sum_tangent += products[n3 * n2_count + n2] * input_tangents[iv + n2];
                            }
                            bond_vals[io + n3] += w.w * sum;
                            bond_tangents[io + n3] += w.w * sum_tangent;
                        }
                    }
                }
            }
        }

        // =========== Backpropagation of values and tangents, seeded on the chunk's atoms only ===========
        for (int ii = ii_begin; ii < ii_end; ii++) {
            bond_energy_ders[bond_offsets[core_count]][ii - ii_begin] = 1.0;
            bond_adjoints[bond_offsets[core_count]][ii - ii_begin] = a;
        }
        for (int c = core_count - 1; c >= 0; c--) {
            const CovariantShape& left = bond_shapes[c];
            const CovariantShape& right = bond_shapes[c + 1];
            for (const CoreTerm& t : core_terms[c]) {
                const Core& core = t.imaginary ? cores_im[c] : cores[c];
                const int in = bond_offsets[c] + t.in_part * left.size;
                const int out = bond_offsets[c + 1] + t.out_part * right.size;
                double* core_grad = loss_grad + (t.imaginary ? core_im_columns[c] : core_columns[c]);
                for (const CoreBlock& block : core.blocks) {
                    const int n1_count = left.ranks[block.l1], n2_count = input_shape.ranks[block.l2], n3_count = right.ranks[block.l3];
                    const int m1_stride = n3_count * n2_count;
                    const int m1_count = 2 * block.l1 + 1;
                    const Lanes* block_products = core_products.data() + t.products_offset + block.products_offset;
                    const Lanes* block_product_tangents = core_product_tangents.data() + t.products_offset + block.products_offset;

                    // The input's derivatives and adjoints, and the products' summed per m1
                    std::fill(core_product_ders.begin(), core_product_ders.begin() + m1_count * m1_stride, Lanes());
                    std::fill(core_product_adjoints.begin(), core_product_adjoints.begin() + m1_count * m1_stride, Lanes());
                    for (int e = block.w_begin; e < block.w_end; e++) {
                        const WignerEntry& w = wigner[e];
                        const int iv = input_shape.index(block.l2, w.m2), io = out + right.index(block.l3, w.m3);
                        const Lanes* products = block_products + (w.m1 + block.l1) * m1_stride;
                        const Lanes* product_tangents = block_product_tangents + (w.m1 + block.l1) * m1_stride;
                        Lanes* product_ders = core_product_ders.data() + (w.m1 + block.l1) * m1_stride;
                        Lanes* product_adjoints = core_product_adjoints.data() + (w.m1 + block.l1) * m1_stride;
                        for (int n3 = 0; n3 < n3_count; n3++) {
                            const Lanes G = w.w * bond_energy_ders[io + n3];
                            const Lanes A = w.w * bond_adjoints[io + n3];
                            for (int n2 = 0; n2 < n2_count; n2++) {
                                const int k = n3 * n2_count + n2;
                                input_energy_ders[iv + n2] += G * products[k];
                                input_adjoints[iv + n2] += A * products[k];
                                input_adjoints[iv + n2] += G * product_tangents[k];
                                product_ders[k] += G * input_vals[iv + n2];
                                product_adjoints[k] += A * input_vals[iv + n2];
                                product_adjoints[k] += G * input_tangents[iv + n2];
                            }
                        }
                    }

                    // The left bond's derivatives and adjoints, and the core's gradient summed over the chunk's atoms, once per m1
                    for (int m1 = -block.l1; m1 <= block.l1; m1++) {
                        const int iu = in + left.index(block.l1, m1);
                        const Lanes* product_ders = core_product_ders.data() + (m1 + block.l1) * m1_stride;
                        const Lanes* product_adjoints = core_product_adjoints.data() + (m1 + block.l1) * m1_stride;
                        for (int n3 = 0; n3 < n3_count; n3++)
                            for (int n2 = 0; n2 < n2_count; n2++) {
                                const Lanes& Gv = product_ders[n3 * n2_count + n2];
                                const Lanes& Av = product_adjoints[n3 * n2_count + n2];
                                const int c2_offset = block.base + n3 * block.stride3 + n2 * block.stride2;
                                const double* c2 = core.coeffs.data() + c2_offset;
                                double* c2_grad = core_grad + c2_offset;
                                for (int n1 = 0; n1 < n1_count; n1++) {
                                    bond_energy_ders[iu + n1] += Gv * c2[n1];
                                    bond_adjoints[iu + n1] += Av * c2[n1];
                                    c2_grad[n1] += sum_of_products(Av, bond_vals[iu + n1]) + sum_of_products(Gv, bond_tangents[iu + n1]);
                                }
                            }
                    }
                }
            }
        }

        for (int ii = ii_begin; ii < ii_end; ii++) {
            const int itype = list.types[list.ilist[ii]];
            const int valid_count = cached_counts[ii - ii_begin];

            select_atom_cache(ii - ii_begin);
            const size_t first_row = (size_t) (ii - ii_begin) * atom_cache_size;

            // ---- Species coefficients ----
            loss_grad[species_columns + itype] += a;

            // ---- The radial mixing of each species pair ----
            for (int jj = 0; jj < valid_count; jj++) {
                const size_t row = first_row + jj;
                const int pair = itype * species_count + list.types[cached_j[jj]];
                const double* ylm = neighbor_harmonics.data() + row * harmonic_count;
                const double* ylm_tangents = neighbor_harmonic_tangents.data() + row * harmonic_count;
                const double dist_tangent = neighbor_dist_tangents[row];
                const double* basis_vals = neighbor_radial_vals.data() + row * radial_basis_size;
                const double* basis_ders = neighbor_radial_ders.data() + row * radial_basis_size;
                for (int l = 0; l < l_count; l++) {
                    const int n_rad = radial_func_counts[l], rank_r = radial_ranks[l];
                    const int shift = l == 0 ? 1 : 0;
                    double* mixing_grad = radial_mixing_grad[l].data() + (size_t) pair * n_rad * rank_r;
                    for (int n = 0; n < rank_r; n++) {
                        // adjoints of the channel's value and derivative
                        double value_adjoint = 0.0, der_adjoint = 0.0;
                        for (int m = -l; m <= l; m++) {
                            const int index = l * l + l + m;
                            const int k = input_shape.index(l, m) + shift + n;
                            value_adjoint += ylm[index] * input_adjoints[k][ii - ii_begin] + ylm_tangents[index] * input_energy_ders[k][ii - ii_begin];
                            der_adjoint += ylm[index] * input_energy_ders[k][ii - ii_begin];
                        }
                        der_adjoint *= dist_tangent;
                        for (int np = 0; np < n_rad; np++) mixing_grad[np * rank_r + n] += value_adjoint * basis_vals[np] + der_adjoint * basis_ders[np];
                    }
                }
            }
        }
    }

    // ---- Radial and species-pair coefficients, from S_l = A_l B_l ----
    for (int l = 0; l < l_count; l++) {
        const int rank_s = species_ranks[l], n_rad = radial_func_counts[l], rank_r = radial_ranks[l];
        const double* radial = radial_basis_coeffs[l].data();
        double* radial_grad = loss_grad + radial_columns[l];
        for (int pair = 0; pair < species_count * species_count; pair++) {
            const double* pair_coeffs = species_pair_coeffs[l].data() + pair * rank_s;
            const double* mixing_grad = radial_mixing_grad[l].data() + (size_t) pair * n_rad * rank_r;
            double* pair_grad = loss_grad + species_pair_columns[l] + pair * rank_s;
            for (int n = 0; n < rank_r; n++)
                for (int np = 0; np < n_rad; np++) {
                    const double s = mixing_grad[np * rank_r + n];
                    if (s == 0.0) continue;
                    for (int k = 0; k < rank_s; k++) {
                        const int index = k + rank_s * (np + n_rad * n);
                        pair_grad[k] += s * radial[index];
                        radial_grad[index] += s * pair_coeffs[k];
                    }
                }
        }
    }
}
