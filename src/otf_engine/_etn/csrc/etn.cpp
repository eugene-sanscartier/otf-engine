/* -*- c++ -*- ----------------------------------------------------------
   Input and bond vectors of an ETN, and derivatives of its site energies,
   and of a loss built on its energy, forces and virial, with respect to
   its coefficients.
------------------------------------------------------------------------- */

#include "etn.h"

#include <cmath>
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

        std::fill(input_vals.begin(), input_vals.end(), 0.0);
        std::fill(bond_vals.begin(), bond_vals.end(), 0.0);
        input_vals[0] = 1.0;
        bond_vals[bond_offsets[0]] = 1.0;

        // ------------ Calculate the Input Vector ------------
        for (int jj = 0; jj < jnum; jj++) {
            const int j = nbrs[jj];
            const int jtype = list.types[j];
            const double r[3] = {dr[jj * 3 + 0], dr[jj * 3 + 1], dr[jj * 3 + 2]};
            const double rsq = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];

            if (rsq > max_cutoff_sq) continue;

            const double dist = std::sqrt(rsq);
            const double inv_dist = 1.0 / dist;
            const double u[3] = {r[0] * inv_dist, r[1] * inv_dist, r[2] * inv_dist};
            radial_basis->calc_radial_basis_ders(dist);
            const double* basis_vals = radial_basis->radial_basis_vals.data();
            harmonics.compute(u);
            const double* ylm = harmonics.vals.data();

            for (int l = 0; l < l_count; l++) {
                const int n_rad = radial_func_counts[l], rank_r = radial_ranks[l];
                const double* s = radial_mixing[l].data() + (size_t) (itype * species_count + jtype) * n_rad * rank_r;
                const int shift = l == 0 ? 1 : 0;
                for (int n = 0; n < rank_r; n++) {
                    double val = 0;
                    for (int np = 0; np < n_rad; np++) val += basis_vals[np] * s[np * rank_r + n];
                    for (int m = -l; m <= l; m++) input_vals[input_shape.index(l, m) + shift + n] += ylm[l * l + l + m] * val;
                }
            }
        }

        // ------------ Tensor Train, Forwards ------------
        for (int c = 0; c < core_count; c++) {
            const CovariantShape& left = bond_shapes[c];
            const CovariantShape& right = bond_shapes[c + 1];
            for (const CoreTerm& t : core_terms[c]) {
                const Core& core = t.imaginary ? cores_im[c] : cores[c];
                const double* u = bond_vals.data() + bond_offsets[c] + t.in_part * left.size;
                double* out = bond_vals.data() + bond_offsets[c + 1] + t.out_part * right.size;
                for (const CoreBlock& block : core.blocks) {
                    const int n1_count = left.ranks[block.l1], n2_count = input_shape.ranks[block.l2], n3_count = right.ranks[block.l3];
                    const int m1_stride = n3_count * n2_count;
                    double* block_products = core_products.data() + t.products_offset + block.products_offset;

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
                        const double* pv = input_vals.data() + input_shape.index(block.l2, w.m2);
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

        std::copy(input_vals.begin(), input_vals.end(), inputs_out + (size_t) ii * input_shape.size);
        std::copy(bond_vals.begin(), bond_vals.end(), bonds_out + (size_t) ii * bonds_size);
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
------------------------------------------------------------------------- */
void ETN::eval_grad(const NeighList& list, double* site_e_grad, double* force_grad, double*) {
    if (force_grad)
        throw std::runtime_error("eval_grad: an ETN has no force gradients");

    const int width = coeff_count();
    const int l_count = (int) species_ranks.size();
    const int core_count = get_core_count();
    const int harmonic_count = l_count * l_count;

    std::fill(site_e_grad, site_e_grad + (size_t) list.inum * width, 0.0);

    int nbr_offset = 0;

    for (int ii = 0; ii < list.inum; ii++) {
        const int i = list.ilist[ii];
        const int itype = list.types[i];
        const int jnum = list.numneigh[ii];
        const int* nbrs = list.firstneigh + nbr_offset;
        const double* dr = list.displacements + nbr_offset * 3;
        nbr_offset += jnum;

        int valid_count = 0;

        grow_cache(jnum);
        grow_neighbor_rows(jnum);

        std::fill(input_vals.begin(), input_vals.end(), 0.0);
        std::fill(bond_vals.begin(), bond_vals.end(), 0.0);
        std::fill(input_energy_ders.begin(), input_energy_ders.end(), 0.0);
        std::fill(bond_energy_ders.begin(), bond_energy_ders.end(), 0.0);
        input_vals[0] = 1.0;
        bond_vals[bond_offsets[0]] = 1.0;

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
            const double u[3] = {r[0] * inv_dist, r[1] * inv_dist, r[2] * inv_dist};
            radial_basis->calc_radial_basis_ders(dist);
            double* basis_vals = neighbor_radial_vals.data() + (size_t) valid_count * radial_basis_size;
            std::copy(radial_basis->radial_basis_vals.begin(), radial_basis->radial_basis_vals.end(), basis_vals);
            std::copy(radial_basis->radial_basis_ders.begin(), radial_basis->radial_basis_ders.end(), neighbor_radial_ders.data() + (size_t) valid_count * radial_basis_size);
            harmonics.compute(u);
            double* ylm = neighbor_harmonics.data() + (size_t) valid_count * harmonic_count;
            std::copy(harmonics.vals.begin(), harmonics.vals.end(), ylm);

            for (int l = 0; l < l_count; l++) {
                const int n_rad = radial_func_counts[l], rank_r = radial_ranks[l];
                const double* s = radial_mixing[l].data() + (size_t) (itype * species_count + jtype) * n_rad * rank_r;
                const int shift = l == 0 ? 1 : 0;
                for (int n = 0; n < rank_r; n++) {
                    double val = 0;
                    for (int np = 0; np < n_rad; np++) val += basis_vals[np] * s[np * rank_r + n];
                    for (int m = -l; m <= l; m++) input_vals[input_shape.index(l, m) + shift + n] += ylm[l * l + l + m] * val;
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
                const double* u = bond_vals.data() + bond_offsets[c] + t.in_part * left.size;
                double* out = bond_vals.data() + bond_offsets[c + 1] + t.out_part * right.size;
                for (const CoreBlock& block : core.blocks) {
                    const int n1_count = left.ranks[block.l1], n2_count = input_shape.ranks[block.l2], n3_count = right.ranks[block.l3];
                    const int m1_stride = n3_count * n2_count;
                    double* block_products = core_products.data() + t.products_offset + block.products_offset;

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
                        const double* pv = input_vals.data() + input_shape.index(block.l2, w.m2);
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

        // =========== Begin Backpropagation, with the cores' columns ===========
        double* row = site_e_grad + (size_t) ii * width;
        bond_energy_ders[bond_offsets[core_count]] = 1.0;
        for (int c = core_count - 1; c >= 0; c--) {
            const CovariantShape& left = bond_shapes[c];
            const CovariantShape& right = bond_shapes[c + 1];
            for (const CoreTerm& t : core_terms[c]) {
                const Core& core = t.imaginary ? cores_im[c] : cores[c];
                const int in = bond_offsets[c] + t.in_part * left.size;
                const int out = bond_offsets[c + 1] + t.out_part * right.size;
                const double* u = bond_vals.data() + in;
                double* u_ders = bond_energy_ders.data() + in;
                double* core_ders = row + (t.imaginary ? core_im_columns[c] : core_columns[c]);
                for (const CoreBlock& block : core.blocks) {
                    const int n1_count = left.ranks[block.l1], n2_count = input_shape.ranks[block.l2], n3_count = right.ranks[block.l3];
                    const int m1_stride = n3_count * n2_count;
                    const int m1_count = 2 * block.l1 + 1;
                    const double* block_products = core_products.data() + t.products_offset + block.products_offset;

                    // The input's derivatives, and the products' summed per m1
                    std::fill(core_product_ders.begin(), core_product_ders.begin() + m1_count * m1_stride, 0.0);
                    for (int e = block.w_begin; e < block.w_end; e++) {
                        const WignerEntry& w = wigner[e];
                        const int iv = input_shape.index(block.l2, w.m2);
                        const double* products = block_products + (w.m1 + block.l1) * m1_stride;
                        double* product_ders = core_product_ders.data() + (w.m1 + block.l1) * m1_stride;
                        const double* po = bond_energy_ders.data() + out + right.index(block.l3, w.m3);
                        for (int n3 = 0; n3 < n3_count; n3++) {
                            const double g = w.w * po[n3];
                            if (g == 0.0) continue;
                            for (int n2 = 0; n2 < n2_count; n2++) {
                                input_energy_ders[iv + n2] += g * products[n3 * n2_count + n2];
                                product_ders[n3 * n2_count + n2] += g * input_vals[iv + n2];
                            }
                        }
                    }

                    // The left bond's and the core's derivatives, once per m1
                    for (int m1 = -block.l1; m1 <= block.l1; m1++) {
                        const int iu = left.index(block.l1, m1);
                        const double* product_ders = core_product_ders.data() + (m1 + block.l1) * m1_stride;
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
                        value_der += ylm[l * l + l + m] * input_energy_ders[input_shape.index(l, m) + shift + n];
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
}

/* ----------------------------------------------------------------------
   Per site, the loss is a E_i + sum_j w_j . dE_i/dr_ij with a = dL/dE and
   w_j = dL/dF_i - dL/dF_j - S r_ij, where S[b][a] = dL/dvirial_ab, since
   virial_ab sums -r_a f_b over the pairs. The forward pass carries each
   entry of the input and bond vectors' derivative along w (its tangent)
   beside its value. Backward, the tangents' adjoints are the energy
   derivatives input_energy_ders and bond_energy_ders, and the values'
   adjoints pick up the tangents through the cores. A harmonic's tangent is
   grad Y . w / |r|, the distance's w . r / |r|.
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

    int nbr_offset = 0;

    for (int ii = 0; ii < list.inum; ii++) {
        const int i = list.ilist[ii];
        const int itype = list.types[i];
        const int jnum = list.numneigh[ii];
        const int* nbrs = list.firstneigh + nbr_offset;
        const double* dr = list.displacements + nbr_offset * 3;
        nbr_offset += jnum;

        int valid_count = 0;

        grow_cache(jnum);
        grow_neighbor_rows(jnum);

        std::fill(input_vals.begin(), input_vals.end(), 0.0);
        std::fill(input_tangents.begin(), input_tangents.end(), 0.0);
        std::fill(bond_vals.begin(), bond_vals.end(), 0.0);
        std::fill(bond_tangents.begin(), bond_tangents.end(), 0.0);
        std::fill(input_energy_ders.begin(), input_energy_ders.end(), 0.0);
        std::fill(input_adjoints.begin(), input_adjoints.end(), 0.0);
        std::fill(bond_energy_ders.begin(), bond_energy_ders.end(), 0.0);
        std::fill(bond_adjoints.begin(), bond_adjoints.end(), 0.0);
        input_vals[0] = 1.0;
        bond_vals[bond_offsets[0]] = 1.0;

        // ------------ The neighbors within the cutoff: the input vector, its tangent and the radial basis ------------
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
            const double u[3] = {r[0] * inv_dist, r[1] * inv_dist, r[2] * inv_dist};
            radial_basis->calc_radial_basis_ders(dist);
            double* basis_vals = neighbor_radial_vals.data() + (size_t) valid_count * radial_basis_size;
            double* basis_ders = neighbor_radial_ders.data() + (size_t) valid_count * radial_basis_size;
            std::copy(radial_basis->radial_basis_vals.begin(), radial_basis->radial_basis_vals.end(), basis_vals);
            std::copy(radial_basis->radial_basis_ders.begin(), radial_basis->radial_basis_ders.end(), basis_ders);
            harmonics.compute(u);
            double* ylm = neighbor_harmonics.data() + (size_t) valid_count * harmonic_count;
            std::copy(harmonics.vals.begin(), harmonics.vals.end(), ylm);

            // The loss weight of this displacement, and the tangents of the harmonics and of the distance along it
            double w[3] = {0.0, 0.0, 0.0};
            if (dloss_dforces)
                for (int d = 0; d < 3; d++) w[d] = dloss_dforces[i * 3 + d] - dloss_dforces[j * 3 + d];
            for (int d = 0; d < 3; d++) w[d] -= S[d][0] * r[0] + S[d][1] * r[1] + S[d][2] * r[2];
            const double dist_tangent = (w[0] * r[0] + w[1] * r[1] + w[2] * r[2]) * inv_dist;
            neighbor_dist_tangents[valid_count] = dist_tangent;
            double* ylm_tangents = neighbor_harmonic_tangents.data() + (size_t) valid_count * harmonic_count;
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
                        input_vals[k] += ylm[index] * val;
                        input_tangents[k] += ylm_tangents[index] * val + ylm[index] * dist_tangent * der;
                    }
                }
            }
            valid_count++;
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
                    double* block_products = core_products.data() + t.products_offset + block.products_offset;
                    double* block_product_tangents = core_product_tangents.data() + t.products_offset + block.products_offset;

                    // The block contracted with the left bond and its tangent, once per m1
                    for (int m1 = -block.l1; m1 <= block.l1; m1++) {
                        const int iu = in + left.index(block.l1, m1);
                        double* products = block_products + (m1 + block.l1) * m1_stride;
                        double* product_tangents = block_product_tangents + (m1 + block.l1) * m1_stride;
                        for (int n3 = 0; n3 < n3_count; n3++)
                            for (int n2 = 0; n2 < n2_count; n2++) {
                                const double* c2 = core.coeffs.data() + block.base + n3 * block.stride3 + n2 * block.stride2;
                                double prod = 0, prod_tangent = 0;
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
                        const double* products = block_products + (w.m1 + block.l1) * m1_stride;
                        const double* product_tangents = block_product_tangents + (w.m1 + block.l1) * m1_stride;
                        for (int n3 = 0; n3 < n3_count; n3++) {
                            double sum = 0, sum_tangent = 0;
                            for (int n2 = 0; n2 < n2_count; n2++) {
                                const double prod = products[n3 * n2_count + n2], prod_tangent = product_tangents[n3 * n2_count + n2];
                                sum += prod * input_vals[iv + n2];
                                sum_tangent += prod_tangent * input_vals[iv + n2] + prod * input_tangents[iv + n2];
                            }
                            bond_vals[io + n3] += w.w * sum;
                            bond_tangents[io + n3] += w.w * sum_tangent;
                        }
                    }
                }
            }
        }

        // =========== Backpropagation of values and tangents ===========
        bond_energy_ders[bond_offsets[core_count]] = 1.0;
        bond_adjoints[bond_offsets[core_count]] = a;
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
                    const double* block_products = core_products.data() + t.products_offset + block.products_offset;
                    const double* block_product_tangents = core_product_tangents.data() + t.products_offset + block.products_offset;

                    // The input's derivatives and adjoints, and the products' summed per m1
                    std::fill(core_product_ders.begin(), core_product_ders.begin() + m1_count * m1_stride, 0.0);
                    std::fill(core_product_adjoints.begin(), core_product_adjoints.begin() + m1_count * m1_stride, 0.0);
                    for (int e = block.w_begin; e < block.w_end; e++) {
                        const WignerEntry& w = wigner[e];
                        const int iv = input_shape.index(block.l2, w.m2), io = out + right.index(block.l3, w.m3);
                        const double* products = block_products + (w.m1 + block.l1) * m1_stride;
                        const double* product_tangents = block_product_tangents + (w.m1 + block.l1) * m1_stride;
                        double* product_ders = core_product_ders.data() + (w.m1 + block.l1) * m1_stride;
                        double* product_adjoints = core_product_adjoints.data() + (w.m1 + block.l1) * m1_stride;
                        for (int n3 = 0; n3 < n3_count; n3++) {
                            const double G = w.w * bond_energy_ders[io + n3];
                            const double A = w.w * bond_adjoints[io + n3];
                            if (G == 0.0 && A == 0.0) continue;
                            for (int n2 = 0; n2 < n2_count; n2++) {
                                const int k = n3 * n2_count + n2;
                                input_energy_ders[iv + n2] += G * products[k];
                                input_adjoints[iv + n2] += A * products[k] + G * product_tangents[k];
                                product_ders[k] += G * input_vals[iv + n2];
                                product_adjoints[k] += A * input_vals[iv + n2] + G * input_tangents[iv + n2];
                            }
                        }
                    }

                    // The left bond's derivatives and adjoints, and the core's gradient, once per m1
                    for (int m1 = -block.l1; m1 <= block.l1; m1++) {
                        const int iu = in + left.index(block.l1, m1);
                        const double* product_ders = core_product_ders.data() + (m1 + block.l1) * m1_stride;
                        const double* product_adjoints = core_product_adjoints.data() + (m1 + block.l1) * m1_stride;
                        for (int n3 = 0; n3 < n3_count; n3++)
                            for (int n2 = 0; n2 < n2_count; n2++) {
                                const double Gv = product_ders[n3 * n2_count + n2], Av = product_adjoints[n3 * n2_count + n2];
                                if (Gv == 0.0 && Av == 0.0) continue;
                                const int c2_offset = block.base + n3 * block.stride3 + n2 * block.stride2;
                                const double* c2 = core.coeffs.data() + c2_offset;
                                double* c2_grad = core_grad + c2_offset;
                                for (int n1 = 0; n1 < n1_count; n1++) {
                                    bond_energy_ders[iu + n1] += Gv * c2[n1];
                                    bond_adjoints[iu + n1] += Av * c2[n1];
                                    c2_grad[n1] += Av * bond_vals[iu + n1] + Gv * bond_tangents[iu + n1];
                                }
                            }
                    }
                }
            }
        }

        // ---- Species coefficients ----
        loss_grad[species_columns + itype] += a;

        // ---- The radial mixing of each species pair ----
        for (int jj = 0; jj < valid_count; jj++) {
            const int pair = itype * species_count + list.types[cached_j[jj]];
            const double* ylm = neighbor_harmonics.data() + (size_t) jj * harmonic_count;
            const double* ylm_tangents = neighbor_harmonic_tangents.data() + (size_t) jj * harmonic_count;
            const double dist_tangent = neighbor_dist_tangents[jj];
            const double* basis_vals = neighbor_radial_vals.data() + (size_t) jj * radial_basis_size;
            const double* basis_ders = neighbor_radial_ders.data() + (size_t) jj * radial_basis_size;
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
                        value_adjoint += ylm[index] * input_adjoints[k] + ylm_tangents[index] * input_energy_ders[k];
                        der_adjoint += ylm[index] * input_energy_ders[k];
                    }
                    der_adjoint *= dist_tangent;
                    for (int np = 0; np < n_rad; np++) mixing_grad[np * rank_r + n] += value_adjoint * basis_vals[np] + der_adjoint * basis_ders[np];
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
