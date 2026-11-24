/* -*- c++ -*- ----------------------------------------------------------
   An ETN that grades extrapolation in compute. It takes one atom at a
   time: the forward pass, then the backwards pass, which gives the forces
   as PairETN::compute does and the site energy's gradient over every
   coefficient as ETN::eval_grad does. The grade is that gradient against
   the inverse active set, as MaxVol::grade takes it.
------------------------------------------------------------------------- */

#include "pair_etn_extrapolation.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

void PairETNExtrapolation::set_active_set(const double* invA, bool configuration_mode, int weight_scaling) {
    const size_t n = (size_t) coeff_count() * coeff_count();
    inverse_active_set.assign(invA, invA + n);
    energy_ders_wrt_coeffs.resize(coeff_count());
    this->configuration_mode = configuration_mode;
    this->weight_scaling = weight_scaling;
}

void PairETNExtrapolation::compute(const NeighList& list) {
    if (!has_active_set())
        throw std::runtime_error("PairETNExtrapolation: no active set; call set_active_set() first");

    const int l_count = (int) species_ranks.size();
    const int core_count = get_core_count();
    const int harmonic_count = l_count * l_count;
    const int mixed_size = mixed_offsets[l_count];
    int nbr_offset = 0;

    energy = 0.0;
    eatom.assign(list.n_atoms, 0.0);
    forces.assign((size_t) list.n_atoms * 3, 0.0);
    virial.fill(0.0);
    max_grade = 0.0;
    if (!configuration_mode) nbh_extrapolation_grades.assign(list.n_atoms, 0.0);
    std::fill(energy_ders_wrt_coeffs.begin(), energy_ders_wrt_coeffs.end(), 0.0);

    for (int ii = 0; ii < list.inum; ii++) {
        int valid_count = 0;
        const int i = list.ilist[ii];
        const int itype = list.types[i];
        const int jnum = list.numneigh[ii];
        const int* nbrs = list.firstneigh + nbr_offset;
        const double* dr = list.displacements + nbr_offset * 3;
        nbr_offset += jnum;

        grow_cache(jnum);
        if (neighbor_radial_vals.size() < (size_t) jnum * radial_basis_size) neighbor_radial_vals.resize((size_t) jnum * radial_basis_size);
        if (!configuration_mode) std::fill(energy_ders_wrt_coeffs.begin(), energy_ders_wrt_coeffs.end(), 0.0);

        std::fill(atom_input_vals.begin(), atom_input_vals.end(), 0.0);
        std::fill(atom_bond_vals.begin(), atom_bond_vals.end(), 0.0);
        std::fill(atom_input_energy_ders.begin(), atom_input_energy_ders.end(), 0.0);
        std::fill(atom_bond_energy_ders.begin(), atom_bond_energy_ders.end(), 0.0);
        atom_input_vals[0] = 1.0;
        atom_bond_vals[bond_offsets[0]] = 1.0;
        calc_neighbor_rows(jnum, dr);

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
            double* cache = neighbor_cache + (size_t) valid_count * cache_stride;
            cache[0] = inv_dist;
            double* ylm = cache + 1;
            double* ylm_grads = ylm + harmonic_count;
            double* mixed = ylm_grads + 3 * harmonic_count;
            double* mixed_ders = mixed + mixed_size;
            radial_basis->select_row(valid_count);
            const double* basis_vals = radial_basis->radial_basis_vals.data();
            const double* basis_ders = radial_basis->radial_basis_ders.data();
            std::copy(basis_vals, basis_vals + radial_basis_size, neighbor_radial_vals.data() + (size_t) valid_count * radial_basis_size);
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
                    double* pv = atom_input_vals.data() + input_shape.index(l, m) + shift;
                    for (int n = 0; n < rank_r; n++) pv[n] += y * b[n];
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

        const double nbh_energy = atom_bond_vals[bond_offsets[core_count]] + species_coeffs[itype];
        eatom[i] = nbh_energy;
        energy += nbh_energy;

        // =========== Begin Backpropagation, with the cores' columns ===========
        double* row = energy_ders_wrt_coeffs.data();
        atom_bond_energy_ders[bond_offsets[core_count]] = 1.0;
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
        row[species_columns + itype] += 1.0;

        for (int jj = 0; jj < valid_count; jj++) {
            const int j = cached_j[jj];
            const double* r = valid_dr[jj].data();
            const double* cache = neighbor_cache + (size_t) jj * cache_stride;
            const double inv_dist = cache[0];
            const double* ylm = cache + 1;
            const double* ylm_grads = ylm + harmonic_count;
            const double* mixed = ylm_grads + 3 * harmonic_count;
            const double* mixed_ders = mixed + mixed_size;
            const double* basis_vals = neighbor_radial_vals.data() + (size_t) jj * radial_basis_size;
            const int pair = itype * species_count + list.types[j];

            // ---- Radial and species-pair coefficients ----
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

            //------------ Forces from the input vector's derivatives ------------
            double angular[3] = {0, 0, 0};
            double radial_force = 0;
            for (int l = 0; l < l_count; l++) {
                const int rank_r = radial_ranks[l];
                const double* b = mixed + mixed_offsets[l];
                const double* db = mixed_ders + mixed_offsets[l];
                const int shift = l == 0 ? 1 : 0;
                for (int m = -l; m <= l; m++) {
                    const int index = l * l + l + m;
                    const double* ders = atom_input_energy_ders.data() + input_shape.index(l, m) + shift;
                    double ylm_der = 0, mixed_der = 0;
                    for (int n = 0; n < rank_r; n++) {
                        ylm_der += b[n] * ders[n];
                        mixed_der += db[n] * ders[n];
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

        if (!configuration_mode) {
            const double grade = calculate_extrapolation_grade();
            max_grade = std::max(grade, max_grade);
            nbh_extrapolation_grades[i] = grade;
        }
    }

    // The configuration grade, normalized by the atom count
    if (configuration_mode) max_grade = list.n_atoms > 0 ? calculate_extrapolation_grade() / std::pow((double) list.n_atoms, 0.5 * weight_scaling) : 0.0;
}

double PairETNExtrapolation::calculate_extrapolation_grade() const {
    const int n = coeff_count();
    double grade_max = 0;
    for (int i = 0; i < n; i++) {
        const double* row = inverse_active_set.data() + (size_t) i * n;
        double current_grade = 0;
        for (int j = 0; j < n; j++) current_grade += energy_ders_wrt_coeffs[j] * row[j];
        grade_max = std::max(std::abs(current_grade), grade_max);
    }
    return grade_max;
}
