/* -*- c++ -*- ----------------------------------------------------------
   Derivatives of an ETN's site energies, and of a loss built on its energy,
   forces and virial, with respect to its coefficients.
------------------------------------------------------------------------- */

#include "etn.h"

#include <stdexcept>

void ETN::prepare_buffers() {
    site_input_vals.resize(input_shape.size);
    input_tangent.resize(input_shape.size);
    input_tangent_adj.resize(input_shape.size);
    site_bonds.resize(bonds_size);
    bond_tangent.resize(bonds_size);
    bond_tangent_adj.resize(bonds_size);
    prepare_columns();
}

void ETN::set_species_count(int n) {
    PairETN::set_species_count(n);
    prepare_columns();
}

void ETN::prepare_columns() {
    const int l_count = (int) species_ranks.size();
    const int core_count = get_core_count();
    int column = 0;

    radial_columns.resize(l_count);
    for (int l = 0; l < l_count; l++) radial_columns[l] = column, column += (int) radial_coeffs[l].size();
    species_pair_columns.resize(l_count);
    for (int l = 0; l < l_count; l++) species_pair_columns[l] = column, column += (int) species_pair_coeffs[l].size();
    core_columns.resize(core_count);
    for (int c = 0; c < core_count; c++) core_columns[c] = column, column += (int) core_coeffs[c].size();
    core_im_columns.resize(core_count);
    for (int c = 0; c < core_count; c++) core_im_columns[c] = column, column += (int) core_coeffs_im[c].size();
    species_columns = column;

    mixing_adj.resize(l_count);
    for (int l = 0; l < l_count; l++) mixing_adj[l].resize(radial_mixing[l].size());
}

void ETN::eval_grad(const NeighList& list, double* site_energy_grad, double* force_grad, double*) {
    if (force_grad)
        throw std::runtime_error("eval_grad: an ETN has no force gradients");
    const int width = coeff_count();
    int nbr_offset = 0;
    for (int ii = 0; ii < list.inum; ii++) {
        double* row = site_energy_grad + (size_t) ii * width;
        std::fill(row, row + width, 0.0);
        for (auto& adj : mixing_adj) std::fill(adj.begin(), adj.end(), 0.0);
        accumulate_site_grad(list, ii, nbr_offset, 1.0, nullptr, nullptr, row);
        scatter_mixing_adj(row);
        nbr_offset += list.numneigh[ii];
    }
}

void ETN::eval_loss_grad(const NeighList& list, double dloss_denergy, const double* dloss_dforces, const double* dloss_dvirial, double* loss_grad) {
    for (auto& adj : mixing_adj) std::fill(adj.begin(), adj.end(), 0.0);
    int nbr_offset = 0;
    for (int ii = 0; ii < list.inum; ii++) {
        accumulate_site_grad(list, ii, nbr_offset, dloss_denergy, dloss_dforces, dloss_dvirial, loss_grad);
        nbr_offset += list.numneigh[ii];
    }
    scatter_mixing_adj(loss_grad);
}

/* ----------------------------------------------------------------------
   The forces enter the loss through g_ij = dE_i/dr_ij: F_i += g_ij,
   F_j -= g_ij, virial -= r_ij (x) g_ij. So the loss's force and virial terms
   add sum_j w_ij . g_ij, with w_ij = dL/dF_i - dL/dF_j - dL/dvirial^T r_ij:
   the derivative of E_i along the neighbor displacements w_ij. It is carried
   forward as the tangent of v and of the bond vectors, and its gradient taken
   with the energy's in one backwards pass.
------------------------------------------------------------------------- */
void ETN::accumulate_site_grad(const NeighList& list, int ii, int nbr_offset, double energy_weight, const double* dloss_dforces, const double* dloss_dvirial, double* grad) {
    const int l_count = (int) species_ranks.size();
    const int harmonic_count = l_count * l_count;
    const int mixed_size = mixed_offsets[l_count];
    const int core_count = get_core_count();
    const int i = list.ilist[ii];
    const int itype = list.types[i];
    const bool tangent = dloss_dforces || dloss_dvirial;

    double* v = site_input_vals.data();
    const int valid_count = site_input(list, ii, nbr_offset, v);

    // ------------ Neighbor Displacements and the Tangent of v ------------
    if (tangent) {
        if (neighbor_displacements.size() < (size_t) cache_size * 3) neighbor_displacements.resize((size_t) cache_size * 3);
        std::fill(input_tangent.begin(), input_tangent.end(), 0.0);
        const double* dV = dloss_dvirial;
        for (int jj = 0; jj < valid_count; jj++) {
            const int j = cached_j[jj];
            const double* r = valid_dr[jj].data();
            double* w = neighbor_displacements.data() + jj * 3;
            for (int a = 0; a < 3; a++) w[a] = dloss_dforces ? dloss_dforces[i * 3 + a] - dloss_dforces[j * 3 + a] : 0.0;
            if (dV) {
                w[0] -= dV[0] * r[0];
                w[1] -= dV[3] * r[0] + dV[1] * r[1];
                w[2] -= dV[4] * r[0] + dV[5] * r[1] + dV[2] * r[2];
            }

            const double* cache = neighbor_cache.data() + (size_t) jj * cache_stride;
            const double inv_dist = cache[0];
            const double* ylm = cache + 1;
            const double* ylm_grads = ylm + harmonic_count;
            const double* mixed = ylm_grads + 3 * harmonic_count;
            const double* mixed_ders = mixed + mixed_size;
            const double dist_tangent = (r[0] * w[0] + r[1] * w[1] + r[2] * w[2]) * inv_dist;
            for (int l = 0; l < l_count; l++) {
                const int rank_r = radial_ranks[l];
                const double* b = mixed + mixed_offsets[l];
                const double* db = mixed_ders + mixed_offsets[l];
                const int shift = l == 0 ? 1 : 0;
                for (int m = -l; m <= l; m++) {
                    const int index = l * l + l + m;
                    const double* g = ylm_grads + index * 3;
                    const double ylm_tangent = (g[0] * w[0] + g[1] * w[1] + g[2] * w[2]) * inv_dist;
                    const double radial_tangent = ylm[index] * dist_tangent;
                    double* pv = input_tangent.data() + input_shape.index(l, m) + shift;
                    for (int n = 0; n < rank_r; n++) pv[n] += ylm_tangent * b[n] + radial_tangent * db[n];
                }
            }
        }
    }

    // ------------ Tensor Train, Forwards: Values and Tangents ------------
    double* bond = site_bonds.data();
    train_forward(v, bond);
    if (tangent) {
        std::fill(bond_tangent.begin(), bond_tangent.end(), 0.0);
        for (int c = 0; c < core_count; c++) {
            const CovariantShape& left = bond_shapes[c];
            const CovariantShape& right = bond_shapes[c + 1];
            for (const CoreTerm& t : core_terms[c]) {
                const Core& core = t.imaginary ? cores_im[c] : cores[c];
                const double* coeffs = t.imaginary ? core_coeffs_im[c].data() : core_coeffs[c].data();
                const int in = bond_offsets[c] + t.in_part * left.size;
                const int out = bond_offsets[c + 1] + t.out_part * right.size;
                contract(core, coeffs, left, right, bond_tangent.data() + in, v, bond_tangent.data() + out);
                contract(core, coeffs, left, right, bond + in, input_tangent.data(), bond_tangent.data() + out);
            }
        }
    }

    // ------------ Tensor Train, Backwards ------------
    std::vector<double*> core_adj(core_count), core_im_adj(core_count);
    for (int c = 0; c < core_count; c++) {
        core_adj[c] = grad + core_columns[c];
        core_im_adj[c] = grad + core_im_columns[c];
    }
    std::fill(bond_adj.begin(), bond_adj.end(), 0.0);
    std::fill(input_adj.begin(), input_adj.end(), 0.0);
    bond_adj[bond_offsets[core_count]] = energy_weight;

    // The tangents' adjoints first: they feed bond_adj, which the values' pass then reads complete
    if (tangent) {
        std::fill(bond_tangent_adj.begin(), bond_tangent_adj.end(), 0.0);
        std::fill(input_tangent_adj.begin(), input_tangent_adj.end(), 0.0);
        bond_tangent_adj[bond_offsets[core_count]] = 1.0;
        for (int c = core_count - 1; c >= 0; c--) {
            const CovariantShape& left = bond_shapes[c];
            const CovariantShape& right = bond_shapes[c + 1];
            for (const CoreTerm& t : core_terms[c]) {
                const Core& core = t.imaginary ? cores_im[c] : cores[c];
                const double* coeffs = t.imaginary ? core_coeffs_im[c].data() : core_coeffs[c].data();
                double* c_adj = t.imaginary ? core_im_adj[c] : core_adj[c];
                const int in = bond_offsets[c] + t.in_part * left.size;
                const int out = bond_offsets[c + 1] + t.out_part * right.size;
                contract_adjoint(core, coeffs, left, right, bond_tangent.data() + in, v, bond_tangent_adj.data() + out, bond_tangent_adj.data() + in, input_adj.data(), c_adj);
                contract_adjoint(core, coeffs, left, right, bond + in, input_tangent.data(), bond_tangent_adj.data() + out, bond_adj.data() + in, input_tangent_adj.data(), c_adj);
            }
        }
    }
    train_backward(v, bond, bond_adj.data(), input_adj.data(), core_adj.data(), core_im_adj.data());
    grad[species_columns + itype] += energy_weight;

    // ------------ Through v and its Tangent into the Radial Mixing ------------
    for (int jj = 0; jj < valid_count; jj++) {
        const int jtype = list.types[cached_j[jj]];
        const double* r = valid_dr[jj].data();
        const double* cache = neighbor_cache.data() + (size_t) jj * cache_stride;
        const double inv_dist = cache[0];
        const double* ylm = cache + 1;
        const double* ylm_grads = ylm + harmonic_count;
        const double* w = tangent ? neighbor_displacements.data() + jj * 3 : nullptr;
        const double dist_tangent = tangent ? (r[0] * w[0] + r[1] * w[1] + r[2] * w[2]) * inv_dist : 0.0;

        radial_basis->calc_radial_basis_ders(1.0 / inv_dist);
        const double* basis_vals = radial_basis->radial_basis_vals.data();
        const double* basis_ders = radial_basis->radial_basis_ders.data();

        for (int l = 0; l < l_count; l++) {
            const int n_rad = radial_func_counts[l], rank_r = radial_ranks[l];
            const int shift = l == 0 ? 1 : 0;
            double* s_adj = mixing_adj[l].data() + (size_t) (itype * species_count + jtype) * n_rad * rank_r;
            for (int n = 0; n < rank_r; n++) {
                // Adjoints of the channel b_n and of its derivative db_n/dr
                double b_adj = 0, db_adj = 0;
                for (int m = -l; m <= l; m++) {
                    const int index = l * l + l + m;
                    const int k = input_shape.index(l, m) + shift + n;
                    b_adj += ylm[index] * input_adj[k];
                    if (tangent) {
                        const double* g = ylm_grads + index * 3;
                        const double ylm_tangent = (g[0] * w[0] + g[1] * w[1] + g[2] * w[2]) * inv_dist;
                        b_adj += ylm_tangent * input_tangent_adj[k];
                        db_adj += ylm[index] * dist_tangent * input_tangent_adj[k];
                    }
                }
                for (int np = 0; np < n_rad; np++) s_adj[np * rank_r + n] += basis_vals[np] * b_adj + basis_ders[np] * db_adj;
            }
        }
    }
}

// S_l[pair](n', n) = sum_k A_l[pair, k] B_l[k, n', n], so dS feeds dA through B and dB through A.
void ETN::scatter_mixing_adj(double* grad) const {
    const int l_count = (int) species_ranks.size();
    for (int l = 0; l < l_count; l++) {
        const int rank_s = species_ranks[l], n_rad = radial_func_counts[l], rank_r = radial_ranks[l];
        const double* radial = radial_coeffs[l].data();
        for (int pair = 0; pair < species_count * species_count; pair++) {
            const double* a = species_pair_coeffs[l].data() + pair * rank_s;
            const double* s_adj = mixing_adj[l].data() + (size_t) pair * n_rad * rank_r;
            double* a_grad = grad + species_pair_columns[l] + pair * rank_s;
            for (int n = 0; n < rank_r; n++)
                for (int np = 0; np < n_rad; np++) {
                    const double s = s_adj[np * rank_r + n];
                    if (s == 0.0) continue;
                    for (int k = 0; k < rank_s; k++) {
                        const int index = k + rank_s * (np + n_rad * n);
                        a_grad[k] += s * radial[index];
                        grad[radial_columns[l] + index] += s * a[k];
                    }
                }
        }
    }
}
