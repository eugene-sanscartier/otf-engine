/* -*- c++ -*- ----------------------------------------------------------
   Derivatives of site energies, forces and virial with respect to the MTP
   coefficients.
------------------------------------------------------------------------- */

#include "mtp_training.h"

#include <cmath>

// Accumulate the six virial components for one Cartesian direction.
static inline void accumulate_virial_grad(double* virial_grad, int width, int d, int col, double contrib, const std::array<double, 3>& r) {
    virial_grad[d * width + col] -= contrib * r[d];
    if (d == 0) {
        virial_grad[3 * width + col] -= contrib * r[1] / 2;
        virial_grad[4 * width + col] -= contrib * r[2] / 2;
    } else if (d == 1) {
        virial_grad[3 * width + col] -= contrib * r[0] / 2;
        virial_grad[5 * width + col] -= contrib * r[2] / 2;
    } else {
        virial_grad[4 * width + col] -= contrib * r[0] / 2;
        virial_grad[5 * width + col] -= contrib * r[1] / 2;
    }
}

/* ----------------------------------------------------------------------
   dM_dc columns are global radial coefficient indices; radial_jacobian
   columns are per-pair ones within the row of the central species.
------------------------------------------------------------------------- */
void MTPTraining::scatter_radial_jacobian(int itype) {
    const int n_radial = radial_coeff_count;
    std::fill(dM_dc.begin(), dM_dc.begin() + (size_t) alpha_moment_count * n_radial, 0.0);

    for (int k = 0; k < alpha_index_basic_count; k++) {
        for (int jtype = 0; jtype < species_count; jtype++) {
            const double* src = radial_jacobian.data() + ((size_t) k * species_count + jtype) * radial_coeff_count_per_pair;
            double* dst = dM_dc.data() + (size_t) k * n_radial + (itype * species_count + jtype) * radial_coeff_count_per_pair;
            std::copy(src, src + radial_coeff_count_per_pair, dst);
        }
    }
}

void MTPTraining::propagate_radial_moment_ders() {
    const int n_radial = radial_coeff_count;

    // Forward propagate dM through the composite moments
    for (int k = 0; k < alpha_index_times_count; k++) {
        const int i0 = alpha_index_times[k][0], i1 = alpha_index_times[k][1];
        const int mul = alpha_index_times[k][2], i3 = alpha_index_times[k][3];
        const double M_i0 = moment_tensor_vals[i0], M_i1 = moment_tensor_vals[i1];
        const double* dM_i0 = dM_dc.data() + (size_t) i0 * n_radial;
        const double* dM_i1 = dM_dc.data() + (size_t) i1 * n_radial;
        double* dM_i3 = dM_dc.data() + (size_t) i3 * n_radial;
        for (int r = 0; r < n_radial; r++)
            dM_i3[r] += mul * (dM_i0[r] * M_i1 + M_i0 * dM_i1[r]);
    }

    // Back propagate dG = d(dE/dM_k)/dc
    std::fill(dG.begin(), dG.begin() + (size_t) alpha_moment_count * n_radial, 0.0);
    for (int k = alpha_index_times_count - 1; k >= 0; k--) {
        const int a0 = alpha_index_times[k][0], a1 = alpha_index_times[k][1];
        const int mul = alpha_index_times[k][2], a3 = alpha_index_times[k][3];
        const double G_a3 = nbh_energy_ders_wrt_moments[a3];
        const double M_a0 = moment_tensor_vals[a0], M_a1 = moment_tensor_vals[a1];
        const double* dM_a0 = dM_dc.data() + (size_t) a0 * n_radial;
        const double* dM_a1 = dM_dc.data() + (size_t) a1 * n_radial;
        const double* dG_a3 = dG.data() + (size_t) a3 * n_radial;
        double* dG_a0 = dG.data() + (size_t) a0 * n_radial;
        double* dG_a1 = dG.data() + (size_t) a1 * n_radial;
        for (int r = 0; r < n_radial; r++) {
            dG_a0[r] += G_a3 * mul * dM_a1[r] + dG_a3[r] * mul * M_a1;
            dG_a1[r] += G_a3 * mul * dM_a0[r] + dG_a3[r] * mul * M_a0;
        }
    }
}

/* ---------------------------------------------------------------------- */
void MTPTraining::get_coeffs(double* c) const {
    std::copy(radial_basis_coeffs.begin(), radial_basis_coeffs.end(), c);
    std::copy(species_coeffs.begin(), species_coeffs.end(), c + radial_coeff_count);
    std::copy(linear_coeffs.begin(), linear_coeffs.end(), c + radial_coeff_count + species_count);
}

void MTPTraining::set_coeffs(const double* c) {
    set_radial_basis_coeffs(c);
    set_species_coeffs(c + radial_coeff_count);
    set_linear_coeffs(c + radial_coeff_count + species_count);
}

/* ---------------------------------------------------------------------- */
void MTPTraining::eval_grad(const NeighList& list, double* site_e_grad, double* force_grad, double* virial_grad, bool radial) {
    const int width = coeff_count();
    const int n_radial = radial_coeff_count;
    const int n_linear = alpha_scalar_count;

    // Column offsets of each coefficient block within a row of width `width`.
    const int rad_off = 0;
    const int sp_off = n_radial;
    const int lin_off = n_radial + species_count;

    if (site_e_grad) std::fill(site_e_grad, site_e_grad + (size_t) list.inum * width, 0.0);
    if (force_grad) std::fill(force_grad, force_grad + (size_t) list.n_atoms * 3 * width, 0.0);
    if (virial_grad) std::fill(virial_grad, virial_grad + 6 * width, 0.0);

    // The forward pass fills radial_jacobian only when a gradient is wanted,
    // and angular factors only when a force gradient is.
    const bool need_radial_jacobian = radial && (site_e_grad || force_grad);
    const bool need_angular = radial && force_grad;

    if (need_radial_jacobian) {
        radial_jacobian.resize((size_t) alpha_index_basic_count * species_count * radial_coeff_count_per_pair);
        dM_dc.resize((size_t) alpha_moment_count * n_radial);
        dG.resize((size_t) alpha_moment_count * n_radial);
    }

    int nbr_offset = 0;

    for (int ii = 0; ii < list.inum; ii++) {
        const int i = list.ilist[ii];
        const int itype = list.types[i];
        const int jnum = list.numneigh[ii];
        const int* nbrs = list.firstneigh + nbr_offset;
        const double* dr = list.displacements + nbr_offset * 3;
        nbr_offset += jnum;

        int valid_count = 0;

        // Resize per neighbor arrays
        if (cache_size < jnum) {
            neighbor_cache.resize((size_t) jnum * (1 + 2 * radial_func_count));
            cached_j.resize(jnum);
            valid_dr.resize(jnum);
            cache_size = jnum;
        }
        if (jac_size < jnum) {
            jac_size = jnum;
            moment_jacobian.resize((size_t) jac_size * alpha_index_basic_count);
        }
        if (need_angular) {
            angular_values.resize((size_t) jac_size * alpha_index_basic_count);
            angular_jacobians.resize((size_t) jac_size * alpha_index_basic_count * 3);
        }

        std::fill(moment_tensor_vals.begin(), moment_tensor_vals.end(), 0.0);
        std::fill(nbh_energy_ders_wrt_moments.begin(), nbh_energy_ders_wrt_moments.end(), 0.0);
        if (need_radial_jacobian)
            std::fill(radial_jacobian.begin(), radial_jacobian.end(), 0.0);

        // ------------ Calculate Basic Moments ------------
        for (int jj = 0; jj < jnum; jj++) {
            const int j = nbrs[jj];
            const int jtype = list.types[j];
            const double r[3] = {dr[jj * 3 + 0], dr[jj * 3 + 1], dr[jj * 3 + 2]};
            const double rsq = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];

            if (rsq > max_cutoff_sq) continue;
            cached_j[valid_count] = j;
            valid_dr[valid_count] = {r[0], r[1], r[2]};

            const double dist = std::sqrt(rsq);
            radial_basis->calc_radial_basis_ders(dist);

            for (int k = 1; k < max_alpha_index_basic; k++) {
                dist_powers[k] = dist_powers[k - 1] * dist;
                for (int a = 0; a < 3; a++) coord_powers[k][a] = coord_powers[k - 1][a] * r[a];
            }

            const int pair_offset = itype * species_count + jtype;
            for (int mu = 0; mu < radial_func_count; mu++) {
                double val = 0;
                double der = 0;
                const int offset = (pair_offset * radial_coeff_count_per_pair) + mu * radial_basis_size;

                for (int ri = 0; ri < radial_basis_size; ri++) {
                    val += radial_basis_coeffs[offset + ri] * radial_basis->radial_basis_vals[ri];
                    der += radial_basis_coeffs[offset + ri] * radial_basis->radial_basis_ders[ri];
                }
                radial_vals[mu] = val;
                radial_ders[mu] = der;
            }

            for (int k = 0; k < alpha_index_basic_count; k++) {
                const int mu = alpha_index_basic[k][0];
                const int px = alpha_index_basic[k][1];
                const int py = alpha_index_basic[k][2];
                const int pz = alpha_index_basic[k][3];

                double val = radial_vals[mu];
                double der = radial_ders[mu];

                const int norm_rank = px + py + pz;
                const double norm_fac = 1.0 / dist_powers[norm_rank];
                const double pow0 = coord_powers[px][0];
                const double pow1 = coord_powers[py][1];
                const double pow2 = coord_powers[pz][2];
                double pow = pow0 * pow1 * pow2;

                if (need_radial_jacobian) {
                    double* jac_row = radial_jacobian.data() + ((size_t) k * species_count + jtype) * radial_coeff_count_per_pair + mu * radial_basis_size;
                    for (int ri = 0; ri < radial_basis_size; ri++)
                        jac_row[ri] += radial_basis->radial_basis_vals[ri] * norm_fac * pow;
                }

                // Angular scalar factor and its derivative w.r.t. this displacement
                if (need_angular) {
                    const double angfac = pow * norm_fac;
                    const size_t slot = (size_t) valid_count * alpha_index_basic_count + k;
                    angular_values[slot] = angfac;
                    double* ajac = angular_jacobians.data() + slot * 3;
                    ajac[0] = ajac[1] = ajac[2] = 0.0;
                    if (px != 0) ajac[0] += norm_fac * px * coord_powers[px - 1][0] * pow1 * pow2;
                    if (py != 0) ajac[1] += norm_fac * py * pow0 * coord_powers[py - 1][1] * pow2;
                    if (pz != 0) ajac[2] += norm_fac * pz * pow0 * pow1 * coord_powers[pz - 1][2];
                    const double rank_fac = norm_rank * angfac / rsq;
                    ajac[0] -= rank_fac * r[0];
                    ajac[1] -= rank_fac * r[1];
                    ajac[2] -= rank_fac * r[2];
                }

                val *= norm_fac;
                der = der * norm_fac - norm_rank * val / dist;
                moment_tensor_vals[k] += val * pow;

                const size_t jac = (size_t) valid_count * alpha_index_basic_count + k;
                pow *= der / dist;
                moment_jacobian[jac][0] = pow * r[0];
                moment_jacobian[jac][1] = pow * r[1];
                moment_jacobian[jac][2] = pow * r[2];
                if (px != 0) moment_jacobian[jac][0] += val * px * coord_powers[px - 1][0] * pow1 * pow2;
                if (py != 0) moment_jacobian[jac][1] += val * py * pow0 * coord_powers[py - 1][1] * pow2;
                if (pz != 0) moment_jacobian[jac][2] += val * pz * pow0 * pow1 * coord_powers[pz - 1][2];
            }
            valid_count++;
        }

        // ------------ Contruct Composite Moment Values  ------------
        for (int k = 0; k < alpha_index_times_count; k++) {
            double val0 = moment_tensor_vals[alpha_index_times[k][0]];
            double val1 = moment_tensor_vals[alpha_index_times[k][1]];
            int val2 = alpha_index_times[k][2];
            moment_tensor_vals[alpha_index_times[k][3]] += val2 * val0 * val1;
        }

        // =========== Begin Backpropogation ===========
        for (int k = 0; k < alpha_scalar_count; k++)
            nbh_energy_ders_wrt_moments[alpha_moment_mapping[k]] = linear_coeffs[k];

        for (int k = alpha_index_times_count - 1; k >= 0; k--) {
            int a0 = alpha_index_times[k][0];
            int a1 = alpha_index_times[k][1];
            int multipiler = alpha_index_times[k][2];
            int a3 = alpha_index_times[k][3];

            double val0 = moment_tensor_vals[a0];
            double val1 = moment_tensor_vals[a1];
            double val3 = nbh_energy_ders_wrt_moments[a3];

            nbh_energy_ders_wrt_moments[a1] += val3 * multipiler * val0;
            nbh_energy_ders_wrt_moments[a0] += val3 * multipiler * val1;
        }

        // ---- Per-atom site energy gradient ----
        if (site_e_grad) {
            double* row = site_e_grad + (size_t) ii * width;
            for (int s = 0; s < n_linear; s++)
                row[lin_off + s] = moment_tensor_vals[alpha_moment_mapping[s]];
            row[sp_off + itype] = 1.0;
            for (int k = 0; radial && k < alpha_index_basic_count; k++) {
                const double der = nbh_energy_ders_wrt_moments[k];
                if (der == 0.0) continue;
                for (int jtype = 0; jtype < species_count; jtype++) {
                    const int offset = rad_off + (itype * species_count + jtype) * radial_coeff_count_per_pair;
                    const double* jac = radial_jacobian.data() + ((size_t) k * species_count + jtype) * radial_coeff_count_per_pair;
                    for (int ri = 0; ri < radial_coeff_count_per_pair; ri++)
                        row[offset + ri] += der * jac[ri];
                }
            }
        }

        if (!force_grad)
            continue;

        // ---- Linear force / virial gradient ----
        // dF/dbeta_s is the derivative of basis value s w.r.t. the displacements,
        // carried forward from the basic moments' Jacobian through the products.
        const int n_ders = valid_count * 3;
        if (moment_ders.size() < (size_t) alpha_moment_count * n_ders)
            moment_ders.resize((size_t) alpha_moment_count * n_ders);
        std::fill(moment_ders.begin(), moment_ders.begin() + (size_t) alpha_moment_count * n_ders, 0.0);
        for (int k = 0; k < alpha_index_basic_count; k++)
            for (int jj = 0; jj < valid_count; jj++)
                for (int d = 0; d < 3; d++)
                    moment_ders[(size_t) k * n_ders + jj * 3 + d] = moment_jacobian[(size_t) jj * alpha_index_basic_count + k][d];

        for (int k = 0; k < alpha_index_times_count; k++) {
            const int a0 = alpha_index_times[k][0], a1 = alpha_index_times[k][1];
            const int mul = alpha_index_times[k][2], a3 = alpha_index_times[k][3];
            const double M_a0 = moment_tensor_vals[a0], M_a1 = moment_tensor_vals[a1];
            const double* D_a0 = moment_ders.data() + (size_t) a0 * n_ders;
            const double* D_a1 = moment_ders.data() + (size_t) a1 * n_ders;
            double* D_a3 = moment_ders.data() + (size_t) a3 * n_ders;
            for (int q = 0; q < n_ders; q++)
                D_a3[q] += mul * (D_a0[q] * M_a1 + M_a0 * D_a1[q]);
        }

        for (int jj = 0; jj < valid_count; jj++) {
            const int j = cached_j[jj];
            const auto& r = valid_dr[jj];
            for (int s = 0; s < n_linear; s++) {
                const double* D = moment_ders.data() + (size_t) alpha_moment_mapping[s] * n_ders + jj * 3;
                const int col = lin_off + s;
                for (int d = 0; d < 3; d++) {
                    force_grad[((size_t) i * 3 + d) * width + col] += D[d];
                    force_grad[((size_t) j * 3 + d) * width + col] -= D[d];
                }
                if (virial_grad) {
                    virial_grad[0 * width + col] -= D[0] * r[0];
                    virial_grad[1 * width + col] -= D[1] * r[1];
                    virial_grad[2 * width + col] -= D[2] * r[2];
                    virial_grad[3 * width + col] -= (D[0] * r[1] + D[1] * r[0]) / 2;
                    virial_grad[4 * width + col] -= (D[0] * r[2] + D[2] * r[0]) / 2;
                    virial_grad[5 * width + col] -= (D[1] * r[2] + D[2] * r[1]) / 2;
                }
            }
        }

        if (!radial)
            continue;

        // ---- Radial force / virial gradient ----
        scatter_radial_jacobian(itype);
        propagate_radial_moment_ders();

        // Term 2: the radial basis itself depends on the coefficients.
        for (int jj = 0; jj < valid_count; jj++) {
            const int j = cached_j[jj];
            const int jtype = list.types[j];
            const auto& r = valid_dr[jj];
            const double dist = std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
            radial_basis->calc_radial_basis_ders(dist);
            const double unit_r[3] = {r[0] / dist, r[1] / dist, r[2] / dist};
            const int pair_off = itype * species_count + jtype;

            for (int k = 0; k < alpha_index_basic_count; k++) {
                const double G_k = nbh_energy_ders_wrt_moments[k];
                if (G_k == 0.0) continue;
                const int mu = alpha_index_basic[k][0];
                const size_t slot = (size_t) jj * alpha_index_basic_count + k;
                const double angular_factor = angular_values[slot];
                const double* angular_jacobian = angular_jacobians.data() + slot * 3;
                const int coeff_offset = rad_off + (pair_off * radial_func_count + mu) * radial_basis_size;

                for (int d = 0; d < 3; d++) {
                    for (int ri = 0; ri < radial_basis_size; ri++) {
                        const double contrib = G_k * (radial_basis->radial_basis_ders[ri] * unit_r[d] * angular_factor + radial_basis->radial_basis_vals[ri] * angular_jacobian[d]);
                        const int col = coeff_offset + ri;
                        force_grad[((size_t) i * 3 + d) * width + col] += contrib;
                        force_grad[((size_t) j * 3 + d) * width + col] -= contrib;
                        if (virial_grad) accumulate_virial_grad(virial_grad, width, d, col, contrib, r);
                    }
                }
            }
        }

        // Term 1: the moment Jacobian is weighted by coefficient-dependent dG.
        for (int jj = 0; jj < valid_count; jj++) {
            const int j = cached_j[jj];
            const auto& r = valid_dr[jj];
            for (int k = 0; k < alpha_index_basic_count; k++) {
                const double* dG_k = dG.data() + (size_t) k * n_radial;
                const auto& jac_k = moment_jacobian[(size_t) jj * alpha_index_basic_count + k];
                for (int d = 0; d < 3; d++) {
                    const double jkd = jac_k[d];
                    if (jkd == 0.0) continue;
                    const size_t i_base = ((size_t) i * 3 + d) * width + rad_off;
                    const size_t j_base = ((size_t) j * 3 + d) * width + rad_off;
                    for (int ri = 0; ri < n_radial; ri++) {
                        const double contrib = dG_k[ri] * jkd;
                        force_grad[i_base + ri] += contrib;
                        force_grad[j_base + ri] -= contrib;
                        if (virial_grad) accumulate_virial_grad(virial_grad, width, d, rad_off + ri, contrib, r);
                    }
                }
            }
        }
    }
}

/* ----------------------------------------------------------------------
   Per site, the loss is a E_i + sum_j w_j . dE_i/dr_ij with a = dL/dE and
   w_j = dL/dF_i - dL/dF_j - S r_ij, S the symmetric dL/dvirial. The forward
   pass carries each moment's derivative along w (its tangent) beside its
   value. Backward, the tangents' adjoints are nbh_energy_ders_wrt_moments,
   and the values' adjoints pick up the tangents through the products. The
   angular factors are the monomials of the unit displacement u, built
   through PairMTP's shared tree, and their tangents follow the same tree
   along v = (w - (w.u) u) / |r|, the tangent of u.
------------------------------------------------------------------------- */
void MTPTraining::eval_loss_grad(const NeighList& list, double dloss_denergy, const double* dloss_dforces, const double* dloss_dvirial, double* loss_grad) {
    const int n_radial = radial_coeff_count;
    const int sp_off = n_radial;
    const int lin_off = n_radial + species_count;
    const double a = dloss_denergy;

    // dL/dvirial as a symmetric matrix; each off-diagonal component is shared by two entries.
    double S[3][3] = {};
    if (dloss_dvirial) {
        S[0][0] = dloss_dvirial[0];
        S[1][1] = dloss_dvirial[1];
        S[2][2] = dloss_dvirial[2];
        S[0][1] = S[1][0] = dloss_dvirial[3] / 2;
        S[0][2] = S[2][0] = dloss_dvirial[4] / 2;
        S[1][2] = S[2][1] = dloss_dvirial[5] / 2;
    }

    moment_tangents.resize(alpha_moment_count);
    moment_adjoints.resize(alpha_moment_count);
    angular_tangents.assign(angular_count, 0.0);
    basic_adjoints_by_mu.resize(alpha_index_basic_count);

    int nbr_offset = 0;

    for (int ii = 0; ii < list.inum; ii++) {
        const int i = list.ilist[ii];
        const int itype = list.types[i];
        const int jnum = list.numneigh[ii];
        const int* nbrs = list.firstneigh + nbr_offset;
        const double* dr = list.displacements + nbr_offset * 3;
        nbr_offset += jnum;

        int valid_count = 0;

        if (cache_size < jnum) {
            neighbor_cache.resize((size_t) jnum * (1 + 2 * radial_func_count));
            cached_j.resize(jnum);
            valid_dr.resize(jnum);
            cache_size = jnum;
        }
        if (w_dot_unit_r.size() < (size_t) jnum) {
            neighbor_radial_vals.resize((size_t) jnum * radial_basis_size);
            neighbor_radial_ders.resize((size_t) jnum * radial_basis_size);
            unit_displacements.resize(jnum);
            unit_tangents.resize(jnum);
            w_dot_unit_r.resize(jnum);
        }

        std::fill(moment_tensor_vals.begin(), moment_tensor_vals.end(), 0.0);
        std::fill(moment_tangents.begin(), moment_tangents.end(), 0.0);
        std::fill(nbh_energy_ders_wrt_moments.begin(), nbh_energy_ders_wrt_moments.end(), 0.0);
        std::fill(moment_adjoints.begin(), moment_adjoints.end(), 0.0);

        // ------------ Basic moments and their tangents ------------
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
            radial_basis->calc_radial_basis_ders(dist);
            double* basis_vals = neighbor_radial_vals.data() + (size_t) valid_count * radial_basis_size;
            double* basis_ders = neighbor_radial_ders.data() + (size_t) valid_count * radial_basis_size;
            std::copy(radial_basis->radial_basis_vals.begin(), radial_basis->radial_basis_vals.end(), basis_vals);
            std::copy(radial_basis->radial_basis_ders.begin(), radial_basis->radial_basis_ders.end(), basis_ders);

            // The loss weight of this displacement
            double w[3] = {0.0, 0.0, 0.0};
            if (dloss_dforces)
                for (int d = 0; d < 3; d++) w[d] = dloss_dforces[i * 3 + d] - dloss_dforces[j * 3 + d];
            for (int d = 0; d < 3; d++) w[d] -= S[d][0] * r[0] + S[d][1] * r[1] + S[d][2] * r[2];
            const double w_r = (w[0] * r[0] + w[1] * r[1] + w[2] * r[2]) * inv_dist;
            w_dot_unit_r[valid_count] = w_r;

            auto& u = unit_displacements[valid_count];
            auto& v = unit_tangents[valid_count];
            for (int d = 0; d < 3; d++) {
                u[d] = r[d] * inv_dist;
                v[d] = (w[d] - w_r * u[d]) * inv_dist;
            }
            for (int k = 1; k < angular_count; k++) {
                const int parent = angular_parent[k], axis = angular_axis[k];
                angular_vals[k] = angular_vals[parent] * u[axis];
                angular_tangents[k] = angular_tangents[parent] * u[axis] + angular_vals[parent] * v[axis];
            }

            const int pair_offset = itype * species_count + jtype;
            for (int mu = 0; mu < radial_func_count; mu++) {
                double val = 0;
                double der = 0;
                const int offset = (pair_offset * radial_coeff_count_per_pair) + mu * radial_basis_size;

                for (int ri = 0; ri < radial_basis_size; ri++) {
                    val += radial_basis_coeffs[offset + ri] * basis_vals[ri];
                    der += radial_basis_coeffs[offset + ri] * basis_ders[ri];
                }

                for (int t = mu_offsets[mu]; t < mu_offsets[mu + 1]; t++) {
                    const int k = basic_by_mu[t];
                    const double ang = angular_vals[angular_by_mu[t]];
                    moment_tensor_vals[k] += val * ang;
                    moment_tangents[k] += der * w_r * ang + val * angular_tangents[angular_by_mu[t]];
                }
            }
            valid_count++;
        }

        // ------------ Composite moments and their tangents ------------
        for (int k = 0; k < alpha_index_times_count; k++) {
            const int a0 = alpha_index_times[k][0], a1 = alpha_index_times[k][1];
            const int mul = alpha_index_times[k][2], a3 = alpha_index_times[k][3];
            const double M_a0 = moment_tensor_vals[a0], M_a1 = moment_tensor_vals[a1];
            moment_tangents[a3] += mul * (moment_tangents[a0] * M_a1 + M_a0 * moment_tangents[a1]);
            moment_tensor_vals[a3] += mul * M_a0 * M_a1;
        }

        // =========== Backpropagation of values and tangents ===========
        for (int s = 0; s < alpha_scalar_count; s++) {
            nbh_energy_ders_wrt_moments[alpha_moment_mapping[s]] = linear_coeffs[s];
            moment_adjoints[alpha_moment_mapping[s]] = a * linear_coeffs[s];
        }

        for (int k = alpha_index_times_count - 1; k >= 0; k--) {
            const int a0 = alpha_index_times[k][0], a1 = alpha_index_times[k][1];
            const int mul = alpha_index_times[k][2], a3 = alpha_index_times[k][3];
            const double M_a0 = moment_tensor_vals[a0], M_a1 = moment_tensor_vals[a1];
            const double G_a3 = nbh_energy_ders_wrt_moments[a3];
            const double A_a3 = moment_adjoints[a3];
            moment_adjoints[a0] += mul * (A_a3 * M_a1 + G_a3 * moment_tangents[a1]);
            moment_adjoints[a1] += mul * (A_a3 * M_a0 + G_a3 * moment_tangents[a0]);
            nbh_energy_ders_wrt_moments[a1] += G_a3 * mul * M_a0;
            nbh_energy_ders_wrt_moments[a0] += G_a3 * mul * M_a1;
        }

        // ---- Species and linear coefficients ----
        loss_grad[sp_off + itype] += a;
        for (int s = 0; s < alpha_scalar_count; s++)
            loss_grad[lin_off + s] += a * moment_tensor_vals[alpha_moment_mapping[s]] + moment_tangents[alpha_moment_mapping[s]];

        // ---- Radial coefficients ----
        for (int t = 0; t < alpha_index_basic_count; t++) {
            basic_ders_by_mu[t] = nbh_energy_ders_wrt_moments[basic_by_mu[t]];
            basic_adjoints_by_mu[t] = moment_adjoints[basic_by_mu[t]];
        }

        for (int jj = 0; jj < valid_count; jj++) {
            const int jtype = list.types[cached_j[jj]];
            const auto& u = unit_displacements[jj];
            const auto& v = unit_tangents[jj];
            for (int k = 1; k < angular_count; k++) {
                const int parent = angular_parent[k], axis = angular_axis[k];
                angular_vals[k] = angular_vals[parent] * u[axis];
                angular_tangents[k] = angular_tangents[parent] * u[axis] + angular_vals[parent] * v[axis];
            }

            const double w_r = w_dot_unit_r[jj];
            const double* basis_vals = neighbor_radial_vals.data() + (size_t) jj * radial_basis_size;
            const double* basis_ders = neighbor_radial_ders.data() + (size_t) jj * radial_basis_size;
            double* row = loss_grad + (size_t) (itype * species_count + jtype) * radial_coeff_count_per_pair;
            for (int mu = 0; mu < radial_func_count; mu++) {
                // adjoints of the radial function's value and derivative
                double value_adjoint = 0.0, der_adjoint = 0.0;
                for (int t = mu_offsets[mu]; t < mu_offsets[mu + 1]; t++) {
                    const double ang = angular_vals[angular_by_mu[t]];
                    value_adjoint += basic_adjoints_by_mu[t] * ang + basic_ders_by_mu[t] * angular_tangents[angular_by_mu[t]];
                    der_adjoint += basic_ders_by_mu[t] * ang;
                }
                der_adjoint *= w_r;
                for (int ri = 0; ri < radial_basis_size; ri++)
                    row[mu * radial_basis_size + ri] += value_adjoint * basis_vals[ri] + der_adjoint * basis_ders[ri];
            }
        }
    }
}
