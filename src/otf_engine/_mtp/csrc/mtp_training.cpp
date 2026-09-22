/* -*- c++ -*- ----------------------------------------------------------
   Derivatives of energy, forces and virial with respect to the MTP
   coefficients.
------------------------------------------------------------------------- */

#include "mtp_training.h"

#include <cmath>

int MTPTraining::cols_width(Cols cols) const {
    if (cols == RADIAL) return radial_coeff_count;
    if (cols == LINEAR) return alpha_scalar_count;
    return coeff_count();
}

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
double MTPTraining::compute_efs_grad(const NeighList& list, Cols cols, double* forces, double* virial, double* site_e_grad, double* force_grad, double* virial_grad) {
    const int width = cols_width(cols);
    const int n_radial = radial_coeff_count;
    const int n_linear = alpha_scalar_count;

    // Column offsets of each coefficient block within a row of width `width`.
    const int rad_off = (cols == LINEAR) ? -1 : 0;
    const int sp_off = (cols == ALL) ? n_radial : -1;
    const int lin_off = (cols == RADIAL) ? -1 : ((cols == ALL) ? n_radial + species_count : 0);

    const bool want_radial = (rad_off >= 0);
    const bool want_linear = (lin_off >= 0);

    if (site_e_grad) std::fill(site_e_grad, site_e_grad + (size_t) list.inum * width, 0.0);
    if (force_grad) std::fill(force_grad, force_grad + (size_t) list.n_atoms * 3 * width, 0.0);
    if (virial_grad) std::fill(virial_grad, virial_grad + 6 * width, 0.0);

    // The forward pass fills radial_jacobian only when a radial block is wanted,
    // and angular factors only when a force gradient is.
    const bool need_radial_jacobian = want_radial && (site_e_grad || force_grad);
    const bool need_angular = want_radial && force_grad;

    if (need_radial_jacobian) {
        radial_jacobian.resize((size_t) alpha_index_basic_count * species_count * radial_coeff_count_per_pair);
        dM_dc.resize((size_t) alpha_moment_count * n_radial);
        dG.resize((size_t) alpha_moment_count * n_radial);
    }
    if (want_linear && force_grad)
        dG_lin.resize((size_t) alpha_moment_count * n_linear);

    double total_energy = 0.0;
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
        if (jac_size < jnum) {
            jac_size = jnum;
            moment_jacobian.resize((size_t) jac_size * alpha_index_basic_count);
            valid_j.resize(jac_size);
            valid_dr.resize(jac_size);
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
            valid_j[valid_count] = j;
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

        double nbh_energy = species_coeffs[itype];
        for (int k = 0; k < alpha_scalar_count; k++)
            nbh_energy += linear_coeffs[k] * moment_tensor_vals[alpha_moment_mapping[k]];
        total_energy += nbh_energy;

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

        // ---- Energy, forces and virial ----
        if (forces) {
            for (int jj = 0; jj < valid_count; jj++) {
                const int j = valid_j[jj];
                double tf[3] = {0, 0, 0};
                for (int k = 0; k < alpha_index_basic_count; k++)
                    for (int a = 0; a < 3; a++)
                        tf[a] += nbh_energy_ders_wrt_moments[k] * moment_jacobian[(size_t) jj * alpha_index_basic_count + k][a];

                forces[i * 3 + 0] += tf[0];
                forces[i * 3 + 1] += tf[1];
                forces[i * 3 + 2] += tf[2];
                forces[j * 3 + 0] -= tf[0];
                forces[j * 3 + 1] -= tf[1];
                forces[j * 3 + 2] -= tf[2];

                if (virial) {
                    const auto& r = valid_dr[jj];
                    virial[0] -= tf[0] * r[0];
                    virial[1] -= tf[1] * r[1];
                    virial[2] -= tf[2] * r[2];
                    virial[3] -= (tf[0] * r[1] + tf[1] * r[0]) / 2;
                    virial[4] -= (tf[0] * r[2] + tf[2] * r[0]) / 2;
                    virial[5] -= (tf[1] * r[2] + tf[2] * r[1]) / 2;
                }
            }
        }

        // ---- Per-atom site energy gradient ----
        if (site_e_grad) {
            double* row = site_e_grad + (size_t) ii * width;
            if (want_linear)
                for (int s = 0; s < n_linear; s++)
                    row[lin_off + s] = moment_tensor_vals[alpha_moment_mapping[s]];
            if (sp_off >= 0)
                row[sp_off + itype] = 1.0;
            if (want_radial) {
                for (int k = 0; k < alpha_index_basic_count; k++) {
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
        }

        if (!force_grad)
            continue;

        // ---- Radial force / virial gradient ----
        if (want_radial) {
            scatter_radial_jacobian(itype);
            propagate_radial_moment_ders();

            // Term 2: the radial basis itself depends on the coefficients.
            for (int jj = 0; jj < valid_count; jj++) {
                const int j = valid_j[jj];
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
                const int j = valid_j[jj];
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

        // ---- Linear force / virial gradient ----
        // The moments do not depend on beta, so only the backward seed does.
        if (want_linear) {
            std::fill(dG_lin.begin(), dG_lin.begin() + (size_t) alpha_moment_count * n_linear, 0.0);
            for (int s = 0; s < n_linear; s++)
                dG_lin[(size_t) alpha_moment_mapping[s] * n_linear + s] = 1.0;

            for (int k = alpha_index_times_count - 1; k >= 0; k--) {
                const int a0 = alpha_index_times[k][0], a1 = alpha_index_times[k][1];
                const int mul = alpha_index_times[k][2], a3 = alpha_index_times[k][3];
                const double M_a0 = moment_tensor_vals[a0], M_a1 = moment_tensor_vals[a1];
                const double* dGl_a3 = dG_lin.data() + (size_t) a3 * n_linear;
                double* dGl_a0 = dG_lin.data() + (size_t) a0 * n_linear;
                double* dGl_a1 = dG_lin.data() + (size_t) a1 * n_linear;
                for (int s = 0; s < n_linear; s++) {
                    dGl_a0[s] += dGl_a3[s] * mul * M_a1;
                    dGl_a1[s] += dGl_a3[s] * mul * M_a0;
                }
            }

            for (int jj = 0; jj < valid_count; jj++) {
                const int j = valid_j[jj];
                const auto& r = valid_dr[jj];
                for (int k = 0; k < alpha_index_basic_count; k++) {
                    const double* dGl_k = dG_lin.data() + (size_t) k * n_linear;
                    const auto& jac_k = moment_jacobian[(size_t) jj * alpha_index_basic_count + k];
                    for (int d = 0; d < 3; d++) {
                        const double jkd = jac_k[d];
                        if (jkd == 0.0) continue;
                        const size_t i_base = ((size_t) i * 3 + d) * width + lin_off;
                        const size_t j_base = ((size_t) j * 3 + d) * width + lin_off;
                        for (int s = 0; s < n_linear; s++) {
                            const double contrib = dGl_k[s] * jkd;
                            force_grad[i_base + s] += contrib;
                            force_grad[j_base + s] -= contrib;
                            if (virial_grad) accumulate_virial_grad(virial_grad, width, d, lin_off + s, contrib, r);
                        }
                    }
                }
            }
        }
    }

    return total_energy;
}

/* ----------------------------------------------------------------------
   Named entry points
------------------------------------------------------------------------- */
void MTPTraining::eval_grad_radial(const NeighList& list, double* energy_grad, double* force_grad, double* virial_grad) {
    const int n_radial = radial_coeff_count;
    site_grads.assign((size_t) list.inum * n_radial, 0.0);
    compute_efs_grad(list, RADIAL, nullptr, nullptr, site_grads.data(), force_grad, virial_grad);

    // energy_grad is the site-energy gradient summed over atoms
    std::fill(energy_grad, energy_grad + n_radial, 0.0);
    for (int ii = 0; ii < list.inum; ii++)
        for (int r = 0; r < n_radial; r++)
            energy_grad[r] += site_grads[(size_t) ii * n_radial + r];
}

void MTPTraining::eval_grad_linear(const NeighList& list, double* site_e_grad, double* force_grad, double* virial_grad) {
    compute_efs_grad(list, LINEAR, nullptr, nullptr, site_e_grad, force_grad, virial_grad);
}

void MTPTraining::eval_grad_all(const NeighList& list, double* site_e_grad, double* force_grad, double* virial_grad) {
    compute_efs_grad(list, ALL, nullptr, nullptr, site_e_grad, force_grad, virial_grad);
}

double MTPTraining::compute_with_radial_grad(const NeighList& list, double* forces, double* virial, double* energy_grad, double* force_grad, double* virial_grad) {
    const int n_radial = radial_coeff_count;
    site_grads.assign((size_t) list.inum * n_radial, 0.0);
    double energy = compute_efs_grad(list, RADIAL, forces, virial, site_grads.data(), force_grad, virial_grad);

    std::fill(energy_grad, energy_grad + n_radial, 0.0);
    for (int ii = 0; ii < list.inum; ii++)
        for (int r = 0; r < n_radial; r++)
            energy_grad[r] += site_grads[(size_t) ii * n_radial + r];

    return energy;
}
