/* -*- c++ -*- ----------------------------------------------------------
   Basis values of an MTP, and derivatives of its site energies, forces and
   virial, and of a loss built on them, with respect to its coefficients.
------------------------------------------------------------------------- */

#include "mtp.h"

#include <algorithm>
#include <cmath>

// Adds the virial terms -r_a f_d, a <= d, of force component d to column col of virial_grad.
static inline void accumulate_virial_grad(double* virial_grad, int width, int d, int col, double contrib, const std::array<double, 3>& r) {
    virial_grad[d * width + col] -= r[d] * contrib;
    if (d == 1) {
        virial_grad[3 * width + col] -= r[0] * contrib;
    } else if (d == 2) {
        virial_grad[4 * width + col] -= r[0] * contrib;
        virial_grad[5 * width + col] -= r[1] * contrib;
    }
}

/* ---------------------------------------------------------------------- */
void MTP::eval_basis(const NeighList& list, double* basis_out) {
    int nbr_offset = 0;

    for (int ii = 0; ii < list.inum; ii++) {
        const int i = list.ilist[ii];
        const int itype = list.types[i];
        const int jnum = list.numneigh[ii];
        const int* nbrs = list.firstneigh + nbr_offset;
        const double* dr = list.displacements + nbr_offset * 3;
        nbr_offset += jnum;

        std::fill(atom_moment_vals.begin(), atom_moment_vals.end(), 0.0);

        // Basic moments
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

            for (int k = 1; k < angular_count; k++)
                angular_vals[k] = angular_vals[angular_parent[k]] * u[angular_axis[k]];

            const int pair_offset = itype * species_count + jtype;
            for (int mu = 0; mu < radial_func_count; mu++) {
                double val = 0;
                const int offset = (pair_offset * radial_coeff_count_per_pair) + mu * radial_basis_size;

                for (int ri = 0; ri < radial_basis_size; ri++)
                    val += radial_basis_coeffs[offset + ri] * basis_vals[ri];

                for (int t = mu_offsets[mu]; t < mu_offsets[mu + 1]; t++)
                    atom_moment_vals[basic_by_mu[t]] += val * angular_vals[angular_by_mu[t]];
            }
        }

        // Composite moments
        for (int k = 0; k < alpha_index_times_count; k++) {
            const int* term = alpha_index_times[k].data();
            atom_moment_vals[term[3]] +=
                term[2] * atom_moment_vals[term[0]] * atom_moment_vals[term[1]];
        }

        double* row = basis_out + (size_t) ii * alpha_scalar_count;
        for (int k = 0; k < alpha_scalar_count; k++)
            row[k] = atom_moment_vals[alpha_moment_mapping[k]];
    }
}

/* ---------------------------------------------------------------------- */
void MTP::grow_neighbor_rows(int jnum) {
    if (rows_size >= jnum) return;
    rows_size = jnum;
    neighbor_units.resize((size_t) 4 * rows_size);
    neighbor_angulars.resize((size_t) angular_count * rows_size);
    neighbor_radial_funcs.resize((size_t) radial_func_count * 2 * rows_size);
    neighbor_unit_tangents.resize((size_t) 4 * rows_size);
    neighbor_angular_tangents.resize((size_t) angular_count * rows_size);
    neighbor_radial_vals.resize((size_t) radial_basis_size * rows_size);
    neighbor_radial_ders.resize((size_t) radial_basis_size * rows_size);
}

/* ----------------------------------------------------------------------
   The site-energy gradient contracts the adjoints of the basic moments with
   the radial basis, neighbor by neighbor. The force gradient carries the
   moment Jacobian, the derivative of each basic moment with respect to the
   neighbor displacement, forward through the products. The derivatives of
   the angular monomials follow the monomial tree, with
   du_b/dr_a = (delta_ab - u_a u_b) / |r|.
------------------------------------------------------------------------- */
void MTP::eval_grad(const NeighList& list, double* site_e_grad, double* force_grad, double* virial_grad, bool radial) {
    const int width = coeff_count();
    const int n_radial = radial_coeff_count;
    const int n_linear = alpha_scalar_count;

    // Column offsets of the coefficient blocks in a row of width `width`
    const int rad_off = 0;
    const int sp_off = n_radial;
    const int lin_off = n_radial + species_count;

    if (site_e_grad) std::fill(site_e_grad, site_e_grad + (size_t) list.inum * width, 0.0);
    if (force_grad) std::fill(force_grad, force_grad + (size_t) list.n_atoms * 3 * width, 0.0);
    if (virial_grad) std::fill(virial_grad, virial_grad + 6 * width, 0.0);

    if (first_product.size() != (size_t) alpha_index_times_count) {
        std::vector<char> written(alpha_moment_count, 0);
        first_product.resize(alpha_index_times_count);
        for (int k = 0; k < alpha_index_times_count; k++) {
            first_product[k] = !written[alpha_index_times[k][3]];
            written[alpha_index_times[k][3]] = 1;
        }
    }

    // The radial force gradient takes the derivatives of the basic moments with respect to the radial coefficients.
    const bool need_radial_jacobian = radial && force_grad;
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

        grow_cache(jnum);
        grow_neighbor_rows(jnum);
        if (force_grad && jac_size < jnum) {
            jac_size = jnum;
            neighbor_angular_grads.resize((size_t) angular_count * 3 * jac_size);
            moment_ders.resize((size_t) alpha_moment_count * jac_size * 3);
        }
        const int ders_stride = 3 * jac_size;    // a row of moment_ders
        const int N = jnum;                      // a row of the neighbor arrays

        std::fill(atom_moment_vals.begin(), atom_moment_vals.end(), 0.0);
        std::fill(atom_energy_ders_wrt_moments.begin(), atom_energy_ders_wrt_moments.end(), 0.0);
        if (need_radial_jacobian)
            std::fill(radial_jacobian.begin(), radial_jacobian.end(), 0.0);

        // Unit displacement and radial basis of the neighbors within the cutoff
        double* u[3] = {neighbor_units.data(), neighbor_units.data() + N, neighbor_units.data() + 2 * N};
        double* inv_dists = neighbor_units.data() + 3 * N;
        for (int jj = 0; jj < jnum; jj++) {
            const int j = nbrs[jj];
            const double r[3] = {dr[jj * 3 + 0], dr[jj * 3 + 1], dr[jj * 3 + 2]};
            const double rsq = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];

            if (rsq > max_cutoff_sq) continue;
            cached_j[valid_count] = j;
            valid_dr[valid_count] = {r[0], r[1], r[2]};

            const double dist = std::sqrt(rsq);
            const double inv_dist = 1.0 / dist;
            for (int a = 0; a < 3; a++) u[a][valid_count] = r[a] * inv_dist;
            inv_dists[valid_count] = inv_dist;
            radial_basis->calc_radial_basis_ders(dist);
            std::copy(radial_basis->radial_basis_vals.begin(), radial_basis->radial_basis_vals.end(), neighbor_radial_vals.data() + (size_t) valid_count * radial_basis_size);
            std::copy(radial_basis->radial_basis_ders.begin(), radial_basis->radial_basis_ders.end(), neighbor_radial_ders.data() + (size_t) valid_count * radial_basis_size);
            valid_count++;
        }
        const int V = valid_count;

        // Angular monomials, their derivatives with respect to r for the force passes, and the radial functions
        double* ang = neighbor_angulars.data();
        for (int jj = 0; jj < V; jj++) ang[jj] = 1.0;
        for (int k = 1; k < angular_count; k++) {
            const double* parent = ang + (size_t) angular_parent[k] * N;
            const double* u_axis = u[angular_axis[k]];
            double* out = ang + (size_t) k * N;
            for (int jj = 0; jj < V; jj++) out[jj] = parent[jj] * u_axis[jj];
        }
        double* ang_grads = neighbor_angular_grads.data();
        if (force_grad) {
            for (int a = 0; a < 3; a++)
                for (int jj = 0; jj < V; jj++) ang_grads[(size_t) a * N + jj] = 0.0;
            for (int k = 1; k < angular_count; k++) {
                const int parent = angular_parent[k], axis = angular_axis[k];
                const double* parent_val = ang + (size_t) parent * N;
                for (int a = 0; a < 3; a++) {
                    const double* parent_grad = ang_grads + ((size_t) parent * 3 + a) * N;
                    double* out = ang_grads + ((size_t) k * 3 + a) * N;
                    const double delta = a == axis ? 1.0 : 0.0;
                    for (int jj = 0; jj < V; jj++)
                        out[jj] = parent_grad[jj] * u[axis][jj] + parent_val[jj] * (delta - u[a][jj] * u[axis][jj]) * inv_dists[jj];
                }
            }
        }
        double* radial_funcs = neighbor_radial_funcs.data();    // value of mu at row 2 mu, derivative at 2 mu + 1
        for (int jj = 0; jj < V; jj++) {
            const int pair_offset = itype * species_count + list.types[cached_j[jj]];
            const double* basis_vals = neighbor_radial_vals.data() + (size_t) jj * radial_basis_size;
            const double* basis_ders = neighbor_radial_ders.data() + (size_t) jj * radial_basis_size;
            for (int mu = 0; mu < radial_func_count; mu++) {
                double val = 0;
                double der = 0;
                const int offset = (pair_offset * radial_coeff_count_per_pair) + mu * radial_basis_size;
                for (int ri = 0; ri < radial_basis_size; ri++) {
                    val += radial_basis_coeffs[offset + ri] * basis_vals[ri];
                    der += radial_basis_coeffs[offset + ri] * basis_ders[ri];
                }
                radial_funcs[(size_t) 2 * mu * N + jj] = val;
                radial_funcs[(size_t) (2 * mu + 1) * N + jj] = der;
            }
        }

        // Basic moments, their Jacobian and their radial Jacobian
        for (int mu = 0; mu < radial_func_count; mu++) {
            const double* val = radial_funcs + (size_t) 2 * mu * N;
            const double* der = val + N;
            for (int t = mu_offsets[mu]; t < mu_offsets[mu + 1]; t++) {
                const int k = basic_by_mu[t];
                const int angular = angular_by_mu[t];
                const double* ang_k = ang + (size_t) angular * N;
                double moment = atom_moment_vals[k];
                for (int jj = 0; jj < V; jj++) moment += val[jj] * ang_k[jj];
                atom_moment_vals[k] = moment;

                if (force_grad)
                    for (int a = 0; a < 3; a++) {
                        double* jac = moment_ders.data() + (size_t) k * ders_stride + a * V;
                        const double* grad = ang_grads + ((size_t) angular * 3 + a) * N;
                        for (int jj = 0; jj < V; jj++) jac[jj] = der[jj] * u[a][jj] * ang_k[jj] + val[jj] * grad[jj];
                    }
                if (need_radial_jacobian)
                    for (int jj = 0; jj < V; jj++) {
                        const double* basis_vals = neighbor_radial_vals.data() + (size_t) jj * radial_basis_size;
                        double* jac_row = radial_jacobian.data() + ((size_t) k * species_count + list.types[cached_j[jj]]) * radial_coeff_count_per_pair + mu * radial_basis_size;
                        for (int ri = 0; ri < radial_basis_size; ri++)
                            jac_row[ri] += basis_vals[ri] * ang_k[jj];
                    }
            }
        }

        // Composite moments
        for (int k = 0; k < alpha_index_times_count; k++) {
            const int* term = alpha_index_times[k].data();
            atom_moment_vals[term[3]] +=
                term[2] * atom_moment_vals[term[0]] * atom_moment_vals[term[1]];
        }

        // Backpropagation, for the radial columns
        if (radial) {
            for (int k = 0; k < alpha_scalar_count; k++)
                atom_energy_ders_wrt_moments[alpha_moment_mapping[k]] = linear_coeffs[k];

            for (int k = alpha_index_times_count - 1; k >= 0; k--) {
                const int* term = alpha_index_times[k].data();
                const int a0 = term[0];
                const int a1 = term[1];

                const double w = term[2] * atom_energy_ders_wrt_moments[term[3]];

                atom_energy_ders_wrt_moments[a1] += w * atom_moment_vals[a0];
                atom_energy_ders_wrt_moments[a0] += w * atom_moment_vals[a1];
            }

            for (int t = 0; t < alpha_index_basic_count; t++)
                basic_ders_by_mu[t] = atom_energy_ders_wrt_moments[basic_by_mu[t]];
        }

        // Site-energy gradient
        if (site_e_grad) {
            double* row = site_e_grad + (size_t) ii * width;
            for (int s = 0; s < n_linear; s++)
                row[lin_off + s] = atom_moment_vals[alpha_moment_mapping[s]];
            row[sp_off + itype] = 1.0;

            for (int jj = 0; radial && jj < valid_count; jj++) {
                const double* basis_vals = neighbor_radial_vals.data() + (size_t) jj * radial_basis_size;
                double* pair_ders = row + rad_off + (itype * species_count + list.types[cached_j[jj]]) * radial_coeff_count_per_pair;

                for (int mu = 0; mu < radial_func_count; mu++) {
                    const int end = mu_offsets[mu + 1];
                    if (mu_offsets[mu] == end) continue;
                    double radial_sum = 0;
                    for (int t = mu_offsets[mu]; t < end; t++)
                        radial_sum += basic_ders_by_mu[t] * ang[(size_t) angular_by_mu[t] * N + jj];
                    double* coeff_ders = pair_ders + mu * radial_basis_size;
                    for (int ri = 0; ri < radial_basis_size; ri++)
                        coeff_ders[ri] += radial_sum * basis_vals[ri];
                }
            }
        }

        if (!force_grad)
            continue;

        // Linear force and virial gradient
        // dF/dbeta_s is the derivative of basis value s with respect to the displacements.
        // The products carry it forward from the moment Jacobian in the basic rows of moment_ders.
        // The first product into a moment sets its row, and the later ones add to it.
        const int n_ders = valid_count * 3;
        for (int k = 0; k < alpha_index_times_count; k++) {
            const int a0 = alpha_index_times[k][0], a1 = alpha_index_times[k][1];
            const double mul = alpha_index_times[k][2];
            const int a3 = alpha_index_times[k][3];
            const double M_a0 = atom_moment_vals[a0], M_a1 = atom_moment_vals[a1];
            const double* D_a0 = moment_ders.data() + (size_t) a0 * ders_stride;
            const double* D_a1 = moment_ders.data() + (size_t) a1 * ders_stride;
            double* D_a3 = moment_ders.data() + (size_t) a3 * ders_stride;
            if (first_product[k])
                for (int q = 0; q < n_ders; q++) D_a3[q] = mul * (D_a0[q] * M_a1 + M_a0 * D_a1[q]);
            else
                for (int q = 0; q < n_ders; q++) D_a3[q] += mul * (D_a0[q] * M_a1 + M_a0 * D_a1[q]);
        }

        // Row d * V + jj of scalar_ders holds the derivatives of the basis values with respect to component d of displacement jj.
        scalar_ders.resize((size_t) n_ders * n_linear);
        for (int s = 0; s < n_linear; s++) {
            const double* D = moment_ders.data() + (size_t) alpha_moment_mapping[s] * ders_stride;
            for (int q = 0; q < n_ders; q++) scalar_ders[(size_t) q * n_linear + s] = D[q];
        }
        for (int jj = 0; jj < valid_count; jj++) {
            const int j = cached_j[jj];
            const auto& r = valid_dr[jj];
            const double* D[3];
            for (int d = 0; d < 3; d++) {
                D[d] = scalar_ders.data() + (size_t) (d * V + jj) * n_linear;
                double* force_i = force_grad + ((size_t) i * 3 + d) * width + lin_off;
                double* force_j = force_grad + ((size_t) j * 3 + d) * width + lin_off;
                for (int s = 0; s < n_linear; s++) force_i[s] += D[d][s];
                for (int s = 0; s < n_linear; s++) force_j[s] -= D[d][s];
            }
            if (virial_grad) {
                double* v = virial_grad + lin_off;
                for (int s = 0; s < n_linear; s++) {
                    v[0 * width + s] -= r[0] * D[0][s];
                    v[1 * width + s] -= r[1] * D[1][s];
                    v[2 * width + s] -= r[2] * D[2][s];
                    v[3 * width + s] -= r[0] * D[1][s];
                    v[4 * width + s] -= r[0] * D[2][s];
                    v[5 * width + s] -= r[1] * D[2][s];
                }
            }
        }

        if (!radial)
            continue;

        // Radial force and virial gradient
        // Copies radial_jacobian into dM_dc. The columns of dM_dc are global radial coefficient indices.
        // The columns of radial_jacobian are per-pair indices in the row of the central species.
        std::fill(dM_dc.begin(), dM_dc.begin() + (size_t) alpha_moment_count * n_radial, 0.0);
        for (int k = 0; k < alpha_index_basic_count; k++)
            for (int jtype = 0; jtype < species_count; jtype++) {
                const double* src = radial_jacobian.data() + ((size_t) k * species_count + jtype) * radial_coeff_count_per_pair;
                double* dst = dM_dc.data() + (size_t) k * n_radial + (itype * species_count + jtype) * radial_coeff_count_per_pair;
                std::copy(src, src + radial_coeff_count_per_pair, dst);
            }

        // Forward propagate dM through the composite moments
        for (int k = 0; k < alpha_index_times_count; k++) {
            const int i0 = alpha_index_times[k][0], i1 = alpha_index_times[k][1];
            const int mul = alpha_index_times[k][2], i3 = alpha_index_times[k][3];
            const double M_i0 = atom_moment_vals[i0], M_i1 = atom_moment_vals[i1];
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
            const double G_a3 = atom_energy_ders_wrt_moments[a3];
            const double M_a0 = atom_moment_vals[a0], M_a1 = atom_moment_vals[a1];
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

        // Term 2: dE/dM_k times the derivative of the moment Jacobian with respect to the coefficients
        for (int jj = 0; jj < valid_count; jj++) {
            const int j = cached_j[jj];
            const int jtype = list.types[j];
            const auto& r = valid_dr[jj];
            const double* basis_vals = neighbor_radial_vals.data() + (size_t) jj * radial_basis_size;
            const double* basis_ders = neighbor_radial_ders.data() + (size_t) jj * radial_basis_size;
            const int pair_off = itype * species_count + jtype;

            for (int k = 0; k < alpha_index_basic_count; k++) {
                const double G_k = atom_energy_ders_wrt_moments[k];
                if (G_k == 0.0) continue;
                const int mu = alpha_index_basic[k][0];
                const double angular_factor = ang[(size_t) basic_to_angular[k] * N + jj];
                const double angular_jacobian[3] = {ang_grads[((size_t) basic_to_angular[k] * 3 + 0) * N + jj], ang_grads[((size_t) basic_to_angular[k] * 3 + 1) * N + jj], ang_grads[((size_t) basic_to_angular[k] * 3 + 2) * N + jj]};
                const int coeff_offset = rad_off + (pair_off * radial_func_count + mu) * radial_basis_size;

                for (int d = 0; d < 3; d++) {
                    for (int ri = 0; ri < radial_basis_size; ri++) {
                        const double contrib = G_k * (basis_ders[ri] * u[d][jj] * angular_factor + basis_vals[ri] * angular_jacobian[d]);
                        const int col = coeff_offset + ri;
                        force_grad[((size_t) i * 3 + d) * width + col] += contrib;
                        force_grad[((size_t) j * 3 + d) * width + col] -= contrib;
                        if (virial_grad) accumulate_virial_grad(virial_grad, width, d, col, contrib, r);
                    }
                }
            }
        }

        // Term 1: the moment Jacobian times dG
        for (int jj = 0; jj < valid_count; jj++) {
            const int j = cached_j[jj];
            const auto& r = valid_dr[jj];
            for (int k = 0; k < alpha_index_basic_count; k++) {
                const double* dG_k = dG.data() + (size_t) k * n_radial;
                const double* jac_k = moment_ders.data() + (size_t) k * ders_stride + jj;
                for (int d = 0; d < 3; d++) {
                    const double jkd = jac_k[d * V];
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
   w_j = dL/dF_i - dL/dF_j - S r_ij, where S[b][a] = dL/dvirial_ab and
   virial_ab = -sum r_a f_b over the pairs. The forward pass carries the
   tangent of each moment, its derivative along w, beside its value. In the
   backward pass, nbh_energy_ders_wrt_moments holds the adjoints of the
   tangents, and moment_adjoints, the adjoints of the values, picks up the
   tangents through the products. The angular monomials of the unit
   displacement u and their tangents follow the monomial tree along
   v = (w - (w.u) u) / |r|, the tangent of u. The central atoms go in
   chunks, and the gradient of a linear coefficient sums over the atoms of
   the chunk.
------------------------------------------------------------------------- */
void MTP::eval_loss_grad(const NeighList& list, double dloss_denergy, const double* dloss_dforces, const double* dloss_dvirial, double* loss_grad) {
    const int n_radial = radial_coeff_count;
    const int sp_off = n_radial;
    const int lin_off = n_radial + species_count;
    const double a = dloss_denergy;

    // S[b][a] = dL/dvirial_ab for a <= b
    double S[3][3] = {};
    if (dloss_dvirial) {
        S[0][0] = dloss_dvirial[0];
        S[1][1] = dloss_dvirial[1];
        S[2][2] = dloss_dvirial[2];
        S[1][0] = dloss_dvirial[3];
        S[2][0] = dloss_dvirial[4];
        S[2][1] = dloss_dvirial[5];
    }

    moment_tangents.resize(alpha_moment_count);
    moment_adjoints.resize(alpha_moment_count);
    basic_adjoints_by_mu.resize(alpha_index_basic_count);

    grow_chunk_cache(list);
    grow_neighbor_rows(chunk_size * atom_cache_size);
    const int N = atom_cache_size;    // a row of the neighbor arrays in an atom block

    int nbr_offset = 0;

    for (int ii_begin = 0; ii_begin < list.inum; ii_begin += chunk_size) {
        const int ii_end = std::min(ii_begin + chunk_size, list.inum);

        for (std::vector<Lanes>* lanes : {&moment_tensor_vals, &moment_tangents, &nbh_energy_ders_wrt_moments, &moment_adjoints})
            std::fill(lanes->begin(), lanes->end(), Lanes());

        for (int ii = ii_begin; ii < ii_end; ii++) {
            const int i = list.ilist[ii];
            const int itype = list.types[i];
            const int jnum = list.numneigh[ii];
            const int* nbrs = list.firstneigh + nbr_offset;
            const double* dr = list.displacements + nbr_offset * 3;
            nbr_offset += jnum;

            int valid_count = 0;
            const int atom = ii - ii_begin;

            select_atom_cache(atom);
            double* units = neighbor_units.data() + (size_t) atom * 4 * N;
            double* unit_tangents = neighbor_unit_tangents.data() + (size_t) atom * 4 * N;
            double* radial_vals = neighbor_radial_vals.data() + (size_t) atom * radial_basis_size * N;
            double* radial_ders = neighbor_radial_ders.data() + (size_t) atom * radial_basis_size * N;

            // Unit displacement, its tangent and radial basis of the neighbors within the cutoff
            double* u[3] = {units, units + N, units + 2 * N};
            double* v[3] = {unit_tangents, unit_tangents + N, unit_tangents + 2 * N};
            double* w_dot_unit_r = unit_tangents + 3 * N;
            for (int jj = 0; jj < jnum; jj++) {
                const int j = nbrs[jj];
                const double r[3] = {dr[jj * 3 + 0], dr[jj * 3 + 1], dr[jj * 3 + 2]};
                const double rsq = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];

                if (rsq > max_cutoff_sq) continue;
                cached_j[valid_count] = j;
                valid_dr[valid_count] = {r[0], r[1], r[2]};

                const double dist = std::sqrt(rsq);
                const double inv_dist = 1.0 / dist;
                radial_basis->calc_radial_basis_ders(dist);
                std::copy(radial_basis->radial_basis_vals.begin(), radial_basis->radial_basis_vals.end(), radial_vals + (size_t) valid_count * radial_basis_size);
                std::copy(radial_basis->radial_basis_ders.begin(), radial_basis->radial_basis_ders.end(), radial_ders + (size_t) valid_count * radial_basis_size);

                // Loss weight w of this displacement
                double w[3] = {0.0, 0.0, 0.0};
                if (dloss_dforces)
                    for (int d = 0; d < 3; d++) w[d] = dloss_dforces[i * 3 + d] - dloss_dforces[j * 3 + d];
                for (int d = 0; d < 3; d++) w[d] -= S[d][0] * r[0] + S[d][1] * r[1] + S[d][2] * r[2];
                const double w_r = (w[0] * r[0] + w[1] * r[1] + w[2] * r[2]) * inv_dist;
                w_dot_unit_r[valid_count] = w_r;

                for (int d = 0; d < 3; d++) {
                    u[d][valid_count] = r[d] * inv_dist;
                    v[d][valid_count] = (w[d] - w_r * u[d][valid_count]) * inv_dist;
                }
                valid_count++;
            }
            const int V = valid_count;
            cached_counts[atom] = valid_count;

            // Angular monomials, their tangents and the radial functions
            double* ang = neighbor_angulars.data() + (size_t) atom * angular_count * N;
            double* tangents = neighbor_angular_tangents.data() + (size_t) atom * angular_count * N;
            for (int jj = 0; jj < V; jj++) {
                ang[jj] = 1.0;
                tangents[jj] = 0.0;
            }
            for (int k = 1; k < angular_count; k++) {
                const int parent = angular_parent[k], axis = angular_axis[k];
                const double* parent_val = ang + (size_t) parent * N;
                const double* parent_tangent = tangents + (size_t) parent * N;
                double* out = ang + (size_t) k * N;
                double* out_tangent = tangents + (size_t) k * N;
                for (int jj = 0; jj < V; jj++) {
                    out[jj] = parent_val[jj] * u[axis][jj];
                    out_tangent[jj] = parent_tangent[jj] * u[axis][jj] + parent_val[jj] * v[axis][jj];
                }
            }
            double* radial_funcs = neighbor_radial_funcs.data() + (size_t) atom * radial_func_count * 2 * N;    // value of mu at row 2 mu, derivative at 2 mu + 1
            for (int jj = 0; jj < V; jj++) {
                const int pair_offset = itype * species_count + list.types[cached_j[jj]];
                const double* basis_vals = radial_vals + (size_t) jj * radial_basis_size;
                const double* basis_ders = radial_ders + (size_t) jj * radial_basis_size;
                for (int mu = 0; mu < radial_func_count; mu++) {
                    double val = 0;
                    double der = 0;
                    const int offset = (pair_offset * radial_coeff_count_per_pair) + mu * radial_basis_size;
                    for (int ri = 0; ri < radial_basis_size; ri++) {
                        val += radial_basis_coeffs[offset + ri] * basis_vals[ri];
                        der += radial_basis_coeffs[offset + ri] * basis_ders[ri];
                    }
                    radial_funcs[(size_t) 2 * mu * N + jj] = val;
                    radial_funcs[(size_t) (2 * mu + 1) * N + jj] = der;
                }
            }

            // Basic moments and their tangents
            for (int mu = 0; mu < radial_func_count; mu++) {
                const double* val = radial_funcs + (size_t) 2 * mu * N;
                const double* der = val + N;
                for (int t = mu_offsets[mu]; t < mu_offsets[mu + 1]; t++) {
                    const int k = basic_by_mu[t];
                    const double* ang_k = ang + (size_t) angular_by_mu[t] * N;
                    const double* tangent_k = tangents + (size_t) angular_by_mu[t] * N;
                    double moment = moment_tensor_vals[k][atom], tangent = moment_tangents[k][atom];
                    for (int jj = 0; jj < V; jj++) {
                        moment += val[jj] * ang_k[jj];
                        tangent += der[jj] * w_dot_unit_r[jj] * ang_k[jj] + val[jj] * tangent_k[jj];
                    }
                    moment_tensor_vals[k][atom] = moment;
                    moment_tangents[k][atom] = tangent;
                }
            }
        }

        // Composite moments and their tangents
        for (int k = 0; k < alpha_index_times_count; k++) {
            const int a0 = alpha_index_times[k][0], a1 = alpha_index_times[k][1];
            const int mul = alpha_index_times[k][2], a3 = alpha_index_times[k][3];
            moment_tangents[a3] += mul * moment_tangents[a0] * moment_tensor_vals[a1];
            moment_tangents[a3] += mul * moment_tensor_vals[a0] * moment_tangents[a1];
            moment_tensor_vals[a3] += mul * moment_tensor_vals[a0] * moment_tensor_vals[a1];
        }

        // Backpropagation of values and tangents
        for (int s = 0; s < alpha_scalar_count; s++) {
            nbh_energy_ders_wrt_moments[alpha_moment_mapping[s]] = linear_coeffs[s];
            moment_adjoints[alpha_moment_mapping[s]] = a * linear_coeffs[s];
        }

        for (int k = alpha_index_times_count - 1; k >= 0; k--) {
            const int a0 = alpha_index_times[k][0], a1 = alpha_index_times[k][1];
            const int mul = alpha_index_times[k][2], a3 = alpha_index_times[k][3];
            const Lanes& G_a3 = nbh_energy_ders_wrt_moments[a3];
            const Lanes& A_a3 = moment_adjoints[a3];
            moment_adjoints[a0] += mul * A_a3 * moment_tensor_vals[a1];
            moment_adjoints[a0] += mul * G_a3 * moment_tangents[a1];
            moment_adjoints[a1] += mul * A_a3 * moment_tensor_vals[a0];
            moment_adjoints[a1] += mul * G_a3 * moment_tangents[a0];
            nbh_energy_ders_wrt_moments[a1] += mul * G_a3 * moment_tensor_vals[a0];
            nbh_energy_ders_wrt_moments[a0] += mul * G_a3 * moment_tensor_vals[a1];
        }

        // Linear coefficients, summed over the atoms of the chunk
        for (int s = 0; s < alpha_scalar_count; s++)
            loss_grad[lin_off + s] += a * sum_over_atoms(moment_tensor_vals[alpha_moment_mapping[s]]) + sum_over_atoms(moment_tangents[alpha_moment_mapping[s]]);

        for (int ii = ii_begin; ii < ii_end; ii++) {
            const int itype = list.types[list.ilist[ii]];
            const int atom = ii - ii_begin;
            const int valid_count = cached_counts[atom];

            select_atom_cache(atom);
            const double* ang = neighbor_angulars.data() + (size_t) atom * angular_count * N;
            const double* tangents = neighbor_angular_tangents.data() + (size_t) atom * angular_count * N;
            const double* w_dot_unit_r = neighbor_unit_tangents.data() + (size_t) atom * 4 * N + 3 * N;
            const double* radial_vals = neighbor_radial_vals.data() + (size_t) atom * radial_basis_size * N;
            const double* radial_ders = neighbor_radial_ders.data() + (size_t) atom * radial_basis_size * N;

            loss_grad[sp_off + itype] += a;

            // Radial coefficients
            for (int t = 0; t < alpha_index_basic_count; t++) {
                basic_ders_by_mu[t] = nbh_energy_ders_wrt_moments[basic_by_mu[t]][atom];
                basic_adjoints_by_mu[t] = moment_adjoints[basic_by_mu[t]][atom];
            }

            for (int jj = 0; jj < valid_count; jj++) {
                const int jtype = list.types[cached_j[jj]];
                const double w_r = w_dot_unit_r[jj];
                const double* basis_vals = radial_vals + (size_t) jj * radial_basis_size;
                const double* basis_ders = radial_ders + (size_t) jj * radial_basis_size;
                double* row = loss_grad + (size_t) (itype * species_count + jtype) * radial_coeff_count_per_pair;
                for (int mu = 0; mu < radial_func_count; mu++) {
                    double value_adjoint = 0.0, der_adjoint = 0.0;
                    for (int t = mu_offsets[mu]; t < mu_offsets[mu + 1]; t++) {
                        const double angular = ang[(size_t) angular_by_mu[t] * N + jj];
                        value_adjoint += basic_adjoints_by_mu[t] * angular + basic_ders_by_mu[t] * tangents[(size_t) angular_by_mu[t] * N + jj];
                        der_adjoint += basic_ders_by_mu[t] * angular;
                    }
                    der_adjoint *= w_r;
                    for (int ri = 0; ri < radial_basis_size; ri++)
                        row[mu * radial_basis_size + ri] += value_adjoint * basis_vals[ri] + der_adjoint * basis_ders[ri];
                }
            }
        }
    }
}
