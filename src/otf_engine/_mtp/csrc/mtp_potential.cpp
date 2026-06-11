/* -*- c++ -*- ----------------------------------------------------------
   Standalone MTP potential — no LAMMPS dependency.
   Refactored from lammps-mtp/src/ML-MTP/pair_mtp.cpp
   Original author: Richard Meng, Queen's University at Kingston, 22.11.24
------------------------------------------------------------------------- */

#include "mtp_potential.h"

#include "file_utils.h"
#include "mtp_radial_basis.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

// ===========================================================================
// Constructor / Destructor
// ===========================================================================

MTPPotential::MTPPotential(const std::string& filename) {
    ::read_file(filename, *this);
}

MTPPotential::~MTPPotential() {
    delete radial_basis;
}

// ===========================================================================
// Internal neighborhood forward-pass
//
// Fills moment_tensor_vals[0..alpha_moment_count) for central atom i.
// Also fills moment_jacobian[0..valid_count)[0..alpha_index_basic_count)[0..3)
// and valid_j[0..valid_count).
//
// displacements: flat array [jnum * 3] — r_j - r_i (PBC-corrected), one entry
//               per neighbor in the same order as neighbors[].
//
// Returns the number of valid (within-cutoff) neighbors.
// ===========================================================================
// ang_factor_out: if non-null, filled with [valid_count * alpha_index_basic_count] values
//   ang_factor_out[jj * alpha_index_basic_count + k] = pow_k(r_ij) / r_ij^rank_k
//   (the angular scalar factor for basic moment k at valid neighbor jj)
static int neighborhood_forward(int i, const int* types, int jnum, const int* neighbors, const double* displacements, int species_count, int radial_func_count, int radial_basis_size, int radial_coeff_count_per_pair, double max_cutoff_sq, int max_alpha_index_basic, int alpha_index_basic_count, int alpha_index_times_count, const double* radial_basis_coeffs, const std::vector<std::array<int, 4>>& alpha_index_basic, const std::vector<std::array<int, 4>>& alpha_index_times, RadialMTPBasis* radial_basis, std::vector<double>& dist_powers, std::vector<std::array<double, 3>>& coord_powers, std::vector<double>& radial_vals, std::vector<double>& radial_ders, std::vector<double>& moment_tensor_vals, std::vector<std::array<double, 3>>& moment_jacobian, std::vector<int>& valid_j, std::vector<std::array<double, 3>>& valid_dr, int& jac_cap, std::vector<double>* ang_factor_out = nullptr, std::vector<double>* angfac_jac_out = nullptr) {
    const int itype = types[i];

    std::fill(moment_tensor_vals.begin(), moment_tensor_vals.end(), 0.0);

    int valid_count = 0;

    // ------------ Calculate Basic Moments ------------
    for (int jj = 0; jj < jnum; jj++) {
        const int j = neighbors[jj];
        const int jtype = types[j];

        // Use pre-computed PBC-corrected displacement (avoids wrong minimum-image)
        const double r0 = displacements[jj * 3 + 0];
        const double r1 = displacements[jj * 3 + 1];
        const double r2 = displacements[jj * 3 + 2];
        const double rsq = r0 * r0 + r1 * r1 + r2 * r2;

        if (rsq > max_cutoff_sq)
            continue;

        // Grow buffers if needed
        if (valid_count >= jac_cap) {
            jac_cap = valid_count + 32;
            moment_jacobian.resize(jac_cap * alpha_index_basic_count);
            if (ang_factor_out)
                ang_factor_out->resize(jac_cap * alpha_index_basic_count);
            if (angfac_jac_out)
                angfac_jac_out->resize(jac_cap * alpha_index_basic_count * 3);
            valid_j.resize(jac_cap);
            valid_dr.resize(jac_cap);
        }
        valid_j[valid_count] = j;
        valid_dr[valid_count] = {r0, r1, r2};

        const double dist = std::sqrt(rsq);
        radial_basis->calc_radial_basis_ders(dist);

        // Precompute distance and coordinate powers
        for (int k = 1; k < max_alpha_index_basic; k++) {
            dist_powers[k] = dist_powers[k - 1] * dist;
            coord_powers[k][0] = coord_powers[k - 1][0] * r0;
            coord_powers[k][1] = coord_powers[k - 1][1] * r1;
            coord_powers[k][2] = coord_powers[k - 1][2] * r2;
        }

        // Radial basis contraction: R_mu(dist) = sum_ri  c[pair,mu,ri] * phi[ri](dist)
        const int pair_offset = itype * species_count + jtype;
        for (int mu = 0; mu < radial_func_count; mu++) {
            double val = 0.0, der = 0.0;
            const int offset = pair_offset * radial_coeff_count_per_pair + mu * radial_basis_size;
            for (int ri = 0; ri < radial_basis_size; ri++) {
                val += radial_basis_coeffs[offset + ri] * radial_basis->radial_basis_vals[ri];
                der += radial_basis_coeffs[offset + ri] * radial_basis->radial_basis_ders[ri];
            }
            radial_vals[mu] = val;
            radial_ders[mu] = der;
        }

        // Accumulate into basic moment elements
        for (int k = 0; k < alpha_index_basic_count; k++) {
            const int mu = alpha_index_basic[k][0];
            const int px = alpha_index_basic[k][1];
            const int py = alpha_index_basic[k][2];
            const int pz = alpha_index_basic[k][3];
            const int norm_rank = px + py + pz;

            double val = radial_vals[mu];
            double der = radial_ders[mu];

            // Normalise by dist^rank
            const double norm_fac = 1.0 / dist_powers[norm_rank];
            val *= norm_fac;
            der = der * norm_fac - norm_rank * val / dist;

            const double pow0 = coord_powers[px][0];
            const double pow1 = coord_powers[py][1];
            const double pow2 = coord_powers[pz][2];
            const double pow = pow0 * pow1 * pow2;
            moment_tensor_vals[k] += val * pow;

            // Angular scalar factor = pow / r^rank  (used for radial gradient)
            const double angfac = pow * norm_fac;
            if (ang_factor_out)
                (*ang_factor_out)[valid_count * alpha_index_basic_count + k] = angfac;

            // Jacobian of angfac w.r.t. displacement r_{jj}: ∂angfac/∂r_d
            // = norm_fac*(px*x^(px-1)*δ_{d,0}*y^py*z^pz + ...) - norm_rank*angfac*r_d/rsq
            if (angfac_jac_out) {
                double* jac = angfac_jac_out->data() + (valid_count * alpha_index_basic_count + k) * 3;
                jac[0] = jac[1] = jac[2] = 0.0;
                if (px != 0) jac[0] += norm_fac * px * coord_powers[px - 1][0] * pow1 * pow2;
                if (py != 0) jac[1] += norm_fac * py * pow0 * coord_powers[py - 1][1] * pow2;
                if (pz != 0) jac[2] += norm_fac * pz * pow0 * pow1 * coord_powers[pz - 1][2];
                const double rank_fac = norm_rank * angfac / rsq;
                jac[0] -= rank_fac * r0;
                jac[1] -= rank_fac * r1;
                jac[2] -= rank_fac * r2;
            }

            // Jacobian of basic moment k w.r.t. position of neighbor jj
            const int jac_base = valid_count * alpha_index_basic_count + k;
            const double dpow_dr = pow * der / dist;
            moment_jacobian[jac_base] = {dpow_dr * r0, dpow_dr * r1, dpow_dr * r2};
            if (px != 0)
                moment_jacobian[jac_base][0] += val * px * coord_powers[px - 1][0] * pow1 * pow2;
            if (py != 0)
                moment_jacobian[jac_base][1] += val * py * pow0 * coord_powers[py - 1][1] * pow2;
            if (pz != 0)
                moment_jacobian[jac_base][2] += val * pz * pow0 * pow1 * coord_powers[pz - 1][2];
        }
        valid_count++;
    }

    // ------------ Construct Composite Moment Values ------------
    for (int k = 0; k < alpha_index_times_count; k++) {
        const int i0 = alpha_index_times[k][0];
        const int i1 = alpha_index_times[k][1];
        const int mul = alpha_index_times[k][2];
        const int i3 = alpha_index_times[k][3];
        moment_tensor_vals[i3] += mul * moment_tensor_vals[i0] * moment_tensor_vals[i1];
    }

    return valid_count;
}

// ===========================================================================
// compute()
// ===========================================================================

double MTPPotential::compute(int n_atoms, const int* types, int inum, const int* ilist, const int* numneigh, const int* firstneigh, const double* displacements, double* forces_out, double* virials_out, double* eatom_out) {
    double total_energy = 0.0;

    int nbr_offset = 0; // running index into flat firstneigh / displacements

    for (int ii = 0; ii < inum; ii++) {
        const int i = ilist[ii];
        const int jnum = numneigh[ii];
        const int* nbrs = firstneigh + nbr_offset;
        const double* dr = displacements + nbr_offset * 3; // PBC-corrected r_j - r_i
        nbr_offset += jnum;

        const int valid_count = neighborhood_forward(i, types, jnum, nbrs, dr, species_count, radial_func_count, radial_basis_size, radial_coeff_count_per_pair, max_cutoff_sq, max_alpha_index_basic, alpha_index_basic_count, alpha_index_times_count, radial_basis_coeffs.data(), alpha_index_basic, alpha_index_times, radial_basis, dist_powers, coord_powers, radial_vals, radial_ders, moment_tensor_vals, moment_jacobian, valid_j, valid_dr, jac_cap);

        // ------------ Energy from scalar moment linear combination ------------
        const int itype = types[i];
        double nbh_energy = species_coeffs[itype];
        for (int k = 0; k < alpha_scalar_count; k++)
            nbh_energy += linear_coeffs[k] * moment_tensor_vals[alpha_moment_mapping[k]];

        total_energy += nbh_energy;
        if (eatom_out)
            eatom_out[i] = nbh_energy;

        // =========== Backpropagation ===========

        // NBH energy derivative w.r.t. each moment = corresponding linear coeff
        std::fill(nbh_energy_ders_wrt_moments.begin(), nbh_energy_ders_wrt_moments.end(), 0.0);
        for (int k = 0; k < alpha_scalar_count; k++)
            nbh_energy_ders_wrt_moments[alpha_moment_mapping[k]] = linear_coeffs[k];

        // Chain rule through composite moments (in reverse order)
        for (int k = alpha_index_times_count - 1; k >= 0; k--) {
            const int a0 = alpha_index_times[k][0];
            const int a1 = alpha_index_times[k][1];
            const int mul = alpha_index_times[k][2];
            const int a3 = alpha_index_times[k][3];
            const double d3 = nbh_energy_ders_wrt_moments[a3];
            nbh_energy_ders_wrt_moments[a0] += d3 * mul * moment_tensor_vals[a1];
            nbh_energy_ders_wrt_moments[a1] += d3 * mul * moment_tensor_vals[a0];
        }

        // dE/dr = sum_k  (dE/dM_k) * (dM_k/dr_j)
        for (int jj = 0; jj < valid_count; jj++) {
            const int j = valid_j[jj];

            double tf0 = 0.0, tf1 = 0.0, tf2 = 0.0;
            for (int k = 0; k < alpha_index_basic_count; k++) {
                const double d = nbh_energy_ders_wrt_moments[k];
                const auto& jac = moment_jacobian[jj * alpha_index_basic_count + k];
                tf0 += d * jac[0];
                tf1 += d * jac[1];
                tf2 += d * jac[2];
            }

            forces_out[i * 3 + 0] += tf0;
            forces_out[i * 3 + 1] += tf1;
            forces_out[i * 3 + 2] += tf2;
            forces_out[j * 3 + 0] -= tf0;
            forces_out[j * 3 + 1] -= tf1;
            forces_out[j * 3 + 2] -= tf2;

            if (virials_out) {
                // Use the PBC-corrected displacement stored during the forward pass
                const auto& r = valid_dr[jj];
                virials_out[0] -= tf0 * r[0];                      // xx
                virials_out[1] -= tf1 * r[1];                      // yy
                virials_out[2] -= tf2 * r[2];                      // zz
                virials_out[3] -= (tf0 * r[1] + tf1 * r[0]) / 2.0; // xy
                virials_out[4] -= (tf0 * r[2] + tf2 * r[0]) / 2.0; // xz
                virials_out[5] -= (tf1 * r[2] + tf2 * r[1]) / 2.0; // yz
            }
        }
    }

    return total_energy;
}

// ===========================================================================
// eval_basis()  — moment tensor values per atom, before dotting with coeffs
// ===========================================================================

void MTPPotential::eval_basis(int n_atoms, const int* types, int inum, const int* ilist, const int* numneigh, const int* firstneigh, const double* displacements, double* basis_out) {
    int nbr_offset = 0;

    for (int ii = 0; ii < inum; ii++) {
        const int i = ilist[ii];
        const int jnum = numneigh[ii];
        const int* nbrs = firstneigh + nbr_offset;
        const double* dr = displacements + nbr_offset * 3;
        nbr_offset += jnum;

        neighborhood_forward(i, types, jnum, nbrs, dr, species_count, radial_func_count, radial_basis_size, radial_coeff_count_per_pair, max_cutoff_sq, max_alpha_index_basic, alpha_index_basic_count, alpha_index_times_count, radial_basis_coeffs.data(), alpha_index_basic, alpha_index_times, radial_basis, dist_powers, coord_powers, radial_vals, radial_ders, moment_tensor_vals, moment_jacobian, valid_j, valid_dr, jac_cap);

        // Copy the alpha_scalar_count selected moment values into output
        double* row = basis_out + ii * alpha_scalar_count;
        for (int k = 0; k < alpha_scalar_count; k++)
            row[k] = moment_tensor_vals[alpha_moment_mapping[k]];
    }
}

// ===========================================================================
// eval_grad()  — full per-atom information vector for grade computation
// ===========================================================================

void MTPPotential::eval_grad(int n_atoms, const int* types, int inum, const int* ilist, const int* numneigh, const int* firstneigh, const double* displacements, double* grad_out) {
    const int cc = coeff_count();
    std::fill(grad_out, grad_out + (long)inum * cc, 0.0);

    int nbr_offset = 0;

    for (int ii = 0; ii < inum; ii++) {
        const int i = ilist[ii];
        const int jnum = numneigh[ii];
        const int* nbrs = firstneigh + nbr_offset;
        const double* dr = displacements + nbr_offset * 3;
        nbr_offset += jnum;

        // Resize angular factor buffer if needed
        if (jac_cap * alpha_index_basic_count > (int)nbh_angular_factor.size())
            nbh_angular_factor.resize((jac_cap + 32) * alpha_index_basic_count);

        const int valid_count = neighborhood_forward(i, types, jnum, nbrs, dr, species_count, radial_func_count, radial_basis_size, radial_coeff_count_per_pair, max_cutoff_sq, max_alpha_index_basic, alpha_index_basic_count, alpha_index_times_count, radial_basis_coeffs.data(), alpha_index_basic, alpha_index_times, radial_basis, dist_powers, coord_powers, radial_vals, radial_ders, moment_tensor_vals, moment_jacobian, valid_j, valid_dr, jac_cap, &nbh_angular_factor);

        const int itype = types[i];
        double* row = grad_out + ii * cc;

        // --- Linear basis values ---
        const int lin_off = radial_coeff_count + species_count;
        for (int k = 0; k < alpha_scalar_count; k++)
            row[lin_off + k] = moment_tensor_vals[alpha_moment_mapping[k]];

        // --- Species one-hot ---
        row[radial_coeff_count + itype] = 1.0;

        // --- Radial gradient ---
        // Backward pass: fill nbh_energy_ders_wrt_moments
        std::fill(nbh_energy_ders_wrt_moments.begin(), nbh_energy_ders_wrt_moments.end(), 0.0);
        for (int k = 0; k < alpha_scalar_count; k++)
            nbh_energy_ders_wrt_moments[alpha_moment_mapping[k]] = linear_coeffs[k];
        for (int k = alpha_index_times_count - 1; k >= 0; k--) {
            const int a0 = alpha_index_times[k][0];
            const int a1 = alpha_index_times[k][1];
            const int mul = alpha_index_times[k][2];
            const int a3 = alpha_index_times[k][3];
            const double d3 = nbh_energy_ders_wrt_moments[a3];
            nbh_energy_ders_wrt_moments[a0] += d3 * mul * moment_tensor_vals[a1];
            nbh_energy_ders_wrt_moments[a1] += d3 * mul * moment_tensor_vals[a0];
        }

        // Accumulate radial coefficient gradients over valid neighbors
        for (int jj = 0; jj < valid_count; jj++) {
            const int j = valid_j[jj];
            const int jtype = types[j];

            // Re-evaluate raw radial basis (phi_ri) at this neighbor's distance
            const auto& rr = valid_dr[jj];
            const double dist = std::sqrt(rr[0]*rr[0] + rr[1]*rr[1] + rr[2]*rr[2]);
            radial_basis->calc_radial_basis(dist);

            const int pair_off = itype * species_count + jtype;

            for (int k = 0; k < alpha_index_basic_count; k++) {
                const double dE_dMk = nbh_energy_ders_wrt_moments[k];
                if (dE_dMk == 0.0) continue;
                const int mu = alpha_index_basic[k][0];
                const double angfac = nbh_angular_factor[jj * alpha_index_basic_count + k];
                const double scale = dE_dMk * angfac;
                const int coeff_off = (pair_off * radial_func_count + mu) * radial_basis_size;
                for (int ri = 0; ri < radial_basis_size; ri++)
                    row[coeff_off + ri] += scale * radial_basis->radial_basis_vals[ri];
            }
        }
    }
}

// ===========================================================================
// compute_with_radial_grad()
//
// Fused compute + eval_grad (energy, radial part) + eval_force_radial_grad.
// One neighborhood_forward per atom instead of three separate calls.
// ===========================================================================

double MTPPotential::compute_with_radial_grad(int n_atoms, const int* types, int inum, const int* ilist, const int* numneigh, const int* firstneigh, const double* displacements, double* forces_out, double* virials_out, double* energy_grad_out, double* force_grad_out, double* virial_grad_out) {
    const int n_radial = radial_coeff_count;
    const long fgs = (long)n_atoms * 3 * n_radial;

    double total_energy = 0.0;
    std::fill(forces_out, forces_out + n_atoms * 3, 0.0);
    if (virials_out) std::fill(virials_out, virials_out + 6, 0.0);
    std::fill(energy_grad_out, energy_grad_out + n_radial, 0.0);
    std::fill(force_grad_out, force_grad_out + fgs, 0.0);
    if (virial_grad_out) std::fill(virial_grad_out, virial_grad_out + 6 * n_radial, 0.0);

    const int buf_size = alpha_moment_count * n_radial;
    if ((int)dM_dc_buf.size() < buf_size) dM_dc_buf.resize(buf_size);
    if ((int)dG_buf.size()    < buf_size) dG_buf.resize(buf_size);

    int nbr_offset = 0;

    for (int ii = 0; ii < inum; ii++) {
        const int i = ilist[ii];
        const int jnum = numneigh[ii];
        const int* nbrs = firstneigh + nbr_offset;
        const double* dr = displacements + nbr_offset * 3;
        nbr_offset += jnum;

        if ((int)nbh_angular_factor.size() < (jac_cap + 32) * alpha_index_basic_count)
            nbh_angular_factor.resize((jac_cap + 32) * alpha_index_basic_count);
        if ((int)nbh_angular_factor_jac.size() < (jac_cap + 32) * alpha_index_basic_count * 3)
            nbh_angular_factor_jac.resize((jac_cap + 32) * alpha_index_basic_count * 3);

        const int valid_count = neighborhood_forward(
            i, types, jnum, nbrs, dr,
            species_count, radial_func_count, radial_basis_size, radial_coeff_count_per_pair,
            max_cutoff_sq, max_alpha_index_basic, alpha_index_basic_count, alpha_index_times_count,
            radial_basis_coeffs.data(), alpha_index_basic, alpha_index_times, radial_basis,
            dist_powers, coord_powers, radial_vals, radial_ders, moment_tensor_vals,
            moment_jacobian, valid_j, valid_dr, jac_cap,
            &nbh_angular_factor, &nbh_angular_factor_jac);

        const int itype = types[i];

        // --- Backward pass: G_k = dE/dM_k ---
        std::fill(nbh_energy_ders_wrt_moments.begin(), nbh_energy_ders_wrt_moments.end(), 0.0);
        for (int k = 0; k < alpha_scalar_count; k++)
            nbh_energy_ders_wrt_moments[alpha_moment_mapping[k]] = linear_coeffs[k];
        for (int k = alpha_index_times_count - 1; k >= 0; k--) {
            const int a0 = alpha_index_times[k][0], a1 = alpha_index_times[k][1];
            const int mul = alpha_index_times[k][2], a3 = alpha_index_times[k][3];
            const double d3 = nbh_energy_ders_wrt_moments[a3];
            nbh_energy_ders_wrt_moments[a0] += d3 * mul * moment_tensor_vals[a1];
            nbh_energy_ders_wrt_moments[a1] += d3 * mul * moment_tensor_vals[a0];
        }

        // --- Energy ---
        double nbh_energy = species_coeffs[itype];
        for (int k = 0; k < alpha_scalar_count; k++)
            nbh_energy += linear_coeffs[k] * moment_tensor_vals[alpha_moment_mapping[k]];
        total_energy += nbh_energy;

        // --- Pass 1: neighbors → dM_dc + energy_grad + forces + virials ---
        std::fill(dM_dc_buf.begin(), dM_dc_buf.begin() + buf_size, 0.0);

        for (int jj = 0; jj < valid_count; jj++) {
            const int j = valid_j[jj];
            const int jtype = types[j];
            const auto& rr = valid_dr[jj];
            const double rsq = rr[0]*rr[0] + rr[1]*rr[1] + rr[2]*rr[2];
            const double dist = std::sqrt(rsq);

            radial_basis->calc_radial_basis_ders(dist);   // vals + ders in one call

            const int pair_jj = itype * species_count + jtype;
            const int coeff_off_pair = pair_jj * radial_coeff_count_per_pair;

            // Forces
            double tf0 = 0.0, tf1 = 0.0, tf2 = 0.0;

            for (int k = 0; k < alpha_index_basic_count; k++) {
                const double G_k    = nbh_energy_ders_wrt_moments[k];
                const int mu_k      = alpha_index_basic[k][0];
                const double angfac = nbh_angular_factor[jj * alpha_index_basic_count + k];
                const int coeff_off = coeff_off_pair + mu_k * radial_basis_size;

                // Accumulate dM_dc and energy_grad from this neighbor/moment/ri
                double* dM_row = dM_dc_buf.data() + (long)k * n_radial + coeff_off;
                for (int ri = 0; ri < radial_basis_size; ri++) {
                    const double phi = radial_basis->radial_basis_vals[ri];
                    dM_row[ri]               += angfac * phi;
                    energy_grad_out[coeff_off + ri] += G_k * angfac * phi;
                }

                // Force accumulation via moment Jacobian
                const auto& jac = moment_jacobian[jj * alpha_index_basic_count + k];
                tf0 += G_k * jac[0];
                tf1 += G_k * jac[1];
                tf2 += G_k * jac[2];
            }

            forces_out[i * 3 + 0] += tf0;
            forces_out[i * 3 + 1] += tf1;
            forces_out[i * 3 + 2] += tf2;
            forces_out[j * 3 + 0] -= tf0;
            forces_out[j * 3 + 1] -= tf1;
            forces_out[j * 3 + 2] -= tf2;

            if (virials_out) {
                virials_out[0] -= tf0 * rr[0];
                virials_out[1] -= tf1 * rr[1];
                virials_out[2] -= tf2 * rr[2];
                virials_out[3] -= (tf0 * rr[1] + tf1 * rr[0]) / 2.0;
                virials_out[4] -= (tf0 * rr[2] + tf2 * rr[0]) / 2.0;
                virials_out[5] -= (tf1 * rr[2] + tf2 * rr[1]) / 2.0;
            }
        }

        // --- Forward propagate dM through composites ---
        for (int k = 0; k < alpha_index_times_count; k++) {
            const int i0 = alpha_index_times[k][0], i1 = alpha_index_times[k][1];
            const int mul = alpha_index_times[k][2], i3 = alpha_index_times[k][3];
            const double M_i0 = moment_tensor_vals[i0], M_i1 = moment_tensor_vals[i1];
            const double* dM_i0 = dM_dc_buf.data() + (long)i0 * n_radial;
            const double* dM_i1 = dM_dc_buf.data() + (long)i1 * n_radial;
            double* dM_i3       = dM_dc_buf.data() + (long)i3 * n_radial;
            for (int r = 0; r < n_radial; r++)
                dM_i3[r] += mul * (dM_i0[r] * M_i1 + M_i0 * dM_i1[r]);
        }

        // --- dG backward pass ---
        std::fill(dG_buf.begin(), dG_buf.begin() + buf_size, 0.0);
        for (int k = alpha_index_times_count - 1; k >= 0; k--) {
            const int a0 = alpha_index_times[k][0], a1 = alpha_index_times[k][1];
            const int mul = alpha_index_times[k][2], a3 = alpha_index_times[k][3];
            const double G_a3  = nbh_energy_ders_wrt_moments[a3];
            const double M_a0  = moment_tensor_vals[a0], M_a1 = moment_tensor_vals[a1];
            const double imul  = (double)mul;
            const double* dM_a0 = dM_dc_buf.data() + (long)a0 * n_radial;
            const double* dM_a1 = dM_dc_buf.data() + (long)a1 * n_radial;
            const double* dG_a3 = dG_buf.data()    + (long)a3 * n_radial;
            double* dG_a0 = dG_buf.data() + (long)a0 * n_radial;
            double* dG_a1 = dG_buf.data() + (long)a1 * n_radial;
            for (int r = 0; r < n_radial; r++) {
                dG_a0[r] += G_a3 * imul * dM_a1[r] + dG_a3[r] * imul * M_a1;
                dG_a1[r] += G_a3 * imul * dM_a0[r] + dG_a3[r] * imul * M_a0;
            }
        }

        // --- Pass 2a: term 2 force grad (pair-local r range) ---
        for (int jj = 0; jj < valid_count; jj++) {
            const int j = valid_j[jj];
            const int jtype = types[j];
            const auto& rr = valid_dr[jj];
            const double dist = std::sqrt(rr[0]*rr[0] + rr[1]*rr[1] + rr[2]*rr[2]);
            radial_basis->calc_radial_basis_ders(dist);
            const double rd_over_dist[3] = {rr[0]/dist, rr[1]/dist, rr[2]/dist};
            const int pair_off = itype * species_count + jtype;

            for (int k = 0; k < alpha_index_basic_count; k++) {
                const double G_k = nbh_energy_ders_wrt_moments[k];
                if (G_k == 0.0) continue;
                const int mu     = alpha_index_basic[k][0];
                const double af  = nbh_angular_factor[jj * alpha_index_basic_count + k];
                const double* afj = nbh_angular_factor_jac.data() + (jj * alpha_index_basic_count + k) * 3;
                const int coeff_off = (pair_off * radial_func_count + mu) * radial_basis_size;

                for (int d = 0; d < 3; d++) {
                    const double afd = afj[d], rdd = rd_over_dist[d];
                    for (int ri = 0; ri < radial_basis_size; ri++) {
                        const double phi = radial_basis->radial_basis_vals[ri];
                        const double dph = radial_basis->radial_basis_ders[ri];
                        const int r_idx  = coeff_off + ri;
                        const double v2  = G_k * (dph * rdd * af + phi * afd);
                        const long ii_   = ((long)i * 3 + d) * n_radial + r_idx;
                        const long jj_   = ((long)j * 3 + d) * n_radial + r_idx;
                        force_grad_out[ii_] += v2;
                        force_grad_out[jj_] -= v2;
                        if (virial_grad_out) {
                            virial_grad_out[d * n_radial + r_idx] -= v2 * rr[d];
                            if (d == 0) {
                                virial_grad_out[3 * n_radial + r_idx] -= v2 * rr[1] / 2.0;
                                virial_grad_out[4 * n_radial + r_idx] -= v2 * rr[2] / 2.0;
                            } else if (d == 1) {
                                virial_grad_out[3 * n_radial + r_idx] -= v2 * rr[0] / 2.0;
                                virial_grad_out[5 * n_radial + r_idx] -= v2 * rr[2] / 2.0;
                            } else {
                                virial_grad_out[4 * n_radial + r_idx] -= v2 * rr[0] / 2.0;
                                virial_grad_out[5 * n_radial + r_idx] -= v2 * rr[1] / 2.0;
                            }
                        }
                    }
                }
            }
        }

        // --- Pass 2b: term 1 force grad (all r) ---
        for (int jj = 0; jj < valid_count; jj++) {
            const int j = valid_j[jj];
            const auto& rr = valid_dr[jj];
            for (int k = 0; k < alpha_index_basic_count; k++) {
                const double* dG_k = dG_buf.data() + (long)k * n_radial;
                const auto& jac_k  = moment_jacobian[jj * alpha_index_basic_count + k];
                for (int d = 0; d < 3; d++) {
                    const double jkd = jac_k[d];
                    if (jkd == 0.0) continue;
                    const long ib = ((long)i * 3 + d) * n_radial;
                    const long jb = ((long)j * 3 + d) * n_radial;
                    for (int r = 0; r < n_radial; r++) {
                        const double v1 = dG_k[r] * jkd;
                        force_grad_out[ib + r] += v1;
                        force_grad_out[jb + r] -= v1;
                    }
                    if (virial_grad_out) {
                        for (int r = 0; r < n_radial; r++) {
                            const double v1 = dG_k[r] * jkd;
                            virial_grad_out[d * n_radial + r] -= v1 * rr[d];
                            if (d == 0) {
                                virial_grad_out[3 * n_radial + r] -= v1 * rr[1] / 2.0;
                                virial_grad_out[4 * n_radial + r] -= v1 * rr[2] / 2.0;
                            } else if (d == 1) {
                                virial_grad_out[3 * n_radial + r] -= v1 * rr[0] / 2.0;
                                virial_grad_out[5 * n_radial + r] -= v1 * rr[2] / 2.0;
                            } else {
                                virial_grad_out[4 * n_radial + r] -= v1 * rr[0] / 2.0;
                                virial_grad_out[5 * n_radial + r] -= v1 * rr[1] / 2.0;
                            }
                        }
                    }
                }
            }
        }
    }
    return total_energy;
}

// ===========================================================================
// eval_grad_radial() — ∂(E_total, F_i, virial) / ∂c_radial
//
// Same algorithm as the gradient part of compute_with_radial_grad but without
// computing the actual energy/force/virial values.
// ===========================================================================

void MTPPotential::eval_grad_radial(int n_atoms, const int* types, int inum, const int* ilist, const int* numneigh, const int* firstneigh, const double* displacements, double* energy_grad_out, double* force_grad_out, double* virial_grad_out) {
    const int n_radial = radial_coeff_count;

    std::fill(energy_grad_out, energy_grad_out + n_radial, 0.0);
    std::fill(force_grad_out,  force_grad_out  + (long)n_atoms * 3 * n_radial, 0.0);
    if (virial_grad_out) std::fill(virial_grad_out, virial_grad_out + 6 * n_radial, 0.0);

    const int buf_size = alpha_moment_count * n_radial;
    if ((int)dM_dc_buf.size() < buf_size) dM_dc_buf.resize(buf_size);
    if ((int)dG_buf.size()    < buf_size) dG_buf.resize(buf_size);

    int nbr_offset = 0;

    for (int ii = 0; ii < inum; ii++) {
        const int i = ilist[ii];
        const int jnum = numneigh[ii];
        const int* nbrs = firstneigh + nbr_offset;
        const double* dr = displacements + nbr_offset * 3;
        nbr_offset += jnum;

        if ((int)nbh_angular_factor.size() < (jac_cap + 32) * alpha_index_basic_count)
            nbh_angular_factor.resize((jac_cap + 32) * alpha_index_basic_count);
        if ((int)nbh_angular_factor_jac.size() < (jac_cap + 32) * alpha_index_basic_count * 3)
            nbh_angular_factor_jac.resize((jac_cap + 32) * alpha_index_basic_count * 3);

        const int valid_count = neighborhood_forward(
            i, types, jnum, nbrs, dr,
            species_count, radial_func_count, radial_basis_size, radial_coeff_count_per_pair,
            max_cutoff_sq, max_alpha_index_basic, alpha_index_basic_count, alpha_index_times_count,
            radial_basis_coeffs.data(), alpha_index_basic, alpha_index_times, radial_basis,
            dist_powers, coord_powers, radial_vals, radial_ders, moment_tensor_vals,
            moment_jacobian, valid_j, valid_dr, jac_cap,
            &nbh_angular_factor, &nbh_angular_factor_jac);

        const int itype = types[i];

        // Backward pass
        std::fill(nbh_energy_ders_wrt_moments.begin(), nbh_energy_ders_wrt_moments.end(), 0.0);
        for (int k = 0; k < alpha_scalar_count; k++)
            nbh_energy_ders_wrt_moments[alpha_moment_mapping[k]] = linear_coeffs[k];
        for (int k = alpha_index_times_count - 1; k >= 0; k--) {
            const int a0 = alpha_index_times[k][0], a1 = alpha_index_times[k][1];
            const int mul = alpha_index_times[k][2], a3 = alpha_index_times[k][3];
            const double d3 = nbh_energy_ders_wrt_moments[a3];
            nbh_energy_ders_wrt_moments[a0] += d3 * mul * moment_tensor_vals[a1];
            nbh_energy_ders_wrt_moments[a1] += d3 * mul * moment_tensor_vals[a0];
        }

        // Pass 1: dM_dc + energy_grad_radial
        std::fill(dM_dc_buf.begin(), dM_dc_buf.begin() + buf_size, 0.0);
        for (int jj = 0; jj < valid_count; jj++) {
            const int jtype = types[valid_j[jj]];
            const auto& rr_jj = valid_dr[jj];
            const double dist_jj = std::sqrt(rr_jj[0]*rr_jj[0] + rr_jj[1]*rr_jj[1] + rr_jj[2]*rr_jj[2]);
            radial_basis->calc_radial_basis(dist_jj);
            const int pair_jj = itype * species_count + jtype;
            const int coeff_off_pair = pair_jj * radial_coeff_count_per_pair;
            for (int k = 0; k < alpha_index_basic_count; k++) {
                const double G_k = nbh_energy_ders_wrt_moments[k];
                const int mu_k = alpha_index_basic[k][0];
                const double af = nbh_angular_factor[jj * alpha_index_basic_count + k];
                const int coeff_off = coeff_off_pair + mu_k * radial_basis_size;
                double* dM_row = dM_dc_buf.data() + (long)k * n_radial + coeff_off;
                for (int ri = 0; ri < radial_basis_size; ri++) {
                    const double phi = radial_basis->radial_basis_vals[ri];
                    dM_row[ri]                    += af * phi;
                    energy_grad_out[coeff_off + ri] += G_k * af * phi;
                }
            }
        }
        // Forward composite propagation of dM_dc
        for (int k = 0; k < alpha_index_times_count; k++) {
            const int i0 = alpha_index_times[k][0], i1 = alpha_index_times[k][1];
            const int mul = alpha_index_times[k][2], i3 = alpha_index_times[k][3];
            const double M_i0 = moment_tensor_vals[i0], M_i1 = moment_tensor_vals[i1];
            const double* dM_i0 = dM_dc_buf.data() + (long)i0 * n_radial;
            const double* dM_i1 = dM_dc_buf.data() + (long)i1 * n_radial;
            double* dM_i3       = dM_dc_buf.data() + (long)i3 * n_radial;
            for (int r = 0; r < n_radial; r++)
                dM_i3[r] += mul * (dM_i0[r] * M_i1 + M_i0 * dM_i1[r]);
        }
        // dG backward pass
        std::fill(dG_buf.begin(), dG_buf.begin() + buf_size, 0.0);
        for (int k = alpha_index_times_count - 1; k >= 0; k--) {
            const int a0 = alpha_index_times[k][0], a1 = alpha_index_times[k][1];
            const int mul = alpha_index_times[k][2], a3 = alpha_index_times[k][3];
            const double G_a3 = nbh_energy_ders_wrt_moments[a3];
            const double M_a0 = moment_tensor_vals[a0], M_a1 = moment_tensor_vals[a1];
            const double imul = (double)mul;
            const double* dM_a0 = dM_dc_buf.data() + (long)a0 * n_radial;
            const double* dM_a1 = dM_dc_buf.data() + (long)a1 * n_radial;
            const double* dG_a3 = dG_buf.data()    + (long)a3 * n_radial;
            double* dG_a0 = dG_buf.data() + (long)a0 * n_radial;
            double* dG_a1 = dG_buf.data() + (long)a1 * n_radial;
            for (int r = 0; r < n_radial; r++) {
                dG_a0[r] += G_a3 * imul * dM_a1[r] + dG_a3[r] * imul * M_a1;
                dG_a1[r] += G_a3 * imul * dM_a0[r] + dG_a3[r] * imul * M_a0;
            }
        }
        // Pass 2a: term 2 force/virial grad
        for (int jj = 0; jj < valid_count; jj++) {
            const int j = valid_j[jj]; const int jtype = types[j];
            const auto& rr = valid_dr[jj];
            const double dist = std::sqrt(rr[0]*rr[0]+rr[1]*rr[1]+rr[2]*rr[2]);
            radial_basis->calc_radial_basis_ders(dist);
            const double rdd[3] = {rr[0]/dist, rr[1]/dist, rr[2]/dist};
            const int pair_off = itype * species_count + jtype;
            for (int k = 0; k < alpha_index_basic_count; k++) {
                const double G_k = nbh_energy_ders_wrt_moments[k];
                if (G_k == 0.0) continue;
                const int mu = alpha_index_basic[k][0];
                const double af = nbh_angular_factor[jj * alpha_index_basic_count + k];
                const double* afj = nbh_angular_factor_jac.data() + (jj*alpha_index_basic_count+k)*3;
                const int co = (pair_off * radial_func_count + mu) * radial_basis_size;
                for (int d = 0; d < 3; d++) {
                    for (int ri = 0; ri < radial_basis_size; ri++) {
                        const double v2 = G_k * (radial_basis->radial_basis_ders[ri]*rdd[d]*af + radial_basis->radial_basis_vals[ri]*afj[d]);
                        const int r_ = co + ri;
                        force_grad_out[((long)i*3+d)*n_radial+r_] += v2;
                        force_grad_out[((long)j*3+d)*n_radial+r_] -= v2;
                        if (virial_grad_out) {
                            virial_grad_out[d*n_radial+r_] -= v2*rr[d];
                            if (d==0){virial_grad_out[3*n_radial+r_]-=v2*rr[1]/2;virial_grad_out[4*n_radial+r_]-=v2*rr[2]/2;}
                            else if(d==1){virial_grad_out[3*n_radial+r_]-=v2*rr[0]/2;virial_grad_out[5*n_radial+r_]-=v2*rr[2]/2;}
                            else{virial_grad_out[4*n_radial+r_]-=v2*rr[0]/2;virial_grad_out[5*n_radial+r_]-=v2*rr[1]/2;}
                        }
                    }
                }
            }
        }
        // Pass 2b: term 1 force/virial grad (all r)
        for (int jj = 0; jj < valid_count; jj++) {
            const int j = valid_j[jj];
            const auto& rr = valid_dr[jj];
            for (int k = 0; k < alpha_index_basic_count; k++) {
                const double* dG_k = dG_buf.data() + (long)k * n_radial;
                const auto& jac_k  = moment_jacobian[jj * alpha_index_basic_count + k];
                for (int d = 0; d < 3; d++) {
                    const double jkd = jac_k[d]; if (jkd == 0.0) continue;
                    const long ib = ((long)i*3+d)*n_radial, jb = ((long)j*3+d)*n_radial;
                    for (int r = 0; r < n_radial; r++) {
                        const double v1 = dG_k[r] * jkd;
                        force_grad_out[ib+r] += v1; force_grad_out[jb+r] -= v1;
                    }
                    if (virial_grad_out) {
                        for (int r = 0; r < n_radial; r++) {
                            const double v1 = dG_k[r]*jkd;
                            virial_grad_out[d*n_radial+r]-=v1*rr[d];
                            if(d==0){virial_grad_out[3*n_radial+r]-=v1*rr[1]/2;virial_grad_out[4*n_radial+r]-=v1*rr[2]/2;}
                            else if(d==1){virial_grad_out[3*n_radial+r]-=v1*rr[0]/2;virial_grad_out[5*n_radial+r]-=v1*rr[2]/2;}
                            else{virial_grad_out[4*n_radial+r]-=v1*rr[0]/2;virial_grad_out[5*n_radial+r]-=v1*rr[1]/2;}
                        }
                    }
                }
            }
        }
    }
}

// ===========================================================================
// eval_grad_linear() — ∂(E_i, F_i, virial) / ∂β_linear
//
// Per-atom site-energy gradient = eval_basis (moment values at alpha_moment_mapping).
// Force and virial gradients use the dG_lin analytical backward pass:
//   seed dG_lin[mapping[s], s] = 1, then propagate through composite moments.
//   ∂F_{i,d}/∂β_s = Σ_{jj,k} dG_lin[k,s] * jac[jj,k,d]  + Newton's 3rd.
// ===========================================================================

void MTPPotential::eval_grad_linear(int n_atoms, const int* types, int inum, const int* ilist, const int* numneigh, const int* firstneigh, const double* displacements, double* site_energy_grad_out, double* force_grad_out, double* virial_grad_out) {
    const int n_lin = alpha_scalar_count;

    std::fill(site_energy_grad_out, site_energy_grad_out + (long)inum * n_lin, 0.0);
    std::fill(force_grad_out, force_grad_out + (long)n_atoms * 3 * n_lin, 0.0);
    if (virial_grad_out) std::fill(virial_grad_out, virial_grad_out + 6 * n_lin, 0.0);

    const int buf_size = alpha_moment_count * n_lin;
    if ((int)dG_lin_buf.size() < buf_size) dG_lin_buf.resize(buf_size);

    int nbr_offset = 0;

    for (int ii = 0; ii < inum; ii++) {
        const int i = ilist[ii];
        const int jnum = numneigh[ii];
        const int* nbrs = firstneigh + nbr_offset;
        const double* dr = displacements + nbr_offset * 3;
        nbr_offset += jnum;

        const int valid_count = neighborhood_forward(
            i, types, jnum, nbrs, dr,
            species_count, radial_func_count, radial_basis_size, radial_coeff_count_per_pair,
            max_cutoff_sq, max_alpha_index_basic, alpha_index_basic_count, alpha_index_times_count,
            radial_basis_coeffs.data(), alpha_index_basic, alpha_index_times, radial_basis,
            dist_powers, coord_powers, radial_vals, radial_ders, moment_tensor_vals,
            moment_jacobian, valid_j, valid_dr, jac_cap);

        // Site-energy linear gradient: ∂E_i/∂β_s = M_{mapping[s]}
        double* e_row = site_energy_grad_out + (long)ii * n_lin;
        for (int s = 0; s < n_lin; s++)
            e_row[s] = moment_tensor_vals[alpha_moment_mapping[s]];

        // Build dG_lin[k, s] — backward pass seeded at mapping[s]
        // dG_lin[k, s] = ∂G_k/∂β_s where G_k = dE/dM_k
        // Seed: dG_lin[mapping[s], s] = 1; propagate: same chain as primary backward
        // No dM/dβ term since moment values do not depend on linear coefficients.
        std::fill(dG_lin_buf.begin(), dG_lin_buf.begin() + buf_size, 0.0);
        for (int s = 0; s < n_lin; s++)
            dG_lin_buf[(long)alpha_moment_mapping[s] * n_lin + s] = 1.0;

        for (int k = alpha_index_times_count - 1; k >= 0; k--) {
            const int a0 = alpha_index_times[k][0], a1 = alpha_index_times[k][1];
            const int mul = alpha_index_times[k][2], a3 = alpha_index_times[k][3];
            const double M_a0 = moment_tensor_vals[a0], M_a1 = moment_tensor_vals[a1];
            const double imul = (double)mul;
            const double* dGl_a3 = dG_lin_buf.data() + (long)a3 * n_lin;
            double* dGl_a0 = dG_lin_buf.data() + (long)a0 * n_lin;
            double* dGl_a1 = dG_lin_buf.data() + (long)a1 * n_lin;
            for (int s = 0; s < n_lin; s++) {
                dGl_a0[s] += dGl_a3[s] * imul * M_a1;
                dGl_a1[s] += dGl_a3[s] * imul * M_a0;
            }
        }

        // Force/virial linear gradient: ∂F_{i,d}/∂β_s = Σ_{jj,k} dG_lin[k,s] * jac[jj,k,d]
        for (int jj = 0; jj < valid_count; jj++) {
            const int j = valid_j[jj];
            const auto& rr = valid_dr[jj];
            for (int k = 0; k < alpha_index_basic_count; k++) {
                const double* dGl_k = dG_lin_buf.data() + (long)k * n_lin;
                const auto& jac_k   = moment_jacobian[jj * alpha_index_basic_count + k];
                for (int d = 0; d < 3; d++) {
                    const double jkd = jac_k[d]; if (jkd == 0.0) continue;
                    const long ib = ((long)i*3+d)*n_lin, jb = ((long)j*3+d)*n_lin;
                    for (int s = 0; s < n_lin; s++) {
                        const double val = dGl_k[s] * jkd;
                        force_grad_out[ib+s] += val; force_grad_out[jb+s] -= val;
                    }
                    if (virial_grad_out) {
                        for (int s = 0; s < n_lin; s++) {
                            const double val = dGl_k[s]*jkd;
                            virial_grad_out[d*n_lin+s]-=val*rr[d];
                            if(d==0){virial_grad_out[3*n_lin+s]-=val*rr[1]/2;virial_grad_out[4*n_lin+s]-=val*rr[2]/2;}
                            else if(d==1){virial_grad_out[3*n_lin+s]-=val*rr[0]/2;virial_grad_out[5*n_lin+s]-=val*rr[2]/2;}
                            else{virial_grad_out[4*n_lin+s]-=val*rr[0]/2;virial_grad_out[5*n_lin+s]-=val*rr[1]/2;}
                        }
                    }
                }
            }
        }
    }
}

// ===========================================================================
// eval_grad_all() — ∂(E_i, F_i, virial) / ∂c_all
//
// c_all = [c_radial | c_species | β_linear], shape (*, coeff_count).
// Combines one radial-grad pass and one linear-grad pass, sharing the same
// neighborhood_forward call when possible via the two internal helpers.
//
// site_energy_grad_out : [inum * coeff_count]
//   — per-atom ∂E_i/∂c_all = [radial | species-one-hot | linear]
//   (same layout as eval_grad's existing output)
// force_grad_out       : [n_atoms * 3 * coeff_count]
// virial_grad_out      : [6 * coeff_count]  (nullptr to skip)
// ===========================================================================

void MTPPotential::eval_grad_all(int n_atoms, const int* types, int inum, const int* ilist, const int* numneigh, const int* firstneigh, const double* displacements, double* site_energy_grad_out, double* force_grad_out, double* virial_grad_out) {
    const int cc       = coeff_count();
    const int n_radial = radial_coeff_count;
    const int n_lin    = alpha_scalar_count;

    std::fill(site_energy_grad_out, site_energy_grad_out + (long)inum * cc, 0.0);
    std::fill(force_grad_out,  force_grad_out  + (long)n_atoms * 3 * cc, 0.0);
    if (virial_grad_out) std::fill(virial_grad_out, virial_grad_out + 6 * cc, 0.0);

    // Temporary storage for the three component-specific outputs
    std::vector<double> rad_e(n_radial, 0.0);
    std::vector<double> rad_f((long)n_atoms * 3 * n_radial, 0.0);
    std::vector<double> rad_v(virial_grad_out ? 6 * n_radial : 0, 0.0);
    std::vector<double> lin_e((long)inum * n_lin, 0.0);
    std::vector<double> lin_f((long)n_atoms * 3 * n_lin, 0.0);
    std::vector<double> lin_v(virial_grad_out ? 6 * n_lin : 0, 0.0);

    eval_grad_radial(n_atoms, types, inum, ilist, numneigh, firstneigh, displacements,
                     rad_e.data(), rad_f.data(), virial_grad_out ? rad_v.data() : nullptr);
    eval_grad_linear(n_atoms, types, inum, ilist, numneigh, firstneigh, displacements,
                     lin_e.data(), lin_f.data(), virial_grad_out ? lin_v.data() : nullptr);

    // Site-energy rows: existing eval_grad already gives (inum, coeff_count)
    // which is [radial | species-one-hot | linear_basis_values].
    // Reuse that — just call eval_grad.
    eval_grad(n_atoms, types, inum, ilist, numneigh, firstneigh, displacements, site_energy_grad_out);

    // Assemble force_grad_out = [radial | zeros(species) | linear]
    // Layout: force_grad_out[i * 3 * cc + d * cc + col]
    for (int atom = 0; atom < n_atoms; atom++) {
        for (int d = 0; d < 3; d++) {
            const long base = ((long)atom * 3 + d) * cc;
            // Radial columns
            const double* fr = rad_f.data() + ((long)atom * 3 + d) * n_radial;
            for (int r = 0; r < n_radial; r++)
                force_grad_out[base + r] = fr[r];
            // Species columns: 0 (forces don't depend on species reference energies)
            // Linear columns
            const double* fl = lin_f.data() + ((long)atom * 3 + d) * n_lin;
            for (int s = 0; s < n_lin; s++)
                force_grad_out[base + n_radial + species_count + s] = fl[s];
        }
    }

    // Assemble virial_grad_out = [radial | zeros | linear]
    if (virial_grad_out) {
        for (int ab = 0; ab < 6; ab++) {
            const long base = (long)ab * cc;
            for (int r = 0; r < n_radial; r++)
                virial_grad_out[base + r] = rad_v[ab * n_radial + r];
            for (int s = 0; s < n_lin; s++)
                virial_grad_out[base + n_radial + species_count + s] = lin_v[ab * n_lin + s];
        }
    }
}

// ===========================================================================
// eval_radial_basis()
// ===========================================================================

void MTPPotential::eval_radial_basis(double dist, double* vals_out, double* ders_out) {
    if (ders_out)
        radial_basis->calc_radial_basis_ders(dist);
    else
        radial_basis->calc_radial_basis(dist);

    for (int i = 0; i < radial_basis_size; i++)
        vals_out[i] = radial_basis->radial_basis_vals[i];
    if (ders_out)
        for (int i = 0; i < radial_basis_size; i++)
            ders_out[i] = radial_basis->radial_basis_ders[i];
}
