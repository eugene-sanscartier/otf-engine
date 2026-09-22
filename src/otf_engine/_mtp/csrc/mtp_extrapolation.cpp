/* -*- c++ -*- ----------------------------------------------------------
   Standalone MTP extrapolation grading — no LAMMPS dependency.
   Ported from lammps-mtp/src/ML-MTP/pair_mtp_extrapolation.cpp
   Original author: Richard Meng, Queen's University at Kingston, 10.02.25
------------------------------------------------------------------------- */

#include "mtp_extrapolation.h"

#include <cmath>
#include <stdexcept>

void PairMTPExtrapolation::set_active_set(const double* invA, bool configuration_mode) {
    const size_t n = (size_t) coeff_count() * coeff_count();
    inverse_active_set.assign(invA, invA + n);
    this->configuration_mode = configuration_mode;
}

/* ----------------------------------------------------------------------
   Extrapolation Calculation Function
------------------------------------------------------------------------- */
double PairMTPExtrapolation::calculate_extrapolation_grade() {
    const int n = coeff_count();
    double max_grade = 0;
    for (int i = 0; i < n; i++) {
        double current_grade = 0;
        const double* row = inverse_active_set.data() + (size_t) i * n;
        for (int j = 0; j < n; j++) {
            current_grade += energy_ders_wrt_coeffs[j] * row[j];
        }
        max_grade = std::max(std::abs(current_grade), max_grade);
    }
    return max_grade;
}

/* ----------------------------------------------------------------------
   Straightfoward MTP implementation based on MLIP3
   ---------------------------------------------------------------------- */
double PairMTPExtrapolation::compute(const NeighList& list, double* forces, double* virial, double* eatom, double* rows_out, double* grades_out) {
    // Neither the information vector nor a grade is wanted
    if (!rows_out && !grades_out)
        return PairMTP::compute(list, forces, virial, eatom);

    if (grades_out && !has_active_set())
        throw std::runtime_error("PairMTPExtrapolation: no active set — call set_active_set() first");

    const int cc = coeff_count();
    const int linear_basis_offset = radial_coeff_count + species_count;
    const size_t radial_jacobian_size = (size_t) alpha_index_basic_count * species_count * radial_coeff_count_per_pair;

    radial_jacobian.resize(radial_jacobian_size);
    energy_ders_wrt_coeffs.resize(cc);

    max_grade = 0;
    double total_energy = 0.0;
    int nbr_offset = 0;

    // In configuration mode the information vector covers the whole cell, so
    // its accumulator is reset once per compute call rather than per atom.
    if (configuration_mode)
        cfg_ders_wrt_coeffs.assign(cc, 0.0);

    // Loop over all provided neighbourhoods
    for (int ii = 0; ii < list.inum; ii++) {
        int valid_count = 0;
        const int i = list.ilist[ii];
        const int itype = list.types[i];
        const int jnum = list.numneigh[ii];
        const int* nbrs = list.firstneigh + nbr_offset;
        const double* dr = list.displacements + nbr_offset * 3;
        double nbh_energy = 0;
        nbr_offset += jnum;

        // Resize per neighbor arrays
        if (jac_size < jnum) {
            jac_size = jnum;
            moment_jacobian.resize((size_t) jac_size * alpha_index_basic_count);
            valid_j.resize(jac_size);
            valid_dr.resize(jac_size);
        }

        // Reset the working arrays
        std::fill(moment_tensor_vals.begin(), moment_tensor_vals.end(), 0.0);
        std::fill(nbh_energy_ders_wrt_moments.begin(), nbh_energy_ders_wrt_moments.end(), 0.0);
        std::fill(radial_jacobian.begin(), radial_jacobian.end(), 0.0);
        std::fill(energy_ders_wrt_coeffs.begin(), energy_ders_wrt_coeffs.end(), 0.0);

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

            // Precompute the coord and distance power
            for (int k = 1; k < max_alpha_index_basic; k++) {
                dist_powers[k] = dist_powers[k - 1] * dist;
                for (int a = 0; a < 3; a++) coord_powers[k][a] = coord_powers[k - 1][a] * r[a];
            }

            // Compute the radial basis values and derivatives
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

            // Accumulate into the basic moment elements
            for (int k = 0; k < alpha_index_basic_count; k++) {
                int mu = alpha_index_basic[k][0];

                double val = radial_vals[mu];
                double der = radial_ders[mu];

                // Normalize by the rank of alpha's coresponding tensor
                int norm_rank = alpha_index_basic[k][1] + alpha_index_basic[k][2] + alpha_index_basic[k][3];
                double norm_fac = 1.0 / dist_powers[norm_rank];

                double pow0 = coord_powers[alpha_index_basic[k][1]][0];
                double pow1 = coord_powers[alpha_index_basic[k][2]][1];
                double pow2 = coord_powers[alpha_index_basic[k][3]][2];
                double pow = pow0 * pow1 * pow2;

                // Calculate the radial jacobian
                int mu_offset = mu * radial_basis_size;
                double* jac_row = radial_jacobian.data() + ((size_t) k * species_count + jtype) * radial_coeff_count_per_pair + mu_offset;
                for (int ri = 0; ri < radial_basis_size; ri++)
                    jac_row[ri] += radial_basis->radial_basis_vals[ri] * norm_fac * pow;

                val *= norm_fac;
                der = der * norm_fac - norm_rank * val / dist;
                moment_tensor_vals[k] += val * pow;

                // Calculate the Jacobian from derivatives
                const size_t jac = (size_t) valid_count * alpha_index_basic_count + k;
                pow *= der / dist;
                moment_jacobian[jac][0] = pow * r[0];
                moment_jacobian[jac][1] = pow * r[1];
                moment_jacobian[jac][2] = pow * r[2];
                if (alpha_index_basic[k][1] != 0) {
                    moment_jacobian[jac][0] += val * alpha_index_basic[k][1] *
                        coord_powers[alpha_index_basic[k][1] - 1][0] * pow1 * pow2;
                }    //Chain rule for nonzero rank
                if (alpha_index_basic[k][2] != 0) {
                    moment_jacobian[jac][1] += val * alpha_index_basic[k][2] * pow0 *
                        coord_powers[alpha_index_basic[k][2] - 1][1] * pow2;
                }    //Chain rule for nonzero rank
                if (alpha_index_basic[k][3] != 0) {
                    moment_jacobian[jac][2] += val * alpha_index_basic[k][3] * pow0 * pow1 *
                        coord_powers[alpha_index_basic[k][3] - 1][2];
                }    //Chain rule for nonzero rank
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

        // ------------ Compute Basis Set From Alpha Map ------------
        nbh_energy = species_coeffs[itype];    // Essentially the reference point energy per species
        for (int k = 0; k < alpha_scalar_count; k++) {
            double basis_member = moment_tensor_vals[alpha_moment_mapping[k]];
            energy_ders_wrt_coeffs[linear_basis_offset + k] = basis_member;
            nbh_energy += linear_coeffs[k] * basis_member;
        }
        energy_ders_wrt_coeffs[radial_coeff_count + itype] = 1;

        total_energy += nbh_energy;
        if (eatom) eatom[i] = nbh_energy;

        // =========== Begin Backpropogation ===========
        //------------ NBH energy derivative is the corresponding linear combination------------
        for (int k = 0; k < alpha_scalar_count; k++)
            nbh_energy_ders_wrt_moments[alpha_moment_mapping[k]] = linear_coeffs[k];

        //------------ Propogate chain rule through the composite moment elements times to the basics ------------
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

        //------------  Multiply energy ders wrt basic moments by the Jacobian to get forces ------------
        if (forces) {
            for (int jj = 0; jj < valid_count; jj++) {
                int j = valid_j[jj];

                double temp_force[3] = {0, 0, 0};
                for (int k = 0; k < alpha_index_basic_count; k++)
                    for (int a = 0; a < 3; a++)
                        temp_force[a] += nbh_energy_ders_wrt_moments[k] * moment_jacobian[(size_t) jj * alpha_index_basic_count + k][a];

                forces[i * 3 + 0] += temp_force[0];
                forces[i * 3 + 1] += temp_force[1];
                forces[i * 3 + 2] += temp_force[2];

                forces[j * 3 + 0] -= temp_force[0];
                forces[j * 3 + 1] -= temp_force[1];
                forces[j * 3 + 2] -= temp_force[2];

                if (virial) {
                    const auto& r = valid_dr[jj];
                    virial[0] -= temp_force[0] * r[0];    //xx
                    virial[1] -= temp_force[1] * r[1];    //yy
                    virial[2] -= temp_force[2] * r[2];    //zz

                    virial[3] -= (temp_force[0] * r[1] + temp_force[1] * r[0]) / 2;    //xy
                    virial[4] -= (temp_force[0] * r[2] + temp_force[2] * r[0]) / 2;    //xz
                    virial[5] -= (temp_force[1] * r[2] + temp_force[2] * r[1]) / 2;    //yz
                }
            }
        }

        //------------ Multiply energy ders wrt moment by the radial jacobian to get rad ders ------------
        for (int k = 0; k < alpha_index_basic_count; k++) {
            const double der = nbh_energy_ders_wrt_moments[k];
            if (der == 0.0) continue;
            for (int jjtype = 0; jjtype < species_count; jjtype++) {
                const int offset = (itype * species_count + jjtype) * radial_coeff_count_per_pair;
                const double* jac_row = radial_jacobian.data() + ((size_t) k * species_count + jjtype) * radial_coeff_count_per_pair;
                for (int ri = 0; ri < radial_coeff_count_per_pair; ri++)
                    energy_ders_wrt_coeffs[offset + ri] += der * jac_row[ri];
            }
        }

        // The per-atom row is what select_add needs; the reference discards it.
        if (rows_out)
            std::copy(energy_ders_wrt_coeffs.begin(), energy_ders_wrt_coeffs.end(), rows_out + (size_t) ii * cc);

        if (configuration_mode) {
            for (int k = 0; k < cc; k++)
                cfg_ders_wrt_coeffs[k] += energy_ders_wrt_coeffs[k];
        } else if (grades_out) {
            // Directly calculate extraplation grade for neighbourhood mode
            double grade = calculate_extrapolation_grade();
            max_grade = std::max(grade, max_grade);
            grades_out[ii] = grade;
        }
    }

    if (configuration_mode && grades_out) {
        energy_ders_wrt_coeffs.swap(cfg_ders_wrt_coeffs);
        max_grade = calculate_extrapolation_grade();
        energy_ders_wrt_coeffs.swap(cfg_ders_wrt_coeffs);

        max_grade = (list.n_atoms > 0) ? max_grade / list.n_atoms : 0.0;    // Normalize
        std::fill(grades_out, grades_out + list.inum, max_grade);
    }

    return total_energy;
}

/* ---------------------------------------------------------------------- */
double PairMTPExtrapolation::grade(const NeighList& list, double* grades_out) {
    compute(list, nullptr, nullptr, nullptr, nullptr, grades_out);
    return max_grade;
}
