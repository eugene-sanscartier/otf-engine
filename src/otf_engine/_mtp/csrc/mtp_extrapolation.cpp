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
double PairMTPExtrapolation::calculate_extrapolation_grade(int itype) {
    const int n = coeff_count();
    const int begin = itype < 0 ? 0 : itype * species_count * radial_coeff_count_per_pair;
    const int end = itype < 0 ? n : begin + species_count * radial_coeff_count_per_pair;
    const int linear_offset = radial_coeff_count + species_count;
    double grade_max = 0;
    for (int i = 0; i < n; i++) {
        double current_grade = 0;
        const double* row = inverse_active_set.data() + (size_t) i * n;
        for (int j = begin; j < end; j++) { current_grade += energy_ders_wrt_coeffs[j] * row[j]; }
        if (itype >= 0) {
            const int species_offset = radial_coeff_count + itype;
            current_grade += energy_ders_wrt_coeffs[species_offset] * row[species_offset];
            for (int j = linear_offset; j < n; j++)
                current_grade += energy_ders_wrt_coeffs[j] * row[j];
        }
        grade_max = std::max(std::abs(current_grade), grade_max);
    }
    return grade_max;
}

/* ---------------------------------------------------------------------- */
double PairMTPExtrapolation::compute(const NeighList& list, double* forces, double* virial, double* eatom, double* rows_out, double* grades_out) {
    // Neither the information vector nor a grade is wanted
    if (!rows_out && !grades_out)
        return PairMTP::compute(list, forces, virial, eatom);

    if (grades_out && !has_active_set())
        throw std::runtime_error("PairMTPExtrapolation: no active set — call set_active_set() first");

    const int cc = coeff_count();
    const int stride = 1 + 2 * radial_func_count;
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
        if (cache_size < jnum) {
            neighbor_cache.resize((size_t) jnum * stride);
            cached_j.resize(jnum);
            valid_dr.resize(jnum);
            cache_size = jnum;
        }
        if (radial_basis_cache_size < jnum) {
            radial_basis_cache.resize((size_t) jnum * radial_basis_size);
            radial_basis_cache_size = jnum;
        }

        // Reset the working arrays
        std::fill(moment_tensor_vals.begin(), moment_tensor_vals.end(), 0.0);
        std::fill(nbh_energy_ders_wrt_moments.begin(), nbh_energy_ders_wrt_moments.end(), 0.0);
        std::fill(energy_ders_wrt_coeffs.begin(), energy_ders_wrt_coeffs.end(), 0.0);

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
            const double inv_dist = 1.0 / dist;
            const double u[3] = {r[0] * inv_dist, r[1] * inv_dist, r[2] * inv_dist};
            double* cache = neighbor_cache.data() + (size_t) valid_count * stride;
            cache[0] = inv_dist;
            double* vals = cache + 1;
            double* ders = vals + radial_func_count;
            radial_basis->calc_radial_basis_ders(dist);
            const double* basis_vals = radial_basis->radial_basis_vals.data();
            const double* basis_ders = radial_basis->radial_basis_ders.data();
            std::copy(basis_vals, basis_vals + radial_basis_size, radial_basis_cache.data() + (size_t) valid_count * radial_basis_size);

            // Evaluate each shared angular monomial once.
            for (int k = 1; k < angular_count; k++)
                angular_vals[k] = angular_vals[angular_parent[k]] * u[angular_axis[k]];

            // Compute the radial basis values and derivatives
            const int pair_offset = itype * species_count + jtype;
            for (int mu = 0; mu < radial_func_count; mu++) {
                double val = 0;
                double der = 0;
                const int offset = (pair_offset * radial_coeff_count_per_pair) + mu * radial_basis_size;

                for (int ri = 0; ri < radial_basis_size; ri++) {
                    val += radial_basis_coeffs[offset + ri] * basis_vals[ri];
                    der += radial_basis_coeffs[offset + ri] * basis_ders[ri];
                }
                vals[mu] = val;
                ders[mu] = der;

                // Accumulate into the basic moment elements
                for (int t = mu_offsets[mu]; t < mu_offsets[mu + 1]; t++) {
                    const int k = basic_by_mu[t];
                    const double ang = angular_vals[angular_by_mu[t]];
                    moment_tensor_vals[k] += val * ang;
                }
            }
            valid_count++;
        }

        // ------------ Construct Composite Moment Values  ------------
        for (int k = 0; k < alpha_index_times_count; k++) {
            const int* term = alpha_index_times[k].data();
            moment_tensor_vals[term[3]] +=
                term[2] * moment_tensor_vals[term[0]] * moment_tensor_vals[term[1]];
        }

        // ------------ Compute Basis Set From Alpha Map ------------
        const int linear_basis_offset = radial_coeff_count + species_count;
        nbh_energy = species_coeffs[itype];    // Essentially the reference point energy per species
        for (int k = 0; k < alpha_scalar_count; k++) {
            double basis_member = moment_tensor_vals[alpha_moment_mapping[k]];
            energy_ders_wrt_coeffs[linear_basis_offset + k] += basis_member;
            nbh_energy += linear_coeffs[k] * basis_member;
        }
        total_energy += nbh_energy;
        if (eatom) eatom[i] = nbh_energy;

        energy_ders_wrt_coeffs[radial_coeff_count + itype] += 1;

        // =========== Begin Backpropagation ===========
        //------------ NBH energy derivative is the corresponding linear combination------------
        for (int k = 0; k < alpha_scalar_count; k++)
            nbh_energy_ders_wrt_moments[alpha_moment_mapping[k]] = linear_coeffs[k];

        //------------ Propagate chain rule through the composite moment elements times to the basics ------------
        for (int k = alpha_index_times_count - 1; k >= 0; k--) {
            const int* term = alpha_index_times[k].data();
            const int a0 = term[0];
            const int a1 = term[1];

            const double w = term[2] * nbh_energy_ders_wrt_moments[term[3]];

            nbh_energy_ders_wrt_moments[a1] += w * moment_tensor_vals[a0];
            nbh_energy_ders_wrt_moments[a0] += w * moment_tensor_vals[a1];
        }

        for (int t = 0; t < alpha_index_basic_count; t++)
            basic_ders_by_mu[t] = nbh_energy_ders_wrt_moments[basic_by_mu[t]];

        //------------ Compute forces from basic moment derivatives ------------
        for (int jj = 0; jj < valid_count; jj++) {
            const int j = cached_j[jj];
            const auto& r = valid_dr[jj];
            const double* cache = neighbor_cache.data() + (size_t) jj * stride;
            const double inv_dist = cache[0];
            const double u[3] = {r[0] * inv_dist, r[1] * inv_dist, r[2] * inv_dist};
            const double* vals = cache + 1;
            const double* ders = vals + radial_func_count;
            const double* basis_vals = radial_basis_cache.data() + (size_t) jj * radial_basis_size;
            double* pair_ders = energy_ders_wrt_coeffs.data() +
                (itype * species_count + list.types[j]) * radial_coeff_count_per_pair;
            for (int k = 1; k < angular_count; k++)
                angular_vals[k] = angular_vals[angular_parent[k]] * u[angular_axis[k]];
            if (forces) std::fill(angular_ders.begin(), angular_ders.end(), 0.0);

            double temp_force[3] = {0, 0, 0};
            double radial_force = 0;
            for (int mu = 0; mu < radial_func_count; mu++) {
                const int end = mu_offsets[mu + 1];
                if (mu_offsets[mu] == end) continue;
                const double val = vals[mu];
                double radial_sum = 0;
                for (int t = mu_offsets[mu]; t < end; t++) {
                    const int angular = angular_by_mu[t];
                    const double adj = basic_ders_by_mu[t];
                    radial_sum += adj * angular_vals[angular];
                    if (forces) angular_ders[angular] += val * adj;
                }
                radial_force += ders[mu] * radial_sum;
                double* coeff_ders = pair_ders + mu * radial_basis_size;
                for (int ri = 0; ri < radial_basis_size; ri++)
                    coeff_ders[ri] += radial_sum * basis_vals[ri];
            }

            // The information vector alone needs no forces
            if (!forces) continue;

            // Reverse the shared angular products
            for (int k = angular_count - 1; k > 0; k--) {
                const int parent = angular_parent[k];
                const int axis = angular_axis[k];
                const double adj = angular_ders[k];
                temp_force[axis] += adj * angular_vals[parent];
                angular_ders[parent] += adj * u[axis];
            }
            radial_force -=
                inv_dist * (temp_force[0] * u[0] + temp_force[1] * u[1] + temp_force[2] * u[2]);
            for (int a = 0; a < 3; a++) temp_force[a] = inv_dist * temp_force[a] + radial_force * u[a];

            forces[i * 3 + 0] += temp_force[0];
            forces[i * 3 + 1] += temp_force[1];
            forces[i * 3 + 2] += temp_force[2];

            forces[j * 3 + 0] -= temp_force[0];
            forces[j * 3 + 1] -= temp_force[1];
            forces[j * 3 + 2] -= temp_force[2];

            // Accumulate virial stress only if requested
            if (virial) {
                virial[0] -= temp_force[0] * r[0];    //xx
                virial[1] -= temp_force[1] * r[1];    //yy
                virial[2] -= temp_force[2] * r[2];    //zz

                virial[3] -= (temp_force[0] * r[1] + temp_force[1] * r[0]) / 2;    //xy
                virial[4] -= (temp_force[0] * r[2] + temp_force[2] * r[0]) / 2;    //xz
                virial[5] -= (temp_force[1] * r[2] + temp_force[2] * r[1]) / 2;    //yz
            }
        }

        // The per-atom row is what select_add needs; the reference discards it.
        if (rows_out)
            std::copy(energy_ders_wrt_coeffs.begin(), energy_ders_wrt_coeffs.end(), rows_out + (size_t) ii * cc);

        if (configuration_mode) {
            for (int k = 0; k < cc; k++)
                cfg_ders_wrt_coeffs[k] += energy_ders_wrt_coeffs[k];
        } else if (grades_out) {
            // Directly calculate extrapolation grade for neighbourhood mode
            double grade = calculate_extrapolation_grade(itype);
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
