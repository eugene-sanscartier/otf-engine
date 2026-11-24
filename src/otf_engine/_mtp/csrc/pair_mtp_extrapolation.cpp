/* -*- c++ -*- ----------------------------------------------------------
   Standalone MTP extrapolation grading — no LAMMPS dependency.
   Ported from lammps-mtp/src/ML-MTP/pair_mtp_extrapolation.cpp
   Original author: Richard Meng, Queen's University at Kingston, 10.02.25
------------------------------------------------------------------------- */

#include "pair_mtp_extrapolation.h"

#include "maxvol.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

void PairMTPExtrapolation::set_active_set(const double* invA, bool configuration_mode, int weight_scaling) {
    const size_t n = (size_t) coeff_count() * coeff_count();
    inverse_active_set.assign(invA, invA + n);
    energy_ders_wrt_coeffs.resize((size_t) chunk_size * coeff_count());
    this->configuration_mode = configuration_mode;
    this->weight_scaling = weight_scaling;
}

/* ----------------------------------------------------------------------
   Straightforward MTP implementation based on MLIP3, a chunk of atoms at
   a time as PairMTP::compute takes them; each atom's site-energy gradient
   is filled after the chunk's backwards pass
   ---------------------------------------------------------------------- */
void PairMTPExtrapolation::compute(const NeighList& list) {
    if (!has_active_set())
        throw std::runtime_error("PairMTPExtrapolation: no active set — call set_active_set() first");

    max_grade = 0;

    int nbr_offset = 0;

    energy = 0.0;
    eatom.assign(list.n_atoms, 0.0);
    forces.assign((size_t) list.n_atoms * 3, 0.0);
    virial.fill(0.0);

    // Resize the nbh extrapolation grades if needed.
    if (!configuration_mode)
        nbh_extrapolation_grades.assign(list.n_atoms, 0.0);

    // If are in configuration, we need to reset the working array once per compute call / config
    if (configuration_mode)
        std::fill(energy_ders_wrt_coeffs.begin(), energy_ders_wrt_coeffs.end(), 0.0);

    grow_chunk_cache(list);
    if (radial_basis_cache_size < chunk_size * atom_cache_size) {
        radial_basis_cache.resize((size_t) chunk_size * atom_cache_size * radial_basis_size);
        radial_basis_cache_size = chunk_size * atom_cache_size;
    }

    // Loop over all provided neighbourhoods, a chunk of them at a time
    for (int ii_begin = 0; ii_begin < list.inum; ii_begin += chunk_size) {
        const int ii_end = std::min(ii_begin + chunk_size, list.inum);

        // Reset the working arrays
        std::fill(moment_tensor_vals.begin(), moment_tensor_vals.end(), Lanes());
        std::fill(nbh_energy_ders_wrt_moments.begin(), nbh_energy_ders_wrt_moments.end(), Lanes());

        for (int ii = ii_begin; ii < ii_end; ii++) {
            int valid_count = 0;
            const int i = list.ilist[ii];
            const int itype = list.types[i];
            const int jnum = list.numneigh[ii];
            const int* nbrs = list.firstneigh + nbr_offset;
            const double* dr = list.displacements + nbr_offset * 3;
            nbr_offset += jnum;

            select_atom_cache(ii - ii_begin);
            double* atom_radial_basis = radial_basis_cache.data() + (size_t) (ii - ii_begin) * atom_cache_size * radial_basis_size;

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
                double* cache = neighbor_cache + (size_t) valid_count * cache_stride;
                cache[0] = inv_dist;
                double* vals = cache + 1;
                double* ders = vals + radial_func_count;
                radial_basis->calc_radial_basis_ders(dist);
                const double* basis_vals = radial_basis->radial_basis_vals.data();
                const double* basis_ders = radial_basis->radial_basis_ders.data();
                std::copy(basis_vals, basis_vals + radial_basis_size, atom_radial_basis + (size_t) valid_count * radial_basis_size);

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
                        moment_tensor_vals[k][ii - ii_begin] += val * angular_vals[angular_by_mu[t]];
                    }
                }
                valid_count++;
            }
            cached_counts[ii - ii_begin] = valid_count;
        }

        // ------------ Construct Composite Moment Values  ------------
        for (int k = 0; k < alpha_index_times_count; k++) {
            const int* term = alpha_index_times[k].data();
            moment_tensor_vals[term[3]] +=
                term[2] * moment_tensor_vals[term[0]] * moment_tensor_vals[term[1]];
        }

        // ------------ Compute Basis Set From Alpha Map ------------
        Lanes nbh_energy;
        for (int ii = ii_begin; ii < ii_end; ii++)
            nbh_energy[ii - ii_begin] = species_coeffs[list.types[list.ilist[ii]]];    // Essentially the reference point energy per species
        for (int k = 0; k < alpha_scalar_count; k++)
            nbh_energy += linear_coeffs[k] * moment_tensor_vals[alpha_moment_mapping[k]];

        for (int ii = ii_begin; ii < ii_end; ii++) {
            eatom[list.ilist[ii]] = nbh_energy[ii - ii_begin];
            energy += nbh_energy[ii - ii_begin];
        }

        // =========== Begin Backpropagation ===========
        //------------ NBH energy derivative is the corresponding linear combination------------
        for (int k = 0; k < alpha_scalar_count; k++)
            nbh_energy_ders_wrt_moments[alpha_moment_mapping[k]] = linear_coeffs[k];

        //------------ Propagate chain rule through the composite moment elements times to the basics ------------
        for (int k = alpha_index_times_count - 1; k >= 0; k--) {
            const int* term = alpha_index_times[k].data();
            const int a0 = term[0];
            const int a1 = term[1];

            const Lanes w = term[2] * nbh_energy_ders_wrt_moments[term[3]];

            nbh_energy_ders_wrt_moments[a1] += w * moment_tensor_vals[a0];
            nbh_energy_ders_wrt_moments[a0] += w * moment_tensor_vals[a1];
        }

        for (int ii = ii_begin; ii < ii_end; ii++) {
            const int i = list.ilist[ii];
            const int itype = list.types[i];
            const int valid_count = cached_counts[ii - ii_begin];
            const double* atom_radial_basis = radial_basis_cache.data() + (size_t) (ii - ii_begin) * atom_cache_size * radial_basis_size;

            select_atom_cache(ii - ii_begin);

            // The configuration's candidate vector, or the atom's row of the chunk's
            double* atom_ders = energy_ders_wrt_coeffs.data() + (configuration_mode ? 0 : (size_t) (ii - ii_begin) * coeff_count());
            if (!configuration_mode)
                std::fill(atom_ders, atom_ders + coeff_count(), 0.0);

            // The site energy's derivatives w.r.t. the linear and species coefficients
            const int linear_basis_offset = radial_coeff_count + species_count;
            for (int k = 0; k < alpha_scalar_count; k++)
                atom_ders[linear_basis_offset + k] += moment_tensor_vals[alpha_moment_mapping[k]][ii - ii_begin];
            atom_ders[radial_coeff_count + itype] += 1;

            for (int t = 0; t < alpha_index_basic_count; t++)
                basic_ders_by_mu[t] = nbh_energy_ders_wrt_moments[basic_by_mu[t]][ii - ii_begin];

            //------------ Compute forces from basic moment derivatives ------------
            for (int jj = 0; jj < valid_count; jj++) {
                const int j = cached_j[jj];
                const auto& r = valid_dr[jj];
                const double* cache = neighbor_cache + (size_t) jj * cache_stride;
                const double inv_dist = cache[0];
                const double u[3] = {r[0] * inv_dist, r[1] * inv_dist, r[2] * inv_dist};
                const double* vals = cache + 1;
                const double* ders = vals + radial_func_count;
                const double* basis_vals = atom_radial_basis + (size_t) jj * radial_basis_size;
                double* pair_ders = atom_ders +
                    (itype * species_count + list.types[j]) * radial_coeff_count_per_pair;
                for (int k = 1; k < angular_count; k++)
                    angular_vals[k] = angular_vals[angular_parent[k]] * u[angular_axis[k]];
                std::fill(angular_ders.begin(), angular_ders.end(), 0.0);

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
                        angular_ders[angular] += val * adj;
                    }
                    radial_force += ders[mu] * radial_sum;
                    double* coeff_ders = pair_ders + mu * radial_basis_size;
                    for (int ri = 0; ri < radial_basis_size; ri++)
                        coeff_ders[ri] += radial_sum * basis_vals[ri];
                }

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

                // Accumulate virial stress as the reference's ev_tally_xyz, whose del is -r
                virial[0] -= r[0] * temp_force[0];    //xx
                virial[1] -= r[1] * temp_force[1];    //yy
                virial[2] -= r[2] * temp_force[2];    //zz
                virial[3] -= r[0] * temp_force[1];    //xy
                virial[4] -= r[0] * temp_force[2];    //xz
                virial[5] -= r[1] * temp_force[2];    //yz
            }
        }

        // The chunk's neighbourhood grades, at once
        if (!configuration_mode) {
            const int count = ii_end - ii_begin;
            chunk_grades.resize(count);
            grade_rows(inverse_active_set.data(), coeff_count(), energy_ders_wrt_coeffs.data(), count, chunk_grades.data(), grade_block);
            for (int ii = ii_begin; ii < ii_end; ii++) {
                nbh_extrapolation_grades[list.ilist[ii]] = chunk_grades[ii - ii_begin];
                max_grade = std::max(chunk_grades[ii - ii_begin], max_grade);
            }
        }
    }

    compile_grades(list.n_atoms);
}

/* ----------------------------------------------------------------------
   Configuration grade, normalized by the atom count
------------------------------------------------------------------------- */
void PairMTPExtrapolation::compile_grades(int natoms) {
    if (configuration_mode) {    // Configuration mode
        grade_rows(inverse_active_set.data(), coeff_count(), energy_ders_wrt_coeffs.data(), 1, &max_grade, grade_block);

        if (natoms > 0)
            max_grade /= std::pow((double) natoms, 0.5 * weight_scaling);    // Normalize
        else
            max_grade = 0.0;
    }
}
