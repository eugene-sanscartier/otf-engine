/* -*- c++ -*- ----------------------------------------------------------
   Standalone MTP potential — no LAMMPS dependency.
   Ported from lammps-mtp/src/ML-MTP/pair_mtp.cpp
   Original author: Richard Meng, Queen's University at Kingston, 22.11.24
------------------------------------------------------------------------- */

#include "pair_mtp.h"

#include "mtp_basis.h"
#include "text_file_reader.h"

#include <cmath>
#include <fstream>
#include <map>
#include <stdexcept>
#include <utility>

PairMTP::PairMTP(const std::string& filename) {
    std::ifstream f(filename);
    if (!f.is_open())
        throw std::runtime_error("PairMTP: cannot open file '" + filename + "'");
    PairMTP::read_file(f);
}

void PairMTP::get_coeffs(double* c) const {
    std::copy(radial_basis_coeffs.begin(), radial_basis_coeffs.end(), c);
    std::copy(species_coeffs.begin(), species_coeffs.end(), c + radial_coeff_count);
    std::copy(linear_coeffs.begin(), linear_coeffs.end(), c + radial_coeff_count + species_count);
}

void PairMTP::set_coeffs(const double* c) {
    set_radial_basis_coeffs(c);
    set_species_coeffs(c + radial_coeff_count);
    set_linear_coeffs(c + radial_coeff_count + species_count);
}

void PairMTP::set_species_count(int n) {
    const int kept = std::min(n, species_count);
    std::vector<double> sized((size_t) n * n * radial_coeff_count_per_pair, 0.0);
    for (int si = 0; si < kept; si++)
        for (int sj = 0; sj < kept; sj++)
            std::copy_n(radial_basis_coeffs.begin() + (si * species_count + sj) * radial_coeff_count_per_pair, radial_coeff_count_per_pair, sized.begin() + (si * n + sj) * radial_coeff_count_per_pair);
    radial_basis_coeffs = std::move(sized);
    species_coeffs.resize(n, 0.0);
    species_count = n;
    radial_coeff_count = n * n * radial_coeff_count_per_pair;
}

/* ----------------------------------------------------------------------
   Main Computation Function
   ---------------------------------------------------------------------- */
void PairMTP::compute(const NeighList& list) {
    int nbr_offset = 0;

    energy = 0.0;
    eatom.assign(list.n_atoms, 0.0);
    forces.assign((size_t) list.n_atoms * 3, 0.0);
    virial.fill(0.0);

    grow_chunk_cache(list);

    // Loop over all provided neighbourhoods, a chunk of them at a time
    for (int ii_begin = 0; ii_begin < list.inum; ii_begin += chunk_size) {
        const int ii_end = std::min(ii_begin + chunk_size, list.inum);

        // Clear moment and derivative arrays
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
            const int valid_count = cached_counts[ii - ii_begin];

            select_atom_cache(ii - ii_begin);

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
    }
}

/* ----------------------------------------------------------------------
   MTP file parsing. Includes allocation. Excludes the radial basis
   hyperparameters, which the radial basis constructor reads.
------------------------------------------------------------------------- */
void PairMTP::read_file(std::istream& is) {
    TextFileReader tfr(is);

    // An untrained potential may omit the species count, the radial coeffs
    // and the linear coeffs; mlip-3's MLMTPR::Load gives them defaults.
    read_header(tfr, "MTP", "1.1.0");

    read_radial_basis(tfr);

    radial_func_count = tfr.next_int("radial_funcs_count");
    if (radial_func_count < 1)
        throw std::runtime_error("PairMTP: MTP radial function count must be positive");

    tfr.advance();
    if (tfr.keyword() == "magnetic_basis_type")
        throw std::runtime_error("PairMTP: magnetic basis is currently not supported");

    // Read the radial basis coeffs, which an untrained potential may omit
    const int pairs_count = species_count * species_count;
    const int coeffs_per_pair = radial_basis_size * radial_func_count;
    const bool has_radial_coeffs = tfr.keyword() == "radial_coeffs";
    std::vector<double> radial_coeffs_read(has_radial_coeffs ? pairs_count * coeffs_per_pair : 0);
    for (int i = 0; has_radial_coeffs && i < pairs_count; i++) {
        // pair-type label "0-1"; '-' is a separator only on this line, since
        // the coefficient lines below carry negative numbers
        if (!tfr.next_line("-"))
            throw std::runtime_error("PairMTP: unexpected end of file in radial_coeffs");
        int type1, type2;
        std::istringstream label(tfr.line());
        if (!(label >> type1 >> type2))
            throw std::runtime_error("PairMTP: cannot read radial_coeffs pair label");
        if (type1 < 0 || type1 >= species_count || type2 < 0 || type2 >= species_count)
            throw std::runtime_error("PairMTP: invalid species pair " + std::to_string(type1) + "-" + std::to_string(type2) + " in MTP radial coefficients");

        const int pair_offset = (type1 * species_count + type2) * coeffs_per_pair;
        for (int j = 0; j < radial_func_count; j++) {
            tfr.advance();
            // The whole line is coefficients — there is no leading keyword.
            std::istringstream ss(tfr.line());
            for (int k = 0; k < radial_basis_size; k++)
                if (!(ss >> radial_coeffs_read[pair_offset + j * radial_basis_size + k]))
                    throw std::runtime_error("PairMTP: not enough radial coefficients");
        }
    }

    if (has_radial_coeffs)
        alpha_moment_count = tfr.next_int("alpha_moments_count");
    else if (tfr.keyword() != "alpha_moments_count" || !(tfr.rest() >> alpha_moment_count))
        throw std::runtime_error("PairMTP: cannot read radial coeffs");
    if (alpha_moment_count < 1)
        throw std::runtime_error("PairMTP: MTP alpha moment count must be positive");

    alpha_index_basic_count = tfr.next_int("alpha_index_basic_count");
    if (alpha_index_basic_count < 1 || alpha_index_basic_count > alpha_moment_count)
        throw std::runtime_error("PairMTP: MTP alpha index basic count is out of range");

    // Read the basic alphas
    tfr.expect("alpha_index_basic");
    alpha_index_basic.resize(alpha_index_basic_count);
    {
        std::istringstream ss = tfr.rest();
        int radial_func_max = 0;
        for (int i = 0; i < alpha_index_basic_count; i++) {
            for (int j = 0; j < 4; j++)
                if (!(ss >> alpha_index_basic[i][j]))
                    throw std::runtime_error("PairMTP: not enough values in alpha_index_basic");
            if (alpha_index_basic[i][0] < 0 || alpha_index_basic[i][0] >= radial_func_count)
                throw std::runtime_error("PairMTP: radial function index out of range in alpha_index_basic");
            if (alpha_index_basic[i][1] < 0 || alpha_index_basic[i][2] < 0 || alpha_index_basic[i][3] < 0)
                throw std::runtime_error("PairMTP: negative angular exponent in alpha_index_basic");
            radial_func_max = std::max(radial_func_max, alpha_index_basic[i][0]);
        }
        if (radial_func_max != radial_func_count - 1)    //Index validity check
            throw std::runtime_error("PairMTP: wrong number of radial functions specified");
    }

    alpha_index_times_count = tfr.next_int("alpha_index_times_count");
    if (alpha_index_times_count < 0)
        throw std::runtime_error("PairMTP: MTP alpha index times count is negative");

    tfr.expect("alpha_index_times");
    alpha_index_times.resize(alpha_index_times_count);
    {
        std::istringstream ss = tfr.rest();
        for (int i = 0; i < alpha_index_times_count; i++) {
            for (int j = 0; j < 4; j++)
                if (!(ss >> alpha_index_times[i][j]))
                    throw std::runtime_error("PairMTP: not enough values in alpha_index_times");
            for (int j = 0; j < 4; j++)
                if (j != 2 && (alpha_index_times[i][j] < 0 || alpha_index_times[i][j] >= alpha_moment_count))
                    throw std::runtime_error("PairMTP: moment index out of range in alpha_index_times");
        }
    }

    alpha_scalar_count = tfr.next_int("alpha_scalar_moments");
    if (alpha_scalar_count < 0)
        throw std::runtime_error("PairMTP: MTP alpha scalar moment count is negative");

    tfr.expect("alpha_moment_mapping");
    alpha_moment_mapping.resize(alpha_scalar_count);
    {
        std::istringstream ss = tfr.rest();
        for (int i = 0; i < alpha_scalar_count; i++) {
            if (!(ss >> alpha_moment_mapping[i]))
                throw std::runtime_error("PairMTP: not enough values in alpha_moment_mapping");
            if (alpha_moment_mapping[i] < 0 || alpha_moment_mapping[i] >= alpha_moment_count)
                throw std::runtime_error("PairMTP: moment index out of range in alpha_moment_mapping");
        }
    }

    // An untrained potential may omit the species and moment coeffs too
    const bool has_species_coeffs = tfr.next_line() && tfr.keyword() == "species_coeffs";
    std::vector<double> species_coeffs_read(has_species_coeffs ? species_count : 0);
    std::vector<double> linear_coeffs_read(has_species_coeffs ? alpha_scalar_count : 0);
    if (has_species_coeffs) {
        std::istringstream species_ss = tfr.rest();
        for (int i = 0; i < species_count; i++)
            if (!(species_ss >> species_coeffs_read[i]))
                throw std::runtime_error("PairMTP: not enough species coefficients");

        tfr.expect("moment_coeffs");
        std::istringstream ss = tfr.rest();
        for (int i = 0; i < alpha_scalar_count; i++)
            if (!(ss >> linear_coeffs_read[i]))
                throw std::runtime_error("PairMTP: not enough moment coefficients");
    }

    prepare_basis();

    // mlip-3's MLMTPR::Load fills in what the file omits
    trained = has_radial_coeffs && has_species_coeffs;
    if (!trained) reset_coeffs();
    if (has_radial_coeffs) radial_basis_coeffs = radial_coeffs_read;
    if (has_species_coeffs) {
        species_coeffs = species_coeffs_read;
        linear_coeffs = linear_coeffs_read;
    }
}

PairMTP::PairMTP(const std::vector<std::vector<int>>& basis_functions, int species_count, const std::string& radial_basis_type, double min_dist, double max_dist, int radial_basis_size) : Pair(species_count, radial_basis_type, min_dist, max_dist, radial_basis_size) {
    MTP_Basis basis(basis_functions);
    radial_func_count = basis.radial_function_count();
    alpha_moment_count = basis.alpha_moment_count;
    alpha_index_basic = std::move(basis.alpha_index_basic);
    alpha_index_times = std::move(basis.alpha_index_times);
    alpha_moment_mapping = std::move(basis.alpha_moment_mapping);
    alpha_index_basic_count = (int) alpha_index_basic.size();
    alpha_index_times_count = (int) alpha_index_times.size();
    alpha_scalar_count = (int) alpha_moment_mapping.size();

    prepare_basis();
    trained = false;
    reset_coeffs();
}

void PairMTP::reset_coeffs() {
    for (int p = 0; p < species_count * species_count; p++)
        for (int i = 0; i < radial_func_count; i++) {
            double* c = radial_basis_coeffs.data() + p * radial_coeff_count_per_pair + i * radial_basis_size;
            for (int j = 0; j < radial_basis_size; j++)
                c[j] = 1e-6 + i * 1e-7 + j * 1e-7;
            c[std::min(i, radial_basis_size)] = 1e-3 * (p + 1) + i * 1e-4;
        }
    species_coeffs.assign(species_count, 1e-3);
    linear_coeffs.assign(alpha_scalar_count, 1e-3);
}

void PairMTP::prepare_basis() {
    cache_stride = 1 + 2 * radial_func_count;
    radial_coeff_count_per_pair = radial_basis_size * radial_func_count;
    radial_coeff_count = species_count * species_count * radial_coeff_count_per_pair;
    radial_basis_coeffs.resize(radial_coeff_count);
    species_coeffs.resize(species_count);
    linear_coeffs.resize(alpha_scalar_count);
    moment_tensor_vals.resize(alpha_moment_count);
    nbh_energy_ders_wrt_moments.resize(alpha_moment_count);
    atom_moment_vals.resize(alpha_moment_count);
    atom_energy_ders_wrt_moments.resize(alpha_moment_count);

    //Find the maximum alpha basic index
    max_alpha_index_basic = 0;
    for (int i = 0; i < alpha_index_basic_count; i++)
        max_alpha_index_basic = std::max(max_alpha_index_basic, alpha_index_basic[i][1] + alpha_index_basic[i][2] + alpha_index_basic[i][3]);
    max_alpha_index_basic++;    // Add 1 to account for zeroth order indicies

    // Set working buffers
    prepare_angular();
    mu_offsets.assign(radial_func_count + 1, 0);
    basic_by_mu.resize(alpha_index_basic_count);
    angular_by_mu.resize(alpha_index_basic_count);
    basic_ders_by_mu.resize(alpha_index_basic_count);
    for (int k = 0; k < alpha_index_basic_count; k++) mu_offsets[alpha_index_basic[k][0] + 1]++;
    for (int mu = 0; mu < radial_func_count; mu++) mu_offsets[mu + 1] += mu_offsets[mu];
    std::vector<int> radial_next(mu_offsets.begin(), mu_offsets.begin() + radial_func_count);
    for (int k = 0; k < alpha_index_basic_count; k++)
        basic_by_mu[radial_next[alpha_index_basic[k][0]]++] = k;
    for (int t = 0; t < alpha_index_basic_count; t++)
        angular_by_mu[t] = basic_to_angular[basic_by_mu[t]];

    // Check the contraction graph: basic moments own [0, alpha_index_basic_count)
    // and no term may read a moment that has not been produced yet.
    std::vector<char> produced(alpha_moment_count, 0);
    std::fill(produced.begin(), produced.begin() + alpha_index_basic_count, 1);
    for (int k = 0; k < alpha_index_times_count; k++) {
        if (alpha_index_times[k][3] < alpha_index_basic_count)
            throw std::runtime_error("PairMTP: MTP contraction " + std::to_string(k) + " overwrites a basic moment");
        if (!produced[alpha_index_times[k][0]] || !produced[alpha_index_times[k][1]])
            throw std::runtime_error("PairMTP: MTP contraction " + std::to_string(k) + " reads a moment that is not yet computed");
        produced[alpha_index_times[k][3]] = 1;
    }
}

/* ----------------------------------------------------------------------
   Share angular monomials across radial channels. Lexicographic exponent
   order puts every parent before its child.
------------------------------------------------------------------------- */
void PairMTP::prepare_angular() {
    using Powers = std::array<int, 3>;
    std::map<Powers, int> indices;
    indices.emplace(Powers{{0, 0, 0}}, 0);

    for (int k = 0; k < alpha_index_basic_count; k++) {
        Powers powers{{alpha_index_basic[k][1], alpha_index_basic[k][2], alpha_index_basic[k][3]}};
        if (powers[0] < 0 || powers[1] < 0 || powers[2] < 0)
            throw std::runtime_error("PairMTP: invalid negative MTP angular exponent");
        while (indices.emplace(powers, 0).second) {
            const int axis = powers[2] ? 2 : (powers[1] ? 1 : 0);
            --powers[axis];
        }
    }

    angular_count = 0;
    for (auto& entry : indices) entry.second = angular_count++;
    basic_to_angular.resize(alpha_index_basic_count);
    angular_parent.resize(angular_count);
    angular_axis.resize(angular_count);
    angular_vals.resize(angular_count);
    angular_ders.resize(angular_count);
    angular_parent[0] = angular_axis[0] = 0;
    angular_vals[0] = 1.0;

    for (const auto& entry : indices) {
        const int index = entry.second;
        if (index == 0) continue;
        Powers parent = entry.first;
        const int axis = parent[2] ? 2 : (parent[1] ? 1 : 0);
        --parent[axis];
        angular_parent[index] = indices.at(parent);
        angular_axis[index] = axis;
    }
    for (int k = 0; k < alpha_index_basic_count; k++)
        basic_to_angular[k] = indices.at(
            Powers{{alpha_index_basic[k][1], alpha_index_basic[k][2], alpha_index_basic[k][3]}});
}
