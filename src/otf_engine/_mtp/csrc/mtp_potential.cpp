/* -*- c++ -*- ----------------------------------------------------------
   Standalone MTP potential — no LAMMPS dependency.
   Ported from lammps-mtp/src/ML-MTP/pair_mtp.cpp
   Original author: Richard Meng, Queen's University at Kingston, 22.11.24
------------------------------------------------------------------------- */

#include "mtp_potential.h"

#include "text_file_reader.h"

#include <cmath>
#include <fstream>
#include <stdexcept>

PairMTP::PairMTP(const std::string& filename) {
    std::ifstream f(filename);
    if (!f.is_open())
        throw std::runtime_error("PairMTP: cannot open file '" + filename + "'");
    PairMTP::read_file(f);
}

PairMTP::~PairMTP() {
    delete radial_basis;
}

/* ----------------------------------------------------------------------
   Straightfoward MTP implementation based on MLIP3
   ---------------------------------------------------------------------- */
double PairMTP::compute(const NeighList& list, double* forces, double* virial, double* eatom) {
    double total_energy = 0.0;
    int nbr_offset = 0;

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

        // Clear moment and derivative arrays
        std::fill(moment_tensor_vals.begin(), moment_tensor_vals.end(), 0.0);
        std::fill(nbh_energy_ders_wrt_moments.begin(), nbh_energy_ders_wrt_moments.end(), 0.0);

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

            // Precompute the coord and distance powers
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
                val *= norm_fac;
                der = der * norm_fac - norm_rank * val / dist;
                double pow0 = coord_powers[alpha_index_basic[k][1]][0];
                double pow1 = coord_powers[alpha_index_basic[k][2]][1];
                double pow2 = coord_powers[alpha_index_basic[k][3]][2];
                double pow = pow0 * pow1 * pow2;
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
        for (int k = 0; k < alpha_scalar_count; k++)
            nbh_energy += linear_coeffs[k] * moment_tensor_vals[alpha_moment_mapping[k]];

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

            // Accumulate virial stress only if requested
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

    return total_energy;
}

/* ----------------------------------------------------------------------
   Basis values per central atom, before the linear coefficients are applied
------------------------------------------------------------------------- */
void PairMTP::eval_basis(const NeighList& list, double* basis_out) {
    int nbr_offset = 0;

    for (int ii = 0; ii < list.inum; ii++) {
        int valid_count = 0;
        const int i = list.ilist[ii];
        const int itype = list.types[i];
        const int jnum = list.numneigh[ii];
        const int* nbrs = list.firstneigh + nbr_offset;
        const double* dr = list.displacements + nbr_offset * 3;
        nbr_offset += jnum;

        if (jac_size < jnum) {
            jac_size = jnum;
            moment_jacobian.resize((size_t) jac_size * alpha_index_basic_count);
            valid_j.resize(jac_size);
            valid_dr.resize(jac_size);
        }

        std::fill(moment_tensor_vals.begin(), moment_tensor_vals.end(), 0.0);

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
                int mu = alpha_index_basic[k][0];

                double val = radial_vals[mu];
                double der = radial_ders[mu];

                int norm_rank = alpha_index_basic[k][1] + alpha_index_basic[k][2] + alpha_index_basic[k][3];
                double norm_fac = 1.0 / dist_powers[norm_rank];
                val *= norm_fac;
                der = der * norm_fac - norm_rank * val / dist;
                double pow0 = coord_powers[alpha_index_basic[k][1]][0];
                double pow1 = coord_powers[alpha_index_basic[k][2]][1];
                double pow2 = coord_powers[alpha_index_basic[k][3]][2];
                double pow = pow0 * pow1 * pow2;
                moment_tensor_vals[k] += val * pow;

                const size_t jac = (size_t) valid_count * alpha_index_basic_count + k;
                pow *= der / dist;
                moment_jacobian[jac][0] = pow * r[0];
                moment_jacobian[jac][1] = pow * r[1];
                moment_jacobian[jac][2] = pow * r[2];
                if (alpha_index_basic[k][1] != 0) {
                    moment_jacobian[jac][0] += val * alpha_index_basic[k][1] *
                        coord_powers[alpha_index_basic[k][1] - 1][0] * pow1 * pow2;
                }
                if (alpha_index_basic[k][2] != 0) {
                    moment_jacobian[jac][1] += val * alpha_index_basic[k][2] * pow0 *
                        coord_powers[alpha_index_basic[k][2] - 1][1] * pow2;
                }
                if (alpha_index_basic[k][3] != 0) {
                    moment_jacobian[jac][2] += val * alpha_index_basic[k][3] * pow0 * pow1 *
                        coord_powers[alpha_index_basic[k][3] - 1][2];
                }
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

        double* row = basis_out + (size_t) ii * alpha_scalar_count;
        for (int k = 0; k < alpha_scalar_count; k++)
            row[k] = moment_tensor_vals[alpha_moment_mapping[k]];
    }
}

/* ---------------------------------------------------------------------- */
void PairMTP::eval_radial_basis(double dist, double* vals_out, double* ders_out) {
    if (ders_out)
        radial_basis->calc_radial_basis_ders(dist);
    else
        radial_basis->calc_radial_basis(dist);

    std::copy(radial_basis->radial_basis_vals.begin(), radial_basis->radial_basis_vals.end(), vals_out);
    if (ders_out)
        std::copy(radial_basis->radial_basis_ders.begin(), radial_basis->radial_basis_ders.end(), ders_out);
}

/* ----------------------------------------------------------------------
   MTP file parsing. Includes allocation. Excludes the radial basis
   hyperparameters, which the radial basis constructor reads.
------------------------------------------------------------------------- */
void PairMTP::read_file(std::istream& is) {
    TextFileReader tfr(is);

    tfr.expect("MTP");
    if (tfr.next_string("version") != "1.1.0")
        throw std::runtime_error("PairMTP: MTP file must have version \"1.1.0\"");

    tfr.advance();

    // Read the potential name (optional field)
    if (tfr.keyword() == "potential_name") {
        tfr.rest() >> potential_name;
        tfr.advance();
    }

    // Check the scaling
    if (tfr.keyword() == "scaling") {
        tfr.rest() >> scaling;
        tfr.advance();
    }

    if (tfr.keyword() != "species_count")
        throw std::runtime_error("PairMTP: species count not found");
    tfr.rest() >> species_count;

    tfr.advance();

    // Read the potential tag (also optional field)
    if (tfr.keyword() == "potential_tag") {
        tfr.rest() >> potential_tag;
        tfr.advance();
    }

    if (tfr.keyword() != "radial_basis_type")
        throw std::runtime_error("PairMTP: no radial basis set type is specified");
    tfr.rest() >> radial_basis_type;

    // Set the type of radial basis. Only RBChebyshev is in the reference; the
    // rest are transcribed from mlip-3 to widen which files load.
    if (radial_basis_type == "RBChebyshev")
        radial_basis = new RBChebyshev(tfr);
    else if (radial_basis_type == "RBChebyshev_repuls")
        radial_basis = new RBChebyshevRepuls(tfr);
    else if (radial_basis_type == "BChebyshev")
        radial_basis = new BChebyshev(tfr);
    else if (radial_basis_type == "BChebyshev_repuls")
        radial_basis = new BChebyshevRepuls(tfr);
    else if (radial_basis_type == "RBTaylor")
        radial_basis = new RBTaylor(tfr);
    else if (radial_basis_type == "RBShapeev")
        throw std::runtime_error("PairMTP: RBShapeev is not implemented");
    else
        throw std::runtime_error("PairMTP: unknown radial basis type '" + radial_basis_type + "'");

    radial_basis->scaling = scaling;
    radial_basis_size = radial_basis->size;
    min_cutoff = radial_basis->min_cutoff;
    max_cutoff = radial_basis->max_cutoff;
    max_cutoff_sq = max_cutoff * max_cutoff;

    radial_func_count = tfr.next_int("radial_funcs_count");

    tfr.advance();
    if (tfr.keyword() == "magnetic_basis_type")
        throw std::runtime_error("PairMTP: magnetic basis is currently not supported");
    if (tfr.keyword() != "radial_coeffs")
        throw std::runtime_error("PairMTP: cannot read radial coeffs");

    // Allocate memory for radial basis
    const int pairs_count = species_count * species_count;
    radial_coeff_count_per_pair = radial_basis_size * radial_func_count;
    radial_coeff_count = pairs_count * radial_coeff_count_per_pair;
    radial_basis_coeffs.resize(radial_coeff_count);

    // Read the radial basis coeffs
    for (int i = 0; i < pairs_count; i++) {
        // pair-type label "0-1"; '-' is a separator only on this line, since
        // the coefficient lines below carry negative numbers
        if (!tfr.next_line("-"))
            throw std::runtime_error("PairMTP: unexpected end of file in radial_coeffs");
        int type1, type2;
        std::istringstream label(tfr.line());
        if (!(label >> type1 >> type2))
            throw std::runtime_error("PairMTP: cannot read radial_coeffs pair label");

        const int pair_offset = (type1 * species_count + type2) * radial_coeff_count_per_pair;
        for (int j = 0; j < radial_func_count; j++) {
            tfr.advance();
            // The whole line is coefficients — there is no leading keyword.
            std::istringstream ss(tfr.line());
            for (int k = 0; k < radial_basis_size; k++)
                if (!(ss >> radial_basis_coeffs[pair_offset + j * radial_basis_size + k]))
                    throw std::runtime_error("PairMTP: not enough radial coefficients");
        }
    }

    alpha_moment_count = tfr.next_int("alpha_moments_count");
    moment_tensor_vals.resize(alpha_moment_count);
    nbh_energy_ders_wrt_moments.resize(alpha_moment_count);

    alpha_index_basic_count = tfr.next_int("alpha_index_basic_count");

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
            radial_func_max = std::max(radial_func_max, alpha_index_basic[i][0]);
        }
        if (radial_func_max != radial_func_count - 1)    //Index validity check
            throw std::runtime_error("PairMTP: wrong number of radial functions specified");
    }

    //Find the maximum alpha basic index
    max_alpha_index_basic = 0;
    for (int i = 0; i < alpha_index_basic_count; i++)
        max_alpha_index_basic = std::max(max_alpha_index_basic, alpha_index_basic[i][1] + alpha_index_basic[i][2] + alpha_index_basic[i][3]);
    max_alpha_index_basic++;    // Add 1 to account for zeroth order indicies

    alpha_index_times_count = tfr.next_int("alpha_index_times_count");

    tfr.expect("alpha_index_times");
    alpha_index_times.resize(alpha_index_times_count);
    {
        std::istringstream ss = tfr.rest();
        for (int i = 0; i < alpha_index_times_count; i++)
            for (int j = 0; j < 4; j++)
                if (!(ss >> alpha_index_times[i][j]))
                    throw std::runtime_error("PairMTP: not enough values in alpha_index_times");
    }

    alpha_scalar_count = tfr.next_int("alpha_scalar_moments");

    tfr.expect("alpha_moment_mapping");
    alpha_moment_mapping.resize(alpha_scalar_count);
    {
        std::istringstream ss = tfr.rest();
        for (int i = 0; i < alpha_scalar_count; i++)
            if (!(ss >> alpha_moment_mapping[i]))
                throw std::runtime_error("PairMTP: not enough values in alpha_moment_mapping");
    }

    tfr.expect("species_coeffs");
    species_coeffs.resize(species_count);
    {
        std::istringstream ss = tfr.rest();
        for (int i = 0; i < species_count; i++)
            if (!(ss >> species_coeffs[i]))
                throw std::runtime_error("PairMTP: not enough species coefficients");
    }

    tfr.expect("moment_coeffs");
    linear_coeffs.resize(alpha_scalar_count);
    {
        std::istringstream ss = tfr.rest();
        for (int i = 0; i < alpha_scalar_count; i++)
            if (!(ss >> linear_coeffs[i]))
                throw std::runtime_error("PairMTP: not enough moment coefficients");
    }

    //Working buffers
    dist_powers.resize(max_alpha_index_basic);
    coord_powers.resize(max_alpha_index_basic);
    radial_vals.resize(radial_func_count);
    radial_ders.resize(radial_func_count);

    // Set working buffers
    dist_powers[0] = coord_powers[0][0] = coord_powers[0][1] = coord_powers[0][2] = 1;
}
