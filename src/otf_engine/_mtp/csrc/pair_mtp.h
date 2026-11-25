/* -*- c++ -*- ----------------------------------------------------------
   Standalone MTP potential — no LAMMPS dependency.
   Ported from lammps-mtp/src/ML-MTP/pair_mtp.h
   Original author: Richard Meng, Queen's University at Kingston, 22.11.24
------------------------------------------------------------------------- */

#pragma once

#include "pair.h"

#include <algorithm>
#include <array>
#include <istream>
#include <string>
#include <vector>

// Pair holds the members PairMTP shares with PairETN.
class PairMTP : public Pair {
  public:
    // Reads an MTP potential file (version 1.1.0).
    explicit PairMTP(const std::string& filename);
    // Builds the untrained MTP of these basis functions (see MTP_Basis) on a radial basis, with the defaults of an untrained file.
    // species_count 0 leaves the count to training.
    PairMTP(const std::vector<std::vector<int>>& basis_functions, int species_count, const std::string& radial_basis_type, double min_dist, double max_dist, int radial_basis_size);

    // Computes the energy, per-atom energies, forces and virial of the central atoms in list into members.
    void compute(const NeighList& list) override;

    // Length of the coefficient vector [radial | species | linear].
    int coeff_count() const override { return radial_coeff_count + species_count + alpha_scalar_count; }
    // Copies the coefficients into one vector c = [c_radial | c_species | beta_linear].
    void get_coeffs(double* c) const override;
    void set_coeffs(const double* c) override;

    int get_radial_func_count() const { return radial_func_count; }
    int get_radial_coeff_count() const { return radial_coeff_count; }
    int get_alpha_scalar_count() const { return alpha_scalar_count; }
    int get_alpha_moment_count() const { return alpha_moment_count; }
    int get_alpha_index_basic_count() const { return alpha_index_basic_count; }
    int get_alpha_index_times_count() const { return alpha_index_times_count; }

    // [species^2 * radial_func_count * radial_basis_size]
    const std::vector<double>& get_radial_basis_coeffs() const { return radial_basis_coeffs; }
    // [alpha_index_basic_count] (mu, px, py, pz)
    const std::vector<std::array<int, 4>>& get_alpha_index_basic() const { return alpha_index_basic; }
    // [alpha_index_times_count] (i0, i1, mult, out)
    const std::vector<std::array<int, 4>>& get_alpha_index_times() const { return alpha_index_times; }
    const std::vector<int>& get_alpha_moment_mapping() const { return alpha_moment_mapping; }    // [alpha_scalar_count]
    const std::vector<double>& get_linear_coeffs() const { return linear_coeffs; }               // [alpha_scalar_count]

    void set_linear_coeffs(const double* c) { std::copy(c, c + alpha_scalar_count, linear_coeffs.begin()); }
    void set_radial_basis_coeffs(const double* c) { std::copy(c, c + radial_coeff_count, radial_basis_coeffs.begin()); }
    // Sizes the coefficients for n species. The remaining species keep their radial coefficients and site energies.
    // The new entries are zero, and the caller sets them.
    void set_species_count(int n);

  protected:
    void read_file(std::istream& is);
    
    // Sizes the coefficients and working buffers for the basis members, and checks the contractions.
    void prepare_basis();
    void prepare_angular();

    void reset_coeffs();    // Sets the coefficients of an untrained potential to their defaults.

    // Radial basis
    int radial_func_count = 0;           // number of radial functions (mu_max)
    int radial_coeff_count = 0;
    int radial_coeff_count_per_pair = 0;

    std::vector<double> radial_basis_coeffs;
    std::vector<double> linear_coeffs;       // moment tensor basis coeffs (xi)

    int alpha_moment_count = 0;
    int alpha_index_basic_count = 0;
    int alpha_index_times_count = 0;
    int alpha_scalar_count = 0;
    int max_alpha_index_basic = 0;
    std::vector<std::array<int, 4>> alpha_index_basic; // builds elementary moments from coords and dist
    std::vector<std::array<int, 4>> alpha_index_times; // combines existing moments into new ones
    std::vector<int> alpha_moment_mapping;             // selects basis values from completed moments

    // Shared angular monomials and per-neighbor scratch.
    int angular_count = 0;
    std::vector<int> basic_to_angular;       // [alpha_index_basic_count] monomial of each basic moment
    std::vector<int> basic_by_mu;            // [alpha_index_basic_count] basic moments grouped by radial function
    std::vector<int> angular_by_mu;          // [alpha_index_basic_count] their monomials
    std::vector<int> mu_offsets;             // [radial_func_count + 1] bounds of each group
    std::vector<int> angular_parent;         // [angular_count] monomial one power lower
    std::vector<int> angular_axis;           // [angular_count] axis of that power
    std::vector<double> angular_vals;        // [angular_count] monomials of the unit displacement
    std::vector<double> angular_ders;        // [angular_count] their adjoints
    std::vector<double> basic_ders_by_mu;    // [alpha_index_basic_count] energy derivatives with respect to the basic moments, in basic_by_mu order

    // Forward and backward pass over the moment graph for the chunk of compute, one value per atom
    std::vector<Lanes> moment_tensor_vals;
    std::vector<Lanes> nbh_energy_ders_wrt_moments; // energy derivatives with respect to the moments

    // The same for the kernels that take one atom at a time
    std::vector<double> atom_moment_vals;              // [alpha_moment_count]
    std::vector<double> atom_energy_ders_wrt_moments;  // [alpha_moment_count]

    // neighbor_cache holds 1/dist, the radial function values and their derivatives per neighbor, with cache_stride 1 + 2 * radial_func_count
};
