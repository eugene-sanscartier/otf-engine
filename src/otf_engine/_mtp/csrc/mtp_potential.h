/* -*- c++ -*- ----------------------------------------------------------
   Standalone MTP potential — no LAMMPS dependency.
   Ported from lammps-mtp/src/ML-MTP/pair_mtp.h
   Original author: Richard Meng, Queen's University at Kingston, 22.11.24
------------------------------------------------------------------------- */

#pragma once

#include "mtp_radial_basis.h"
#include "neigh_list.h"

#include <algorithm>
#include <array>
#include <istream>
#include <string>
#include <vector>

class PairMTP {
  public:
    // Reads an MTP potential file (version 1.1.0).
    explicit PairMTP(const std::string& filename);
    virtual ~PairMTP();

    // Energy, forces and virial for every central atom in list.
    // forces  : [n_atoms * 3], accumulated (zeroed by the caller)
    // virial  : [6] as xx,yy,zz,xy,xz,yz — nullptr to skip
    // eatom   : [n_atoms] — nullptr to skip
    // Returns the total energy.
    double compute(const NeighList& list, double* forces, double* virial, double* eatom);

    // Scalar moment values per central atom: [inum * alpha_scalar_count].
    // Row ii holds the basis values for ilist[ii]; dot with linear_coeffs and
    // add species_coeffs for the site energy.
    void eval_basis(const NeighList& list, double* basis_out);

    // Chebyshev radial basis at one distance, into arrays of radial_basis_size.
    // Pass nullptr for ders_out to skip the derivatives.
    void eval_radial_basis(double dist, double* vals_out, double* ders_out = nullptr);

    int get_species_count() const { return species_count; }
    int get_radial_func_count() const { return radial_func_count; }
    int get_radial_basis_size() const { return radial_basis_size; }
    int get_radial_coeff_count() const { return radial_coeff_count; }
    int get_alpha_scalar_count() const { return alpha_scalar_count; }
    int get_alpha_moment_count() const { return alpha_moment_count; }
    int get_alpha_index_basic_count() const { return alpha_index_basic_count; }
    int get_alpha_index_times_count() const { return alpha_index_times_count; }
    double get_min_cutoff() const { return min_cutoff; }
    double get_max_cutoff() const { return max_cutoff; }
    double get_scaling() const { return scaling; }
    const std::string& get_potential_name() const { return potential_name; }

    // radial_basis_coeffs: flat [species^2 * radial_func_count * radial_basis_size]
    const double* get_radial_basis_coeffs() const { return radial_basis_coeffs.data(); }
    // alpha_index_basic: flat [alpha_index_basic_count * 4]  (mu, px, py, pz)
    const int* get_alpha_index_basic() const { return alpha_index_basic[0].data(); }
    // alpha_index_times: flat [alpha_index_times_count * 4]  (i0, i1, mult, out)
    const int* get_alpha_index_times() const { return alpha_index_times[0].data(); }
    const int* get_alpha_moment_mapping() const { return alpha_moment_mapping.data(); }
    const double* get_linear_coeffs() const { return linear_coeffs.data(); }
    const double* get_species_coeffs() const { return species_coeffs.data(); }

    void set_linear_coeffs(const double* c) { std::copy(c, c + alpha_scalar_count, linear_coeffs.begin()); }
    void set_species_coeffs(const double* c) { std::copy(c, c + species_count, species_coeffs.begin()); }
    void set_radial_basis_coeffs(const double* c) { std::copy(c, c + radial_coeff_count, radial_basis_coeffs.begin()); }
    void set_scaling(double s) { scaling = s; radial_basis->scaling = s; }
    // Mirrors mlip-3 AddSpecies(): min_val = 0.99 * min(training distances).
    void set_min_cutoff(double d) { min_cutoff = d; radial_basis->min_cutoff = d; }

  protected:
    void read_file(std::istream& is);

    std::string potential_name = "Untitled";
    std::string potential_tag;

    int species_count = 0;
    double scaling = 1.0;

    // Radial basis
    RadialMTPBasis* radial_basis = nullptr;
    int radial_func_count = 0;           // Number of radial bases (mu_max)
    int radial_basis_size = 0;           // Number of elements in bases
    int radial_coeff_count = 0;          // Number of total radial coeffs
    int radial_coeff_count_per_pair = 0; // Number of coeffs for species pair

    // The MTP only supports one cutoff set for all species combinations
    double min_cutoff = 0.0;
    double max_cutoff = 0.0;
    double max_cutoff_sq = 0.0;

    std::vector<double> radial_basis_coeffs; // radial basis coeffs (c)
    std::vector<double> linear_coeffs;       // moment tensor basis coeffs (xi)
    std::vector<double> species_coeffs;      // 0th rank moment tensor coeffs

    int alpha_moment_count = 0;
    int alpha_index_basic_count = 0;
    int alpha_index_times_count = 0;
    int alpha_scalar_count = 0;
    int max_alpha_index_basic = 0;
    std::vector<std::array<int, 4>> alpha_index_basic; // builds elementary moments from coords and dist
    std::vector<std::array<int, 4>> alpha_index_times; // combines existing moments into new ones
    std::vector<int> alpha_moment_mapping;             // selects basis values from completed moments

    // Working buffers
    int jac_size = 0;                                // Size of the jacobian (jnum dim)
    std::vector<double> dist_powers;                 // powers of dist (eg. d^i)
    std::vector<std::array<double, 3>> coord_powers; // powers of rel. pos. (eg. [dx^i, dy^i, dz^i])
    std::vector<double> radial_vals;                 // radial basis values for each mu
    std::vector<double> radial_ders;                 // radial basis derivatives for each mu
    std::vector<double> moment_tensor_vals;          // the moments
    std::vector<double> nbh_energy_ders_wrt_moments; // same, for ders
    std::vector<std::array<double, 3>> moment_jacobian; // [jac_size * alpha_index_basic_count]
    std::vector<int> valid_j;                           // [jac_size]
    std::vector<std::array<double, 3>> valid_dr;        // [jac_size] displacement of each valid neighbor
};
