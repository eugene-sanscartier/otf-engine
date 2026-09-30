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

    // Energy, per-atom energies, forces and virial of the central atoms in list,
    // left in the members below.
    void compute(const NeighList& list);

    double get_energy() const { return energy; }
    const double* get_eatom() const { return eatom.data(); }      // [n_atoms]
    const double* get_forces() const { return forces.data(); }    // [n_atoms * 3]
    const double* get_virial() const { return virial; }           // [6] xx,yy,zz,xy,xz,yz

    // Length of the coefficient vector [radial | species | linear].
    int coeff_count() const { return radial_coeff_count + species_count + alpha_scalar_count; }
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
    // False for a file without coefficients, which holds mlip-3's defaults.
    bool is_trained() const { return trained; }
    const std::string& get_potential_name() const { return potential_name; }
    const std::string& get_radial_basis_type() const { return radial_basis_type; }

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
    void set_trained(bool t) { trained = t; }
    // Sizes the coefficients for n species; set them afterwards.
    void set_species_count(int n) {
        species_count = n;
        radial_coeff_count = n * n * radial_coeff_count_per_pair;
        radial_basis_coeffs.resize(radial_coeff_count);
        species_coeffs.resize(n);
    }

  protected:
    void read_file(std::istream& is);
    void prepare_angular();

    std::string potential_name = "Untitled";
    std::string potential_tag;
    std::string radial_basis_type;

    int species_count = 0;
    double scaling = 1.0;
    bool trained = true;

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
    std::vector<double> basic_ders_by_mu;    // [alpha_index_basic_count] energy ders wrt basic moments, grouped

    // Graph traversal, forwards and backwards pass
    std::vector<double> moment_tensor_vals;          // the moments
    std::vector<double> nbh_energy_ders_wrt_moments; // same, for ders

    // Results of compute
    double energy = 0.0;
    std::vector<double> eatom;   // [n_atoms] site energies
    std::vector<double> forces;  // [n_atoms * 3]
    double virial[6] = {};       // xx,yy,zz,xy,xz,yz

    // Cache values between forwards and backwards pass
    int cache_size = 0;
    std::vector<int> cached_j;                    // [cache_size]
    std::vector<std::array<double, 3>> valid_dr;  // [cache_size] displacement of each cached neighbor
    std::vector<double> neighbor_cache;           // [cache_size * (1 + 2 * radial_func_count)] 1/dist, radial vals, radial ders
};
