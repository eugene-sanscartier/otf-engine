/* -*- c++ -*- ----------------------------------------------------------
   Standalone MTP potential — no LAMMPS dependency.
   Refactored from lammps-mtp/src/ML-MTP/pair_mtp.h
   Original author: Richard Meng, Queen's University at Kingston, 22.11.24
------------------------------------------------------------------------- */

#pragma once

#include "mtp_radial_basis.h"

#include <array>
#include <string>
#include <vector>

void read_file(const std::string& filename, class MTPPotential& potential);

class MTPPotential {
  public:
    // Read an MTP potential file (version 1.1.0, RBChebyshev basis).
    explicit MTPPotential(const std::string& filename);
    ~MTPPotential();

    // -----------------------------------------------------------------------
    // Core computation
    //
    // Accepts a pre-built *full* (both-directions) neighbor list with
    // pre-computed displacement vectors (PBC-corrected).
    //
    //   n_atoms       : total number of atoms
    //   types         : [n_atoms] 0-indexed MTP species
    //   inum          : number of central atoms (usually == n_atoms)
    //   ilist         : [inum] indices of central atoms
    //   numneigh      : [inum] neighbor count for each central atom
    //   firstneigh    : flat [sum(numneigh)] original neighbor atom indices;
    //                   consecutive blocks correspond to ilist[0], ilist[1], …
    //   displacements : flat [sum(numneigh) * 3] displacement vectors
    //                   r_j - r_i (including PBC shift), one per neighbor pair
    //   forces_out    : [n_atoms * 3] accumulated forces (zeroed by caller)
    //   virials_out   : [6] virial components (xx,yy,zz,xy,xz,yz), zeroed by caller
    //   eatom_out     : [n_atoms] per-atom energies (nullptr to skip)
    //
    // Returns the total potential energy.
    // -----------------------------------------------------------------------
    double compute(int n_atoms, const int* types, int inum, const int* ilist, const int* numneigh, const int* firstneigh, const double* displacements, double* forces_out, double* virials_out, double* eatom_out = nullptr);

    // -----------------------------------------------------------------------
    // Basis function evaluation
    //
    // Fills basis_out[inum * alpha_scalar_count] with the scalar moment tensor
    // values for each central atom (before multiplying by linear_coeffs).
    // Row ii corresponds to ilist[ii].
    // Same neighbor list + displacement convention as compute().
    // -----------------------------------------------------------------------
    void eval_basis(int n_atoms, const int* types, int inum, const int* ilist, const int* numneigh, const int* firstneigh, const double* displacements, double* basis_out);

    // -----------------------------------------------------------------------
    // Full per-atom information vector (grade computation)
    //
    // Fills grad_out[inum * coeff_count()] with the gradient of site energy
    // w.r.t. ALL MTP coefficients for each central atom.
    //
    // Layout of each row (length coeff_count()):
    //   [0 .. radial_coeff_count)               : dE/d(radial_basis_coeffs)
    //   [radial_coeff_count + itype]             : 1.0  (species one-hot)
    //   [radial_coeff_count + species_count + k] : moment_tensor_vals[alpha_moment_mapping[k]]
    //
    // coeff_count() = radial_coeff_count + species_count + alpha_scalar_count
    // -----------------------------------------------------------------------
    void eval_grad(int n_atoms, const int* types, int inum, const int* ilist, const int* numneigh, const int* firstneigh, const double* displacements, double* grad_out);

    // -----------------------------------------------------------------------
    // Fused energy/force/virial + all radial gradients in a single pass.
    // Preferred entry point for the training outer-loop gradient.
    //
    // Returns total potential energy.
    //
    // forces_out      : [n_atoms * 3]
    // virials_out     : [6]                    (nullptr to skip)
    // energy_grad_out : [radial_coeff_count]   — ∂E_total/∂c_r
    // force_grad_out  : [n_atoms * 3 * radial_coeff_count]
    // virial_grad_out : [6 * radial_coeff_count] (nullptr to skip)
    // -----------------------------------------------------------------------
    double compute_with_radial_grad(int n_atoms, const int* types, int inum, const int* ilist, const int* numneigh, const int* firstneigh, const double* displacements, double* forces_out, double* virials_out, double* energy_grad_out, double* force_grad_out, double* virial_grad_out = nullptr);

    // -----------------------------------------------------------------------
    // Three pure-gradient functions (no EFS values computed).
    //
    // eval_grad_radial  — ∂(E_total, F_i, virial) / ∂c_radial
    //   energy_grad_out : [radial_coeff_count]           — ∂E_total/∂c_r (summed over atoms)
    //   force_grad_out  : [n_atoms * 3 * radial_coeff_count]
    //   virial_grad_out : [6 * radial_coeff_count]       (nullptr to skip)
    //
    // eval_grad_linear  — ∂(E_i, F_i, virial) / ∂β_linear
    //   site_energy_grad_out : [inum * alpha_scalar_count]   — per-atom ∂E_i/∂β_s
    //   force_grad_out       : [n_atoms * 3 * alpha_scalar_count]
    //   virial_grad_out      : [6 * alpha_scalar_count]      (nullptr to skip)
    //
    // eval_grad_all     — assembles radial + species + linear into (*, coeff_count) rows.
    //   site_energy_grad_out : [inum * coeff_count]      — per-atom ∂E_i/∂c_all
    //   force_grad_out       : [n_atoms * 3 * coeff_count]
    //   virial_grad_out      : [6 * coeff_count]         (nullptr to skip)
    // -----------------------------------------------------------------------
    void eval_grad_radial(int n_atoms, const int* types, int inum, const int* ilist, const int* numneigh, const int* firstneigh, const double* displacements, double* energy_grad_out, double* force_grad_out, double* virial_grad_out = nullptr);

    void eval_grad_linear(int n_atoms, const int* types, int inum, const int* ilist, const int* numneigh, const int* firstneigh, const double* displacements, double* site_energy_grad_out, double* force_grad_out, double* virial_grad_out = nullptr);

    void eval_grad_all(int n_atoms, const int* types, int inum, const int* ilist, const int* numneigh, const int* firstneigh, const double* displacements, double* site_energy_grad_out, double* force_grad_out, double* virial_grad_out = nullptr);

    int coeff_count() const { return radial_coeff_count + species_count + alpha_scalar_count; }

    // Evaluate the Chebyshev radial basis at distance 'dist'.
    // vals_out / ders_out must point to arrays of length radial_basis_size.
    // Pass nullptr for ders_out to skip derivative computation.
    void eval_radial_basis(double dist, double* vals_out, double* ders_out = nullptr);

    // -----------------------------------------------------------------------
    // Parameter accessors
    // -----------------------------------------------------------------------
    int get_species_count() const { return species_count; }
    int get_radial_func_count() const { return radial_func_count; }
    int get_radial_basis_size() const { return radial_basis_size; }
    int get_alpha_scalar_count() const { return alpha_scalar_count; }
    int get_alpha_moment_count() const { return alpha_moment_count; }
    int get_alpha_index_basic_count() const { return alpha_index_basic_count; }
    int get_alpha_index_times_count() const { return alpha_index_times_count; }
    double get_min_cutoff() const { return min_cutoff; }
    double get_max_cutoff() const { return max_cutoff; }
    double get_scaling() const { return scaling; }
    const std::string& get_potential_name() const { return potential_name; }

    // -----------------------------------------------------------------------
    // Coefficient setters (for training)
    // -----------------------------------------------------------------------
    void set_linear_coeffs(const double* coeffs) {
        std::copy(coeffs, coeffs + alpha_scalar_count, linear_coeffs.begin());
    }
    void set_species_coeffs(const double* coeffs) {
        std::copy(coeffs, coeffs + species_count, species_coeffs.begin());
    }
    void set_radial_basis_coeffs(const double* coeffs) {
        std::copy(coeffs, coeffs + radial_coeff_count, radial_basis_coeffs.begin());
    }
    void set_scaling(double s) {
        scaling = s;
        if (radial_basis != nullptr)
            radial_basis->scaling = s;
    }
    void set_min_cutoff(double d) {
        min_cutoff = d;
        if (radial_basis != nullptr)
            radial_basis->min_cutoff = d;
    }

    int get_radial_coeff_count() const { return radial_coeff_count; }

    // Raw array accessors (sizes described in comments above).
    const double* get_radial_basis_coeffs() const { return radial_basis_coeffs.data(); }
    // alpha_index_basic: flat [alpha_index_basic_count * 4]  (mu, px, py, pz)
    const int* get_alpha_index_basic() const { return alpha_index_basic[0].data(); }
    // alpha_index_times: flat [alpha_index_times_count * 4]  (i0, i1, mult, out)
    const int* get_alpha_index_times() const { return alpha_index_times[0].data(); }
    // alpha_moment_mapping: [alpha_scalar_count]
    const int* get_alpha_moment_mapping() const { return alpha_moment_mapping.data(); }
    // linear_coeffs: [alpha_scalar_count]
    const double* get_linear_coeffs() const { return linear_coeffs.data(); }
    // species_coeffs: [species_count]
    const double* get_species_coeffs() const { return species_coeffs.data(); }

  private:
    std::string potential_name = "Untitled";
    std::string potential_tag = "";

    int species_count = 0;
    double scaling = 1.0;

    // Radial basis
    RadialMTPBasis* radial_basis = nullptr;
    int radial_func_count = 0;
    int radial_basis_size = 0;
    int radial_coeff_count = 0;
    int radial_coeff_count_per_pair = 0;

    double min_cutoff = 0.0;
    double max_cutoff = 0.0;
    double max_cutoff_sq = 0.0;

    std::vector<double> radial_basis_coeffs;
    std::vector<std::array<int, 4>> alpha_index_basic; // [alpha_index_basic_count][4]  (mu, px, py, pz)
    std::vector<std::array<int, 4>> alpha_index_times; // [alpha_index_times_count][4]  (i0, i1, mult, out)
    std::vector<int> alpha_moment_mapping;             // [alpha_scalar_count]
    std::vector<double> linear_coeffs;                 // [alpha_scalar_count]
    std::vector<double> species_coeffs;                // [species_count]

    int alpha_moment_count = 0;
    int alpha_index_basic_count = 0;
    int alpha_index_times_count = 0;
    int alpha_scalar_count = 0;
    int max_alpha_index_basic = 0;

    // Per-call working buffers (resized lazily)
    std::vector<double> dist_powers;                 // [max_alpha_index_basic]
    std::vector<std::array<double, 3>> coord_powers; // [max_alpha_index_basic][3]
    std::vector<double> radial_vals;
    std::vector<double> radial_ders;
    std::vector<double> moment_tensor_vals;
    std::vector<double> nbh_energy_ders_wrt_moments;
    std::vector<std::array<double, 3>> moment_jacobian;  // [jac_cap * alpha_index_basic_count][3]
    std::vector<double> nbh_angular_factor;              // [jac_cap * alpha_index_basic_count]
    std::vector<double> nbh_angular_factor_jac;          // [jac_cap * alpha_index_basic_count * 3]
    int jac_cap = 0;
    // Working buffers for gradient functions
    std::vector<double> dM_dc_buf;   // [alpha_moment_count * radial_coeff_count]
    std::vector<double> dG_buf;      // [alpha_moment_count * radial_coeff_count]
    std::vector<double> dG_lin_buf;  // [alpha_moment_count * alpha_scalar_count]
    std::vector<int> valid_j;                    // [jac_cap]
    std::vector<std::array<double, 3>> valid_dr; // [jac_cap]

    friend void read_file(const std::string& filename, MTPPotential& potential);
};
