/* -*- c++ -*- ----------------------------------------------------------
   Derivatives of site energies, forces and virial with respect to the MTP
   coefficients, and of a loss built on them.
------------------------------------------------------------------------- */

#pragma once

#include "mtp_extrapolation.h"

class MTPTraining : public PairMTPExtrapolation {
  public:
    explicit MTPTraining(const std::string& filename)
        : PairMTPExtrapolation(filename), dist_powers(max_alpha_index_basic), coord_powers(max_alpha_index_basic), radial_vals(radial_func_count), radial_ders(radial_func_count) {
        dist_powers[0] = coord_powers[0][0] = coord_powers[0][1] = coord_powers[0][2] = 1;
    }

    // The coefficients as one vector c = [c_radial | c_species | beta_linear], of coeff_count().
    void get_coeffs(double* c) const;
    void set_coeffs(const double* c);

    using PairMTPExtrapolation::eval_grad;

    // d(E_i, F, virial)/dc. Any output may be null; the force-gradient passes
    // run only when force_grad is given. With radial false, the radial
    // columns are left zero and cost nothing.
    //
    //   site_energy_grad : [inum * coeff_count]      per-atom dE_i/dc
    //   force_grad       : [n_atoms * 3 * coeff_count]
    //   virial_grad      : [6 * coeff_count], only with force_grad
    void eval_grad(const NeighList& list, double* site_energy_grad, double* force_grad, double* virial_grad, bool radial = true);

    // Adds dL/dc to loss_grad [coeff_count], for a loss L whose derivatives
    // with respect to the energy, forces and virial of this structure are
    // given; dloss_dforces [n_atoms * 3] and dloss_dvirial [6] may be null.
    void eval_loss_grad(const NeighList& list, double dloss_denergy, const double* dloss_dforces, const double* dloss_dvirial, double* loss_grad);

  private:
    // Scatters radial_jacobian into dM_dc, whose columns are global radial
    // coefficient indices rather than per-pair ones.
    void scatter_radial_jacobian(int itype);
    // Propagates dM_dc through the composite moments, then back-propagates dG.
    void propagate_radial_moment_ders();

    // Per-neighbor powers and radial functions, and the moment Jacobian
    std::vector<double> dist_powers;                    // [max_alpha_index_basic] dist^k
    std::vector<std::array<double, 3>> coord_powers;    // [max_alpha_index_basic] [dx^k, dy^k, dz^k]
    std::vector<double> radial_vals;                    // [radial_func_count]
    std::vector<double> radial_ders;                    // [radial_func_count]
    int jac_size = 0;
    std::vector<std::array<double, 3>> moment_jacobian; // [jac_size * alpha_index_basic_count]

    // Per-neighbor angular scalar factor of each basic moment, pow_k(r)/r^rank_k,
    // and its derivative w.r.t. that neighbor's displacement.
    std::vector<double> angular_values;    // [neighbor * alpha_index_basic_count]
    std::vector<double> angular_jacobians; // [jac_size * alpha_index_basic_count * 3]

    std::vector<double> radial_jacobian; // [alpha_index_basic_count * species_count * radial_coeff_count_per_pair] ders of basic moments wrt radial coeffs
    std::vector<double> dM_dc;      // [alpha_moment_count * radial_coeff_count]
    std::vector<double> dG;         // [alpha_moment_count * radial_coeff_count]
    std::vector<double> moment_ders; // [alpha_moment_count * neighbor * 3] d(moment)/d(displacement)

    // eval_loss_grad: per neighbor, the radial basis, the unit displacement u
    // and its tangent along that neighbor's loss weight w; per angular
    // monomial, its tangent; per moment, the tangent and the adjoint of the
    // moment value.
    std::vector<double> neighbor_radial_vals;               // [neighbor * radial_basis_size]
    std::vector<double> neighbor_radial_ders;               // [neighbor * radial_basis_size]
    std::vector<std::array<double, 3>> unit_displacements;  // [neighbor]
    std::vector<std::array<double, 3>> unit_tangents;       // [neighbor]
    std::vector<double> w_dot_unit_r;                       // [neighbor]
    std::vector<double> angular_tangents;                   // [angular_count]
    std::vector<double> moment_tangents;                    // [alpha_moment_count]
    std::vector<double> moment_adjoints;                    // [alpha_moment_count]
    std::vector<double> basic_adjoints_by_mu;               // [alpha_index_basic_count] moment_adjoints of the basic moments, grouped
};
