/* -*- c++ -*- ----------------------------------------------------------
   Derivatives of site energies, forces and virial with respect to the MTP
   coefficients, and of a loss built on them.
------------------------------------------------------------------------- */

#pragma once

#include "mtp_extrapolation.h"

class MTPTraining : public PairMTPExtrapolation {
  public:
    explicit MTPTraining(const std::string& filename) : PairMTPExtrapolation(filename) {}

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

    // Per-neighbor angular scalar factor of each basic moment, pow_k(r)/r^rank_k,
    // and its derivative w.r.t. that neighbor's displacement.
    std::vector<double> angular_values;    // [neighbor * alpha_index_basic_count]
    std::vector<double> angular_jacobians; // [jac_size * alpha_index_basic_count * 3]

    std::vector<double> radial_jacobian; // [alpha_index_basic_count * species_count * radial_coeff_count_per_pair] ders of basic moments wrt radial coeffs
    std::vector<double> dM_dc;      // [alpha_moment_count * radial_coeff_count]
    std::vector<double> dG;         // [alpha_moment_count * radial_coeff_count]
    std::vector<double> moment_ders; // [alpha_moment_count * neighbor * 3] d(moment)/d(displacement)

    // eval_loss_grad: per neighbor, the radial basis and the angular factors'
    // derivative along that neighbor's loss weight w; per moment, the
    // derivative along w and the adjoint of the moment value.
    std::vector<double> neighbor_radial_vals;   // [neighbor * radial_basis_size]
    std::vector<double> neighbor_radial_ders;   // [neighbor * radial_basis_size]
    std::vector<double> angular_ders_along_w;   // [neighbor * alpha_index_basic_count]
    std::vector<double> w_dot_unit_r;           // [neighbor]
    std::vector<double> moment_tangents;        // [alpha_moment_count]
    std::vector<double> moment_adjoints;        // [alpha_moment_count]
    std::vector<double> radial_adjoints;        // [2 * radial_func_count] per neighbor
};
