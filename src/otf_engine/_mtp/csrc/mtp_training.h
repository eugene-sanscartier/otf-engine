/* -*- c++ -*- ----------------------------------------------------------
   Derivatives of energy, forces and virial with respect to the MTP
   coefficients.
------------------------------------------------------------------------- */

#pragma once

#include "mtp_extrapolation.h"

class MTPTraining : public PairMTPExtrapolation {
  public:
    explicit MTPTraining(const std::string& filename) : PairMTPExtrapolation(filename) {}

    // Which block of c = [c_radial | c_species | beta_linear] to differentiate.
    enum Cols { RADIAL, LINEAR, ALL };

    int cols_width(Cols cols) const;

    // Values and d(E_i, F, virial)/dc for one coefficient block.
    // Any output may be null; the force-gradient passes run only when
    // force_grad is given, so a site-energy-only call costs one forward pass.
    //
    //   site_e_grad : [inum * cols_width]      per-atom dE_i/dc
    //   force_grad  : [n_atoms * 3 * cols_width]
    //   virial_grad : [6 * cols_width]
    //
    // Returns the total energy.
    double compute_efs_grad(const NeighList& list, Cols cols, double* forces, double* virial, double* site_e_grad, double* force_grad, double* virial_grad);

    void eval_grad_radial(const NeighList& list, double* energy_grad, double* force_grad, double* virial_grad);
    void eval_grad_linear(const NeighList& list, double* site_e_grad, double* force_grad, double* virial_grad);
    void eval_grad_all(const NeighList& list, double* site_e_grad, double* force_grad, double* virial_grad);
    double compute_with_radial_grad(const NeighList& list, double* forces, double* virial, double* energy_grad, double* force_grad, double* virial_grad);

  private:
    // Scatters radial_jacobian into dM_dc, whose columns are global radial
    // coefficient indices rather than per-pair ones.
    void scatter_radial_jacobian(int itype);
    // Propagates dM_dc through the composite moments, then back-propagates dG.
    void propagate_radial_moment_ders();

    // Per-neighbor angular scalar factor of each basic moment, pow_k(r)/r^rank_k,
    // and its derivative w.r.t. that neighbor's displacement.
    std::vector<double> angular_values;    // [jac_size * alpha_index_basic_count]
    std::vector<double> angular_jacobians; // [jac_size * alpha_index_basic_count * 3]

    std::vector<double> radial_jacobian; // [alpha_index_basic_count * species_count * radial_coeff_count_per_pair] ders of basic moments wrt radial coeffs
    std::vector<double> dM_dc;      // [alpha_moment_count * radial_coeff_count]
    std::vector<double> dG;         // [alpha_moment_count * radial_coeff_count]
    std::vector<double> dG_lin;     // [alpha_moment_count * alpha_scalar_count]
    std::vector<double> site_grads; // [inum * radial_coeff_count], summed by the radial entry points
};
