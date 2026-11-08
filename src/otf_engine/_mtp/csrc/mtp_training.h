/* -*- c++ -*- ----------------------------------------------------------
   Basis values of an MTP, and derivatives of its site energies, forces and
   virial, and of a loss built on them, with respect to its coefficients.
------------------------------------------------------------------------- */

#pragma once

#include "mtp_potential.h"

class MTPTraining : public PairMTP {
  public:
    explicit MTPTraining(const std::string& filename) : PairMTP(filename) {}

    // The coefficients as one vector c = [c_radial | c_species | beta_linear], of coeff_count().
    void get_coeffs(double* c) const;
    void set_coeffs(const double* c);

    // Scalar moment values per central atom: [inum * alpha_scalar_count].
    // Row ii holds the basis values for ilist[ii]; dot with linear_coeffs and
    // add species_coeffs for the site energy.
    void eval_basis(const NeighList& list, double* basis_out);

    // Radial basis at one distance, into arrays of radial_basis_size.
    void eval_radial_basis(double dist, double* vals_out, double* ders_out);

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

    // Per neighbor, the radial basis at its distance
    std::vector<double> neighbor_radial_vals;               // [neighbor * radial_basis_size]
    std::vector<double> neighbor_radial_ders;               // [neighbor * radial_basis_size]

    // Grows the neighbor rows to jnum neighbors.
    void grow_neighbor_rows(int jnum);

    // The neighbors of the current atom, each array in rows over the neighbors: the unit displacement's three
    // components and the inverse distance, the angular monomials, the radial functions' values and derivatives;
    // eval_loss_grad's tangents of the unit displacement along the loss weight, with that weight's radial component,
    // and the monomials' tangents; and for eval_grad's force passes the monomials' derivatives w.r.t. the displacement
    int rows_size = 0, jac_size = 0;
    std::vector<double> neighbor_units;              // [4 * rows_size]
    std::vector<double> neighbor_angulars;           // [angular_count * rows_size]
    std::vector<double> neighbor_radial_funcs;       // [radial_func_count * 2 * rows_size]
    std::vector<double> neighbor_unit_tangents;      // [4 * rows_size]
    std::vector<double> neighbor_angular_tangents;   // [angular_count * rows_size]
    std::vector<double> neighbor_angular_grads;      // [angular_count * 3 * jac_size]

    std::vector<double> radial_jacobian; // [alpha_index_basic_count * species_count * radial_coeff_count_per_pair] ders of basic moments wrt radial coeffs
    std::vector<double> dM_dc;      // [alpha_moment_count * radial_coeff_count]
    std::vector<double> dG;         // [alpha_moment_count * radial_coeff_count]
    std::vector<double> moment_ders;  // [alpha_moment_count * jac_size * 3] d(moment)/d(displacement) by component, then neighbor; the basic rows the moment Jacobian
    std::vector<double> scalar_ders;  // [neighbor * 3 * alpha_scalar_count] the basis values' rows of moment_ders, transposed
    std::vector<char> first_product;  // [alpha_index_times_count] whether the product is the first to add to its moment

    // eval_loss_grad: per moment, the tangent and the adjoint of the moment value.
    std::vector<double> moment_tangents;                    // [alpha_moment_count]
    std::vector<double> moment_adjoints;                    // [alpha_moment_count]
    std::vector<double> basic_adjoints_by_mu;               // [alpha_index_basic_count] moment_adjoints of the basic moments, grouped
};
