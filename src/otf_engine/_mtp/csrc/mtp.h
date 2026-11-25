/* -*- c++ -*- ----------------------------------------------------------
   Basis values of an MTP, and derivatives of its site energies, forces and
   virial, and of a loss built on them, with respect to its coefficients.
------------------------------------------------------------------------- */

#pragma once

#include "pair_mtp.h"

class MTP : public PairMTP {
  public:
    using PairMTP::PairMTP;

    // Computes the basis values of each central atom into basis_out [inum * alpha_scalar_count].
    // Row ii belongs to ilist[ii]. Its dot product with linear_coeffs, plus species_coeffs, gives the site energy.
    void eval_basis(const NeighList& list, double* basis_out);

    // Computes d(E_i, F, virial)/dc. Any output may be null.
    // The force passes run only when force_grad is given. With radial false, the radial columns stay zero and are not computed.
    //
    //   site_energy_grad : [inum * coeff_count]      per-atom dE_i/dc
    //   force_grad       : [n_atoms * 3 * coeff_count]
    //   virial_grad      : [6 * coeff_count], only with force_grad
    void eval_grad(const NeighList& list, double* site_energy_grad, double* force_grad, double* virial_grad, bool radial);
    void eval_grad(const NeighList& list, double* site_energy_grad, double* force_grad, double* virial_grad) override { eval_grad(list, site_energy_grad, force_grad, virial_grad, true); }

    // Adds dL/dc to loss_grad [coeff_count], given the derivatives of L with respect to the structure's energy, forces and virial.
    // dloss_dforces [n_atoms * 3] and dloss_dvirial [6] may be null.
    void eval_loss_grad(const NeighList& list, double dloss_denergy, const double* dloss_dforces, const double* dloss_dvirial, double* loss_grad) override;

  private:
    // Radial basis at the distance of each neighbor
    std::vector<double> neighbor_radial_vals;               // [neighbor * radial_basis_size]
    std::vector<double> neighbor_radial_ders;               // [neighbor * radial_basis_size]

    // Grows the neighbor rows to jnum neighbors.
    void grow_neighbor_rows(int jnum);

    // Neighbor rows: each array holds one row per quantity, with one entry per neighbor.
    // eval_grad fills them for the current atom. eval_loss_grad gives each atom of its chunk a block with rows of atom_cache_size.
    // neighbor_units: rows u_x, u_y, u_z of the unit displacement, then the inverse distance.
    // neighbor_angulars, neighbor_angular_tangents: angular monomials of u and their tangents.
    // neighbor_radial_funcs: value and derivative of each radial function.
    // neighbor_unit_tangents: rows v_x, v_y, v_z of the tangent of u along the loss weight w, then w . u.
    // neighbor_angular_grads: derivatives of the monomials with respect to the displacement, for the force passes of eval_grad.
    int rows_size = 0, jac_size = 0;
    std::vector<double> neighbor_units;              // [4 * rows_size]
    std::vector<double> neighbor_angulars;           // [angular_count * rows_size]
    std::vector<double> neighbor_radial_funcs;       // [radial_func_count * 2 * rows_size]
    std::vector<double> neighbor_unit_tangents;      // [4 * rows_size]
    std::vector<double> neighbor_angular_tangents;   // [angular_count * rows_size]
    std::vector<double> neighbor_angular_grads;      // [angular_count * 3 * jac_size]

    std::vector<double> radial_jacobian; // [alpha_index_basic_count * species_count * radial_coeff_count_per_pair] derivatives of the basic moments with respect to the radial coefficients
    std::vector<double> dM_dc;      // [alpha_moment_count * radial_coeff_count]
    std::vector<double> dG;         // [alpha_moment_count * radial_coeff_count]
    // [alpha_moment_count * 3 * jac_size] derivative of moment k with respect to component a of neighbor jj at k * 3 * jac_size + a * V + jj, V neighbors within the cutoff.
    // The basic rows hold the moment Jacobian.
    std::vector<double> moment_ders;
    std::vector<double> scalar_ders;  // [neighbor * 3 * alpha_scalar_count] rows of moment_ders for the basis values, transposed
    std::vector<char> first_product;  // [alpha_index_times_count] whether the product is the first to add to its moment

    // Tangent and adjoint of each moment value over the chunk of eval_loss_grad
    std::vector<Lanes> moment_tangents;                     // [alpha_moment_count]
    std::vector<Lanes> moment_adjoints;                     // [alpha_moment_count]
    std::vector<double> basic_adjoints_by_mu;               // [alpha_index_basic_count] moment_adjoints of the basic moments, in basic_by_mu order
};
