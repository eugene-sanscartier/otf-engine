/* -*- c++ -*- ----------------------------------------------------------
   Input and bond vectors of an ETN, and derivatives of its site energies,
   and of a loss built on its energy, forces and virial, with respect to
   its coefficients.
------------------------------------------------------------------------- */

#pragma once

#include "pair_etn.h"

class ETN : public PairETN {
  public:
    using PairETN::PairETN;

    // Per central atom, its input vector [inum * input_shape.size] and its bond vectors [inum * bonds_size];
    // row ii for atom ilist[ii], laid out as get_input_shape() and get_bond_offset() give.
    void eval_bonds(const NeighList& list, double* inputs_out, double* bonds_out);

    // dE_i/dc per central atom: [inum * coeff_count]; row ii for atom ilist[ii]. Throws when
    // given force_grad, since the ETN has no force gradients here.
    void eval_grad(const NeighList& list, double* site_energy_grad, double* force_grad, double* virial_grad) override;

    // Adds dL/dc to loss_grad [coeff_count], for a loss L whose derivatives
    // with respect to the energy, forces and virial of this structure are
    // given; dloss_dforces [n_atoms * 3] and dloss_dvirial [6] may be null.
    void eval_loss_grad(const NeighList& list, double dloss_denergy, const double* dloss_dforces, const double* dloss_dvirial, double* loss_grad) override;

  private:
    // Grows the neighbor rows to jnum neighbors.
    void grow_neighbor_rows(int jnum);

    // The neighbors of the current atom, each array in rows over the neighbors: the harmonics, the radial basis'
    // values and derivatives; and eval_loss_grad's tangents of the harmonics and of the distance along the loss weight
    int rows_size = 0;
    std::vector<double> neighbor_harmonics;          // [rows_size * l_count^2]
    std::vector<double> neighbor_radial_vals;        // [rows_size * radial_basis_size]
    std::vector<double> neighbor_radial_ders;        // [rows_size * radial_basis_size]
    std::vector<double> neighbor_harmonic_tangents;  // [rows_size * l_count^2]
    std::vector<double> neighbor_dist_tangents;      // [rows_size]

    // eval_loss_grad: per entry of the input and bond vectors, the tangent and the adjoint of the value.
    std::vector<double> input_tangents;   // [input_shape.size]
    std::vector<double> bond_tangents;    // [bonds_size]
    std::vector<double> input_adjoints;   // [input_shape.size]
    std::vector<double> bond_adjoints;    // [bonds_size]
    std::vector<double> core_product_tangents;  // laid out as core_products
    std::vector<double> core_product_adjoints;  // laid out as core_product_ders
};
