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

    // d(E_i, F, virial)/dc. With nonlinear false, only the columns the energy is linear in, the last core's
    // and the species', are filled; the other columns stay zero. force_grad throws with nonlinear true.
    //
    //   site_energy_grad : [inum * coeff_count]      per-atom dE_i/dc, row ii for atom ilist[ii]
    //   force_grad       : [n_atoms * 3 * coeff_count], or null
    //   virial_grad      : [6 * coeff_count], or null
    void eval_grad(const NeighList& list, double* site_energy_grad, double* force_grad, double* virial_grad, bool nonlinear);
    void eval_grad(const NeighList& list, double* site_energy_grad, double* force_grad, double* virial_grad) override { eval_grad(list, site_energy_grad, force_grad, virial_grad, true); }

    // Adds dL/dc to loss_grad [coeff_count], for a loss L whose derivatives
    // with respect to the energy, forces and virial of this structure are
    // given; dloss_dforces [n_atoms * 3] and dloss_dvirial [6] may be null.
    void eval_loss_grad(const NeighList& list, double dloss_denergy, const double* dloss_dforces, const double* dloss_dvirial, double* loss_grad) override;

  private:
    // eval_grad's kernel, A atoms per chunk; its linear path runs their backwards passes on a lane per atom.
    template <int A>
    void eval_grad_lanes(const NeighList& list, double* site_energy_grad, double* force_grad, double* virial_grad, bool nonlinear);

    // Grows the neighbor rows to jnum neighbors.
    void grow_neighbor_rows(int jnum);

    // The neighbors of the current atom, or of each atom of eval_loss_grad's chunk in blocks of atom_cache_size rows,
    // each array in rows over the neighbors: the harmonics, the radial basis' values and derivatives; and
    // eval_loss_grad's tangents of the harmonics and of the distance along the loss weight
    int rows_size = 0;
    std::vector<double> neighbor_harmonics;          // [rows_size * l_count^2]
    std::vector<double> neighbor_radial_vals;        // [rows_size * radial_basis_size]
    std::vector<double> neighbor_radial_ders;        // [rows_size * radial_basis_size]
    std::vector<double> neighbor_harmonic_tangents;  // [rows_size * l_count^2]
    std::vector<double> neighbor_dist_tangents;      // [rows_size]

    // eval_loss_grad: per entry of the input and bond vectors, the tangent and the adjoint of the value, for its chunk
    std::vector<Lanes> input_tangents;   // [input_shape.size]
    std::vector<Lanes> bond_tangents;    // [bonds_size]
    std::vector<Lanes> input_adjoints;   // [input_shape.size]
    std::vector<Lanes> bond_adjoints;    // [bonds_size]
    std::vector<Lanes> core_product_tangents;  // laid out as core_products
    std::vector<Lanes> core_product_adjoints;  // laid out as core_product_ders
    // dL/dS_l of each species pair's radial mixing over the structure: [l], laid out as radial_mixing
    std::vector<std::vector<double>> radial_mixing_grad;

    // eval_grad without the nonlinear columns: per entry of the input and bond vectors, and of one block's
    // products, the adjoint each backwards pass carries, the passes innermost; then per entry of the input,
    // the derivative of each last-core column's site energy, the columns innermost
    std::vector<double> input_pass_ders;      // [input_shape.size * pass count]
    std::vector<double> bond_pass_ders;       // [bonds_size * pass count]
    std::vector<double> product_pass_ders;    // [atom_core_product_ders.size() * pass count]
    std::vector<double> input_column_ders;    // [input_shape.size * column count]
    std::vector<double> column_pair_ders;     // [4 * column count] per column, dE_k/dr_ij's angular part (x, y, z), then its radial part

    // eval_grad's linear path: the atoms per chunk of a structure that has as many, and per atom of the chunk, a lane
    // innermost in the pass arrays above and in these
    static constexpr int jacobian_chunk = 16;
    std::vector<double> lane_input_vals;      // [input_shape.size * jacobian_chunk]
    std::vector<double> lane_core_products;   // [atom_core_products.size() * jacobian_chunk]
    std::vector<double> lane_column_ders;     // [input_shape.size * column count * jacobian_chunk]
};
