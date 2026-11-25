/* -*- c++ -*- ----------------------------------------------------------
   An ETN potential with its input and bond vectors, and the gradients
   with respect to its coefficients of its site energies and of a loss
   on its energy, forces and virial.
------------------------------------------------------------------------- */

#pragma once

#include "pair_etn.h"

class ETN : public PairETN {
  public:
    using PairETN::PairETN;

    // Computes the input vector [inum * input_shape.size] and the bond vectors [inum * bonds_size] of each central atom.
    // Row ii belongs to atom ilist[ii], laid out by get_input_shape() and get_bond_offset().
    void eval_bonds(const NeighList& list, double* inputs_out, double* bonds_out);

    // Computes d(E_i, F, virial)/dc.
    // With nonlinear false, it fills only the columns in which the energy is linear, those of the last core and of the species.
    // The other columns stay zero. A non-null force_grad with nonlinear true throws.
    //
    //   site_energy_grad : [inum * coeff_count]      per-atom dE_i/dc, row ii for atom ilist[ii]
    //   force_grad       : [n_atoms * 3 * coeff_count], or null
    //   virial_grad      : [6 * coeff_count], or null
    void eval_grad(const NeighList& list, double* site_energy_grad, double* force_grad, double* virial_grad, bool nonlinear);
    void eval_grad(const NeighList& list, double* site_energy_grad, double* force_grad, double* virial_grad) override { eval_grad(list, site_energy_grad, force_grad, virial_grad, true); }

    // Adds dL/dc to loss_grad [coeff_count], given the derivatives of L with respect to the energy,
    // forces [n_atoms * 3] and virial [6] of this structure. dloss_dforces and dloss_dvirial may be null.
    void eval_loss_grad(const NeighList& list, double dloss_denergy, const double* dloss_dforces, const double* dloss_dvirial, double* loss_grad) override;

  private:
    // Runs eval_grad on chunks of A atoms. The linear path runs the backwards passes of a chunk with one lane per atom.
    template <int A>
    void eval_grad_lanes(const NeighList& list, double* site_energy_grad, double* force_grad, double* virial_grad, bool nonlinear);

    // Grows the neighbor rows to jnum neighbors.
    void grow_neighbor_rows(int jnum);

    // Rows over the neighbors of the current atom, or of each atom of the eval_loss_grad chunk in blocks of atom_cache_size rows:
    // the harmonics, the values and derivatives of the radial basis, and in eval_loss_grad the tangents of the harmonics
    // and of the distance along the loss weight
    int rows_size = 0;
    std::vector<double> neighbor_harmonics;          // [rows_size * l_count^2]
    std::vector<double> neighbor_radial_vals;        // [rows_size * radial_basis_size]
    std::vector<double> neighbor_radial_ders;        // [rows_size * radial_basis_size]
    std::vector<double> neighbor_harmonic_tangents;  // [rows_size * l_count^2]
    std::vector<double> neighbor_dist_tangents;      // [rows_size]

    // eval_loss_grad: the tangent and the adjoint of each entry of the input and bond vectors, one lane per atom of the chunk
    std::vector<Lanes> input_tangents;   // [input_shape.size]
    std::vector<Lanes> bond_tangents;    // [bonds_size]
    std::vector<Lanes> input_adjoints;   // [input_shape.size]
    std::vector<Lanes> bond_adjoints;    // [bonds_size]
    std::vector<Lanes> core_product_tangents;  // laid out as core_products
    std::vector<Lanes> core_product_adjoints;  // laid out as core_product_ders
    // dL/dS_l per species pair, summed over the structure: [l], laid out as radial_mixing
    std::vector<std::vector<double>> radial_mixing_grad;

    // eval_grad without the nonlinear columns: the adjoint that each backwards pass carries per entry of the input,
    // of the bond vectors and of the products of one block, passes innermost; then the derivative of the site energy
    // of each last-core column per entry of the input, columns innermost
    std::vector<double> input_pass_ders;      // [input_shape.size * pass count]
    std::vector<double> bond_pass_ders;       // [bonds_size * pass count]
    std::vector<double> product_pass_ders;    // [atom_core_product_ders.size() * pass count]
    std::vector<double> input_column_ders;    // [input_shape.size * column count]
    std::vector<double> column_pair_ders;     // [4 * column count] angular part (x, y, z) of dE_k/dr_ij per column, then its radial part

    // Linear path of eval_grad: jacobian_chunk atoms per chunk in a structure of at least that many atoms.
    // Each atom of the chunk has a lane, innermost in the pass arrays above and in these.
    static constexpr int jacobian_chunk = 16;
    std::vector<double> lane_input_vals;      // [input_shape.size * jacobian_chunk]
    std::vector<double> lane_core_products;   // [atom_core_products.size() * jacobian_chunk]
    std::vector<double> lane_column_ders;     // [input_shape.size * column count * jacobian_chunk]
};
