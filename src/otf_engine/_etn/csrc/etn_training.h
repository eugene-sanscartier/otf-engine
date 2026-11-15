/* -*- c++ -*- ----------------------------------------------------------
   Derivatives of an ETN's site energies, and of a loss built on its energy,
   forces and virial, with respect to its coefficients.
------------------------------------------------------------------------- */

#pragma once

#include "pair_etn.h"

class ETNTraining : public PairETN {
  public:
    explicit ETNTraining(const std::string& filename);

    // dE_i/dc per central atom: [inum * coeff_count]; row ii for atom ilist[ii]. Throws when
    // given force_grad, since the ETN has no force gradients here.
    void eval_grad(const NeighList& list, double* site_energy_grad, double* force_grad, double* virial_grad) override;

    // Adds dL/dc to loss_grad [coeff_count], for a loss L whose derivatives
    // with respect to the energy, forces and virial of this structure are
    // given; dloss_dforces [n_atoms * 3] and dloss_dvirial [6] may be null.
    void eval_loss_grad(const NeighList& list, double dloss_denergy, const double* dloss_dforces, const double* dloss_dvirial, double* loss_grad) override;

  private:
    // Adds to grad the derivative of energy_weight * E_i, plus that of E_i's derivative along the
    // neighbor displacements dloss_dforces and dloss_dvirial give, for central atom ii.
    void accumulate_site_grad(const NeighList& list, int ii, int nbr_offset, double energy_weight, const double* dloss_dforces, const double* dloss_dvirial, double* grad);
    // Adds mixing_adj, the adjoint of the radial mixing, into the radial and species-pair columns of grad.
    void scatter_mixing_adj(double* grad) const;

    // First column of each coefficient block in the params order
    std::vector<int> radial_columns;        // [l]
    std::vector<int> species_pair_columns;  // [l]
    std::vector<int> core_columns;          // [core]
    std::vector<int> core_im_columns;       // [core]
    int species_columns = 0;

    // Per central atom: v, its tangent along the neighbor displacements, and their adjoints;
    // the bond vectors' tangents and adjoints; each neighbor's displacement
    std::vector<double> site_input_vals, input_tangent, input_tangent_adj;
    std::vector<double> site_bonds, bond_tangent, bond_tangent_adj;
    std::vector<double> neighbor_displacements;  // [cache_size * 3]
    std::vector<std::vector<double>> mixing_adj;  // [l] laid out as radial_mixing
};
