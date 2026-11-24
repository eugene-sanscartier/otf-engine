/* -*- c++ -*- ----------------------------------------------------------
   An ETN that grades extrapolation in compute, as PairMTPExtrapolation
   does for an MTP; mlip-4 has no counterpart.
------------------------------------------------------------------------- */

#pragma once

#include "pair_etn.h"

#include <vector>

class PairETNExtrapolation : public PairETN {
  public:
    using PairETN::PairETN;

    // PairETN::compute's results, plus the extrapolation grade of each neighborhood (neighborhood mode) and the
    // largest grade, left in the members below.
    void compute(const NeighList& list) override;

    double get_max_grade() const { return max_grade; }
    const std::vector<double>& get_nbh_extrapolation_grades() const { return nbh_extrapolation_grades; }    // [n_atoms]

    // invA is [coeff_count() * coeff_count()], row-major, as MaxVol holds it.
    void set_active_set(const double* invA, bool configuration_mode, int weight_scaling);

    bool has_active_set() const { return !inverse_active_set.empty(); }
    bool get_configuration_mode() const { return configuration_mode; }

  protected:
    // max_i |sum_j invA[i][j] energy_ders_wrt_coeffs[j]|
    double calculate_extrapolation_grade() const;

    bool configuration_mode = false;
    int weight_scaling = 2;                  // power p in the configuration grade's 1/N^(p/2)
    double max_grade = 0.0;
    std::vector<double> inverse_active_set;  // [coeff_count * coeff_count]

    std::vector<double> energy_ders_wrt_coeffs;    // dE_i/dc of the current atom, or of the configuration
    std::vector<double> nbh_extrapolation_grades;  // [n_atoms]
    std::vector<double> neighbor_radial_vals;      // [chunk_size * atom_cache_size * radial_basis_size] the radial basis at each neighbor, a block per atom of the chunk
    std::vector<Lanes> core_coeff_ders;            // [coeff_count] each atom of the chunk's dE_i/dc in the cores' columns, zero in the others
};
