/* -*- c++ -*- ----------------------------------------------------------
   An ETN potential that grades extrapolation in compute.
------------------------------------------------------------------------- */

#pragma once

#include "pair_etn.h"

#include <vector>

class PairETNExtrapolation : public PairETN {
  public:
    using PairETN::PairETN;

    // Computes the results of PairETN::compute, the largest grade, and in neighborhood mode the grade of each neighborhood.
    void compute(const NeighList& list) override;

    double get_max_grade() const { return max_grade; }
    const std::vector<double>& get_nbh_extrapolation_grades() const { return nbh_extrapolation_grades; }    // [n_atoms]

    // Sets the inverse active set invA [coeff_count() * coeff_count()], row-major, and the grading mode.
    void set_active_set(const double* invA, bool configuration_mode, int weight_scaling);

    bool has_active_set() const { return !inverse_active_set.empty(); }
    bool get_configuration_mode() const { return configuration_mode; }

  protected:
    bool configuration_mode = false;
    int weight_scaling = 2;                  // power p in the configuration grade's 1/N^(p/2)
    double max_grade = 0.0;
    std::vector<double> inverse_active_set;  // [coeff_count * coeff_count]

    std::vector<double> energy_ders_wrt_coeffs;    // dE/dc of the configuration, or dE_i/dc of each atom of the chunk [chunk_size * coeff_count]
    std::vector<double> chunk_grades;              // [chunk_size] neighborhood grades of the chunk
    std::vector<double> grade_block;               // grade_rows' scratch
    std::vector<double> nbh_extrapolation_grades;  // [n_atoms]
    std::vector<double> neighbor_radial_vals;      // [chunk_size * atom_cache_size * radial_basis_size] the radial basis at each neighbor, a block per atom of the chunk
    std::vector<Lanes> core_coeff_ders;            // [coeff_count] dE_i/dc of each atom of the chunk in the core columns, zero in the others
};
