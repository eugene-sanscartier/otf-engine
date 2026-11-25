/* -*- c++ -*- ----------------------------------------------------------
   Standalone MTP extrapolation grading — no LAMMPS dependency.
   Ported from lammps-mtp/src/ML-MTP/pair_mtp_extrapolation.h
   Original author: Richard Meng, Queen's University at Kingston, 10.02.25
------------------------------------------------------------------------- */

#pragma once

#include "pair_mtp.h"

class PairMTPExtrapolation : public PairMTP {
  public:
    using PairMTP::PairMTP;

    // Computes the results of PairMTP::compute, the extrapolation grade of each
    // neighborhood in neighborhood mode, and the largest grade.
    void compute(const NeighList& list) override;

    double get_max_grade() const { return max_grade; }
    const std::vector<double>& get_nbh_extrapolation_grades() const { return nbh_extrapolation_grades; }    // [n_atoms]

    // Sets the active set. invA is [coeff_count() * coeff_count()], row-major.
    // io_mvs.py reads and writes it in the #MVS_v1.1 block of an .almtp.
    void set_active_set(const double* invA, bool configuration_mode, int weight_scaling);

    bool has_active_set() const { return !inverse_active_set.empty(); }
    bool get_configuration_mode() const { return configuration_mode; }

  protected:
    void compile_grades(int natoms);                        // Configuration grade and its normalization

    bool configuration_mode = false;         // true grades the configuration, false each neighbourhood
    int weight_scaling = 2;                  // Power p in the 1/N^(p/2) energy scaling (from the MVS section)
    double max_grade = 0.0;                  // configuration grade or largest neighbourhood grade of the last compute
    std::vector<double> inverse_active_set;  // [coeff_count * coeff_count]

    // Working buffers
    std::vector<double> radial_basis_cache;      // [radial_basis_cache_size * radial_basis_size] radial basis at r_ij, one block of atom_cache_size rows per atom of the chunk
    int radial_basis_cache_size = 0;
    std::vector<double> energy_ders_wrt_coeffs;  // candidate vector of the configuration, or in neighbourhood mode one row per atom of the chunk [chunk_size * coeff_count]
    std::vector<double> chunk_grades;            // [chunk_size] neighbourhood grades of the chunk
    std::vector<double> grade_block;             // scratch of grade_rows

    // Neighbourhood mode only
    std::vector<double> nbh_extrapolation_grades;  // [n_atoms] Extrapolation grades of all neighbourhoods
};
