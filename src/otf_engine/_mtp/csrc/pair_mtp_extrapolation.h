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

    // PairMTP::compute's results, plus the extrapolation grade of each
    // neighborhood (neighborhood mode) and the largest grade, left in the
    // members below.
    void compute(const NeighList& list) override;

    double get_max_grade() const { return max_grade; }
    const std::vector<double>& get_nbh_extrapolation_grades() const { return nbh_extrapolation_grades; }    // [n_atoms]

    // invA is [coeff_count() * coeff_count()], row-major. The #MVS_v1.1 block
    // of an .almtp holding it is read and written by almtp_io.py.
    void set_active_set(const double* invA, bool configuration_mode, int weight_scaling);

    bool has_active_set() const { return !inverse_active_set.empty(); }
    bool get_configuration_mode() const { return configuration_mode; }

  protected:
    void compile_grades(int natoms);                        // Configuration grade and its normalization

    bool configuration_mode = false;         // Is configuration mode?
    int weight_scaling = 2;                  // Power p in the 1/N^(p/2) energy scaling (from the MVS section)
    double max_grade = 0.0;                  // Grade of current iteration
    std::vector<double> inverse_active_set;  // [coeff_count * coeff_count]

    //Working buffers
    std::vector<double> radial_basis_cache;      // [radial_basis_cache_size * radial_basis_size] radial basis at r_ij, a block of atom_cache_size per atom of the chunk
    int radial_basis_cache_size = 0;
    std::vector<double> energy_ders_wrt_coeffs;  // Candidate information vector: the configuration's, or in neighbourhood mode each atom of the chunk's [chunk_size * coeff_count]
    std::vector<double> chunk_grades;            // [chunk_size] the chunk's neighbourhood grades
    std::vector<double> grade_block;             // grade_rows' scratch

    // Only needed for neighbourhood mode
    std::vector<double> nbh_extrapolation_grades;  // [n_atoms] Extrapolation grades of all neighbourhoods
};
