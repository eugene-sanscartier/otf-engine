/* -*- c++ -*- ----------------------------------------------------------
   Standalone MTP extrapolation grading — no LAMMPS dependency.
   Ported from lammps-mtp/src/ML-MTP/pair_mtp_extrapolation.h
   Original author: Richard Meng, Queen's University at Kingston, 10.02.25
------------------------------------------------------------------------- */

#pragma once

#include "mtp_potential.h"

class PairMTPExtrapolation : public PairMTP {
  public:
    explicit PairMTPExtrapolation(const std::string& filename) : PairMTP(filename) {}

    int coeff_count() const { return radial_coeff_count + species_count + alpha_scalar_count; }

    // Energy, forces and virial, plus the candidate information vector and the
    // extrapolation grades when they are asked for. With rows_out and
    // grades_out both null this is PairMTP::compute.
    //
    // rows_out   : [inum * coeff_count()] dE_i/dc per central atom — null to skip
    // grades_out : [inum] per-neighborhood grades — null to skip
    //
    // Returns the total energy.
    double compute(const NeighList& list, double* forces, double* virial, double* eatom, double* rows_out, double* grades_out);

    double compute(const NeighList& list, double* forces, double* virial, double* eatom) {
        return compute(list, forces, virial, eatom, nullptr, nullptr);
    }

    // Per-atom information vectors: [inum * coeff_count()].
    void eval_grad(const NeighList& list, double* rows_out) {
        compute(list, nullptr, nullptr, nullptr, rows_out, nullptr);
    }

    // Per-neighborhood grades into grades_out [inum]; returns the configuration
    // grade. In configuration mode every entry of grades_out is that one grade.
    double grade(const NeighList& list, double* grades_out);

    // Grade of the information vector currently in energy_ders_wrt_coeffs.
    double calculate_extrapolation_grade();

    // invA is [coeff_count() * coeff_count()], row-major. The #MVS_v1.1 block
    // of an .almtp holding it is read and written by almtp_io.py.
    void set_active_set(const double* invA, bool configuration_mode);

    bool has_active_set() const { return !inverse_active_set.empty(); }
    bool get_configuration_mode() const { return configuration_mode; }

  protected:
    bool configuration_mode = false;         // Is configuration mode?
    double max_grade = 0.0;                  // Grade of current iteration
    std::vector<double> inverse_active_set;  // [coeff_count * coeff_count]

    std::vector<double> radial_jacobian;         // ders of basic moments wrt radial coeffs
    std::vector<double> energy_ders_wrt_coeffs;  // candidate information vector
    std::vector<double> cfg_ders_wrt_coeffs;     // its sum over the configuration
};
