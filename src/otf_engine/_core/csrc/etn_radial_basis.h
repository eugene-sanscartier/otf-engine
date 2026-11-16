/* -*- c++ -*- ----------------------------------------------------------
   Standalone ETN radial basis — no LAMMPS dependency.
   Computes mlip-4's RadialBasisCinf (src/pots/radial_basis/cinf/rb_cinf.cpp
   at 79a39bf) as the ETN program evaluates it.
------------------------------------------------------------------------- */

#pragma once

#include "mtp_radial_basis.h"

#include <array>
#include <vector>

// phi_0 = exp(-2 (1 - (min/max)^2) / (1 - x^2)) times a constant, x = r / r_cut, and a
// three-term recursion in x^2 above, whose coefficients depend on min_cutoff / max_cutoff.
class RBCinf : public RadialMTPBasis {
  public:
    // min_cutoff / max_cutoff must not exceed 0.8.
    RBCinf(double min_cutoff, double max_cutoff, int size);
    void calc_radial_basis_ders(double dist) override;

  private:
    double exp_ratio = 0;                                 // exponent of the envelope at x = 0
    double zeroth_poly = 0;                               // normalization of phi_0
    std::vector<std::array<double, 3>> recursive_coeffs;  // [size] recursion coefficients
};
