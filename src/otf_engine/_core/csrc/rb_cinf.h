/* -*- c++ -*- ----------------------------------------------------------
   RBCinf, a radial basis of the ETN and the MTP.
   Computes mlip-4's RadialBasisCinf (src/pots/radial_basis/cinf/rb_cinf.cpp
   at 79a39bf) as the ETN program evaluates it.
------------------------------------------------------------------------- */

#pragma once

#include "mtp_radial_basis.h"

#include <array>
#include <vector>

// phi_0 = c exp(-2 (1 - (min/max)^2) / (1 - x^2)), with x = r / r_cut and c a normalization.
// phi_{i+1} follows a three-term recursion in x^2, whose coefficients depend on min_cutoff / max_cutoff.
class RBCinf : public RadialMTPBasis {
  public:
    // Computes the recursion coefficients for 0 < min_cutoff / max_cutoff <= 0.8, and throws outside that range.
    RBCinf(double min_cutoff, double max_cutoff, int size);
    void calc_radial_basis_ders(double dist) override;
    void calc_radial_basis_rows(int count, const double* dist) override;

  private:
    double exp_ratio = 0;                                 // exponent of the envelope at x = 0
    double zeroth_poly = 0;                               // normalization of phi_0
    std::vector<std::array<double, 3>> recursive_coeffs;  // [size] recursion coefficients
    std::vector<double> row_work;                         // [4 * count] scratch of calc_radial_basis_rows: x^2, d(x^2)/dr, and two recursion derivatives
};
