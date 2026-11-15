/* -*- c++ -*- ----------------------------------------------------------
   mlip-4's C-infinity radial basis (RadialBasisCinf), evaluated as the ETN
   program evaluates it.
------------------------------------------------------------------------- */

#pragma once

#include <array>
#include <vector>

class RadialBasisCinf {
  public:
    // size basis functions on [min_dist, max_dist); min_dist / max_dist must not exceed 0.8.
    RadialBasisCinf(int size, double min_dist, double max_dist);

    // Values at dist < max_dist, times scaling, and their derivatives with respect to dist.
    void calc(double dist, double* vals, double* ders) const;

    int size;
    double min_dist;
    double max_dist;
    double scaling = 1.0;

  private:
    double max_dist_sq;
    double exp_ratio;                                 // exponent of the envelope at x = 0
    double zeroth_poly;                               // normalization of the first function
    std::vector<std::array<double, 3>> recursive_coeffs; // [size - 1] three-term recursion
};
