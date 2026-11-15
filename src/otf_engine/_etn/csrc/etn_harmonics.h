/* -*- c++ -*- ----------------------------------------------------------
   Real spherical harmonics and the 3-j symbols that couple them, in the
   convention of mlip-4's ETN (angular_basis.cpp and wigner.cpp).
------------------------------------------------------------------------- */

#pragma once

#include <vector>

// Real spherical harmonics Y_lm for l < l_count, stored at l * l + l + m: for m > 0 the
// cos(m phi) harmonic and for m < 0 the sin(|m| phi) one, both without the Condon-Shortley
// phase and normalized to 1 on the sphere.
class RealHarmonics {
  public:
    explicit RealHarmonics(int l_count);

    // Values and gradients at the unit vector u; the gradient is tangent to the sphere.
    void compute(const double u[3]);

    int l_count;
    std::vector<double> vals;   // [l_count^2]
    std::vector<double> grads;  // [l_count^2 * 3]

  private:
    std::vector<double> norms;     // [l_count^2] normalization of each harmonic
    std::vector<double> legendre;  // [l_count * (l_count + 1)] d^m P_l / dz^m at l * (l_count + 1) + m
    std::vector<double> cos_m;     // [l_count] Re (x + i y)^m
    std::vector<double> sin_m;     // [l_count] Im (x + i y)^m
};

// Wigner 3-j symbol of complex spherical harmonics.
double wigner_3j(int l1, int l2, int l3, int m1, int m2, int m3);

// 3-j symbol of real harmonics: the real part of the complex one carried to real harmonics when
// l1 + l2 + l3 is even, its imaginary part when odd.
double real_wigner_3j(int l1, int m1, int l2, int m2, int l3, int m3);

struct WignerEntry {
    int m1, m2, m3;
    double w;
};

// The nonzero real 3-j symbols of (l1, l2, l3), ordered m3, m2, m1 with m1 fastest.
std::vector<WignerEntry> real_wigner_entries(int l1, int l2, int l3);
