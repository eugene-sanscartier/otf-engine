/* -*- c++ -*- ----------------------------------------------------------
   Real spherical harmonics and the 3-j symbols that couple them.
------------------------------------------------------------------------- */

#pragma once

#include <vector>

// Real spherical harmonics Y_lm for l < l_count, stored at l * l + l + m.
// Y_lm is the cos(m phi) harmonic for m > 0 and the sin(|m| phi) harmonic for m < 0.
// The harmonics have no Condon-Shortley phase and a norm of 1 on the sphere.
class RealHarmonics {
  public:
    explicit RealHarmonics(int l_count);

    // Computes the values and gradients at the unit vector u; the gradient is tangent to the sphere.
    void compute(const double u[3]);
    // Computes the values and gradients at count unit vectors (x, y, z) into rows_vals and rows_grads, each operation looping over the vectors.
    void compute_rows(int count, const double* x, const double* y, const double* z);
    // Copies vector k of the rows into vals_out [l_count^2] and grads_out [l_count^2 * 3].
    void copy_row(int k, double* vals_out, double* grads_out) const {
        for (size_t e = 0; e < vals.size(); e++) vals_out[e] = rows_vals[e * row_count + k];
        for (size_t e = 0; e < grads.size(); e++) grads_out[e] = rows_grads[e * row_count + k];
    }
    // Copies vector k of the rows into vals and grads.
    void select_row(int k) { copy_row(k, vals.data(), grads.data()); }

    int l_count;
    std::vector<double> vals;   // [l_count^2]
    std::vector<double> grads;  // [l_count^2 * 3]

    // Results of the last compute_rows: the vector count, and the values and gradients [entry * row_count]
    int row_count = 0;
    std::vector<double> rows_vals, rows_grads;

  private:
    std::vector<double> norms;     // [l_count^2] normalization of each harmonic
    std::vector<double> legendre;  // [l_count * (l_count + 1)] d^m P_l / dz^m at l * (l_count + 1) + m
    std::vector<double> cos_m;     // [l_count] Re (x + i y)^m
    std::vector<double> sin_m;     // [l_count] Im (x + i y)^m
    std::vector<double> rows_legendre, rows_cos, rows_sin;    // legendre, cos_m and sin_m for compute_rows, [entry * row_count]
};

// Wigner 3-j symbol of complex spherical harmonics.
double wigner_3j(int l1, int l2, int l3, int m1, int m2, int m3);

// 3-j symbol of real harmonics: the complex symbol carried to real harmonics, its real part for an even
// l1 + l2 + l3 and its imaginary part for an odd one.
double real_wigner_3j(int l1, int m1, int l2, int m2, int l3, int m3);

struct WignerEntry {
    int m1, m2, m3;
    double w;
};

// The nonzero real 3-j symbols of (l1, l2, l3), ordered m3, m2, m1 with m1 fastest.
std::vector<WignerEntry> real_wigner_entries(int l1, int l2, int l3);
