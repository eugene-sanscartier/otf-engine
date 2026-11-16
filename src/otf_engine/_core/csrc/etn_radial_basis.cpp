/* -*- c++ -*- ----------------------------------------------------------
   Standalone ETN radial basis — no LAMMPS dependency.
   Computes mlip-4's RadialBasisCinf (src/pots/radial_basis/cinf/rb_cinf.cpp
   at 79a39bf) as the ETN program evaluates it.
------------------------------------------------------------------------- */

#include "etn_radial_basis.h"
#include "etn_radial_basis_tables.h"

#include <cmath>
#include <stdexcept>
#include <string>

// The recursion coefficients for the ratio min_cutoff / max_cutoff, from their Chebyshev
// expansions, summed as RadialBasisCinf::PostConstruct sums them.
RBCinf::RBCinf(double min_cutoff, double max_cutoff, int size) : RadialMTPBasis(min_cutoff, max_cutoff, size) {
    if (size > RADIAL_BASIS_MAX_SIZE)
        throw std::runtime_error("ETN radial basis: radial_basis_size must not exceed " + std::to_string(RADIAL_BASIS_MAX_SIZE));

    const double min_to_max_ratio = min_cutoff / max_cutoff;
    if (!(min_to_max_ratio > 0.0 && min_to_max_ratio <= 0.8))
        throw std::runtime_error("ETN radial basis: min_dist / max_dist must lie in (0, 0.8]");

    exp_ratio = -2 * (1 - min_to_max_ratio * min_to_max_ratio);

    const double cheb_acos_x = std::acos(1.25 * min_to_max_ratio);
    recursive_coeffs.resize(size);
    for (int i = 0; i < size; i++)
        for (int j = 0; j < 3; j++) {
            double accumulator = 0;
            for (int k = RADIAL_BASIS_CHEB_SIZE - 1; k >= 0; k--)
                accumulator += CHEB_RECURSIVE_COEFFS[i][j][k] * std::cos(2 * k * cheb_acos_x);
            recursive_coeffs[i][j] = accumulator;
        }

    double accumulator = 0;
    for (int k = RADIAL_BASIS_CHEB_SIZE - 1; k >= 0; k--)
        accumulator += CHEB_ZEROTH_POLY[k] * std::cos(2 * k * cheb_acos_x);
    zeroth_poly = accumulator;
}

// RadialBasisCinf::ComposeProgram, in its order of operations, with the derivatives beside.
void RBCinf::calc_radial_basis_ders(double dist) {
    const double max_cutoff_sq = max_cutoff * max_cutoff;
    const double x_sq = dist * dist * (1.0 / max_cutoff_sq);
    const double dx_sq = 2 * dist * (1.0 / max_cutoff_sq);

    const double exponent = 1.0 / exp_ratio - (1.0 / exp_ratio) * x_sq;
    const double exp_term = std::exp(1.0 / exponent);
    const double one_minus_x_sq = 1 - x_sq;

    double* vals = radial_basis_vals.data();
    double* ders = radial_basis_ders.data();
    vals[0] = zeroth_poly * scaling * exp_term;
    double der_prev = 0;
    double der = vals[0] * exp_ratio / (one_minus_x_sq * one_minus_x_sq);
    ders[0] = der * dx_sq;

    for (int i = 0; i < size - 1; i++) {
        const auto& c = recursive_coeffs[i];
        vals[i + 1] = c[0] * x_sq * vals[i] + c[0] * c[1] * vals[i];
        if (i > 0) vals[i + 1] += c[0] * c[2] * vals[i - 1];

        const double der_next = c[0] * (vals[i] + x_sq * der) + c[0] * c[1] * der + c[0] * c[2] * der_prev;
        der_prev = der;
        der = der_next;
        ders[i + 1] = der * dx_sq;
    }
}
