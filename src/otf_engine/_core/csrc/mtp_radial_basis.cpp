/* -*- c++ -*- ----------------------------------------------------------
   Standalone MTP radial basis — no LAMMPS dependency.
   Ported from lammps-mtp/src/ML-MTP/mtp_radial_basis.cpp
   Original author: Richard Meng, Queen's University at Kingston, 22.11.24
------------------------------------------------------------------------- */

#include "mtp_radial_basis.h"

#include <stdexcept>

RadialMTPBasis::RadialMTPBasis(double min_cutoff, double max_cutoff, int size) : size(size), min_cutoff(min_cutoff), max_cutoff(max_cutoff) {
    if (!(max_cutoff > min_cutoff))
        throw std::runtime_error("MTP radial basis: maximum cutoff must exceed the minimum cutoff");
    if (size < 2)
        throw std::runtime_error("MTP radial basis: size must be at least 2");

    radial_basis_vals.resize(size);
    radial_basis_ders.resize(size);
}

// calc_radial_basis_ders at each distance.
void RadialMTPBasis::calc_radial_basis_rows(int count, const double* dist) {
    row_count = count;
    rows_vals.resize((size_t) size * count);
    rows_ders.resize((size_t) size * count);
    for (int k = 0; k < count; k++) {
        calc_radial_basis_ders(dist[k]);
        for (int i = 0; i < size; i++) {
            rows_vals[(size_t) i * count + k] = radial_basis_vals[i];
            rows_ders[(size_t) i * count + k] = radial_basis_ders[i];
        }
    }
}

// ---------------------------------------------------------------------------
void RBChebyshev::calc_radial_basis_ders(double dist) {
    const double delta = dist - max_cutoff;
    const double span = max_cutoff - min_cutoff;
    const double mult = 2.0 / span;
    const double ksi = (2 * dist - (min_cutoff + max_cutoff)) / span;

    double val0 = scaling * delta * delta;
    double val1 = scaling * (ksi * delta * delta);
    double der0 = scaling * 2 * delta;
    double der1 = scaling * (mult * delta * delta + 2 * ksi * delta);
    radial_basis_vals[0] = val0;
    radial_basis_vals[1] = val1;
    radial_basis_ders[0] = der0;
    radial_basis_ders[1] = der1;

    for (int i = 2; i < size; i++) {
        const double val = 2 * ksi * val1 - val0;
        const double der = 2 * (mult * val1 + ksi * der1) - der0;
        radial_basis_vals[i] = val;
        radial_basis_ders[i] = der;
        val0 = val1;
        val1 = val;
        der0 = der1;
        der1 = der;
    }
}
