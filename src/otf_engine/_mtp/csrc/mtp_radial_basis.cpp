/* -*- c++ -*- ----------------------------------------------------------
   Standalone MTP radial basis — no LAMMPS dependency.
   Refactored from lammps-mtp/src/ML-MTP/mtp_radial_basis.cpp
   Original author: Richard Meng, Queen's University at Kingston, 22.11.24
------------------------------------------------------------------------- */

#include "mtp_radial_basis.h"

#include "file_utils.h"

#include <sstream>
#include <stdexcept>
#include <string>

// ---------------------------------------------------------------------------
RadialMTPBasis::RadialMTPBasis(std::istream& is) {
    ::read_basis_properties(is, *this);
}

RadialMTPBasis::~RadialMTPBasis() {
    delete[] radial_basis_vals;
    delete[] radial_basis_ders;
}

// ---------------------------------------------------------------------------
void RBChebyshev::calc_radial_basis(double dist) {
    double ksi = (2 * dist - (min_cutoff + max_cutoff)) / (max_cutoff - min_cutoff);

    radial_basis_vals[0] = scaling * (dist - max_cutoff) * (dist - max_cutoff);
    radial_basis_vals[1] = scaling * (ksi * (dist - max_cutoff) * (dist - max_cutoff));
    for (int i = 2; i < size; i++) {
        radial_basis_vals[i] = 2 * ksi * radial_basis_vals[i - 1] - radial_basis_vals[i - 2];
    }
}

void RBChebyshev::calc_radial_basis_ders(double dist) {
    RBChebyshev::calc_radial_basis(dist);

    double mult = 2.0 / (max_cutoff - min_cutoff);
    double ksi = (2 * dist - (min_cutoff + max_cutoff)) / (max_cutoff - min_cutoff);

    radial_basis_ders[0] = scaling * 2 * (dist - max_cutoff);
    radial_basis_ders[1] =
        scaling * (mult * (dist - max_cutoff) * (dist - max_cutoff) + 2 * ksi * (dist - max_cutoff));
    for (int i = 2; i < size; i++) {
        radial_basis_ders[i] = 2 * (mult * radial_basis_vals[i - 1] + ksi * radial_basis_ders[i - 1]) -
                               radial_basis_ders[i - 2];
    }
}
