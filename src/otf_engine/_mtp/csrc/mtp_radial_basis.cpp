/* -*- c++ -*- ----------------------------------------------------------
   Standalone MTP radial basis — no LAMMPS dependency.
   Ported from lammps-mtp/src/ML-MTP/mtp_radial_basis.cpp
   Original author: Richard Meng, Queen's University at Kingston, 22.11.24
------------------------------------------------------------------------- */

#include "mtp_radial_basis.h"

#include "text_file_reader.h"

#include <algorithm>
#include <stdexcept>

RadialMTPBasis::RadialMTPBasis(TextFileReader& tfr) {
    read_basis_properties(tfr);
}

void RadialMTPBasis::read_basis_properties(TextFileReader& tfr) {
    tfr.advance();

    // Optional scaling line
    if (tfr.keyword() == "scaling") {
        tfr.rest() >> scaling;
        tfr.advance();
    }

    // Lower cutoff — accepts both 'min_dist' and 'min_val'
    if (tfr.keyword() != "min_val" && tfr.keyword() != "min_dist")
        throw std::runtime_error("MTP radial basis: expected min_dist, got '" + tfr.keyword() + "'");
    tfr.rest() >> min_cutoff;

    // Upper cutoff — accepts both 'max_dist' and 'max_val'
    tfr.advance();
    if (tfr.keyword() != "max_val" && tfr.keyword() != "max_dist")
        throw std::runtime_error("MTP radial basis: expected max_dist, got '" + tfr.keyword() + "'");
    tfr.rest() >> max_cutoff;

    size = tfr.next_int("radial_basis_size");

    radial_basis_vals.resize(size);
    radial_basis_ders.resize(size);
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

// ---------------------------------------------------------------------------
void BChebyshev::calc_radial_basis(double dist) {
    double ksi = (2 * dist - (min_cutoff + max_cutoff)) / (max_cutoff - min_cutoff);

    radial_basis_vals[0] = scaling * 1;
    radial_basis_vals[1] = scaling * ksi;
    for (int i = 2; i < size; i++) {
        radial_basis_vals[i] = 2 * ksi * radial_basis_vals[i - 1] - radial_basis_vals[i - 2];
    }
}

void BChebyshev::calc_radial_basis_ders(double dist) {
    BChebyshev::calc_radial_basis(dist);

    double mult = 2.0 / (max_cutoff - min_cutoff);
    double ksi = (2 * dist - (min_cutoff + max_cutoff)) / (max_cutoff - min_cutoff);

    radial_basis_ders[0] = scaling * 0;
    radial_basis_ders[1] = scaling * mult;
    for (int i = 2; i < size; i++) {
        radial_basis_ders[i] = 2 * (mult * radial_basis_vals[i - 1] + ksi * radial_basis_ders[i - 1]) -
                               radial_basis_ders[i - 2];
    }
}

// ---------------------------------------------------------------------------
void RBTaylor::calc_radial_basis(double dist) {
    radial_basis_vals[0] = scaling * 1;
    for (int i = 1; i < size; i++) {
        radial_basis_vals[i] = scaling * dist * radial_basis_vals[i - 1];
    }
}

void RBTaylor::calc_radial_basis_ders(double dist) {
    RBTaylor::calc_radial_basis(dist);

    radial_basis_ders[0] = scaling * 0;
    for (int i = 1; i < size; i++) {
        radial_basis_ders[i] = scaling * i * radial_basis_vals[i - 1];
    }
}

// ---------------------------------------------------------------------------
void RBChebyshevRepuls::calc_radial_basis(double dist) {
    RBChebyshev::calc_radial_basis(std::max(dist, min_cutoff));
}

void RBChebyshevRepuls::calc_radial_basis_ders(double dist) {
    RBChebyshev::calc_radial_basis_ders(std::max(dist, min_cutoff));
    if (dist <= min_cutoff)
        std::fill(radial_basis_ders.begin(), radial_basis_ders.end(), 0.0);
}

void BChebyshevRepuls::calc_radial_basis(double dist) {
    BChebyshev::calc_radial_basis(std::max(dist, min_cutoff));
}

void BChebyshevRepuls::calc_radial_basis_ders(double dist) {
    BChebyshev::calc_radial_basis_ders(std::max(dist, min_cutoff));
    if (dist <= min_cutoff)
        std::fill(radial_basis_ders.begin(), radial_basis_ders.end(), 0.0);
}
