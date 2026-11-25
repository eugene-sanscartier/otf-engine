/* -*- c++ -*- ----------------------------------------------------------
   The MTP radial basis.
   Ported from lammps-mtp/src/ML-MTP/mtp_radial_basis.h
   Original author: Richard Meng, Queen's University at Kingston, 22.11.24
------------------------------------------------------------------------- */

#pragma once

#include <vector>

class RadialMTPBasis {
  public:
    // Sets up a basis of size functions on [min_cutoff, max_cutoff].
    RadialMTPBasis(double min_cutoff, double max_cutoff, int size);
    virtual ~RadialMTPBasis() = default;

    virtual void calc_radial_basis_ders(double dist) = 0;
    // Computes the values and derivatives at count distances into rows_vals and rows_ders, a row of count per function.
    virtual void calc_radial_basis_rows(int count, const double* dist);
    // Copies distance k of the rows into radial_basis_vals and radial_basis_ders.
    void select_row(int k) {
        for (int i = 0; i < size; i++) {
            radial_basis_vals[i] = rows_vals[(size_t) i * row_count + k];
            radial_basis_ders[i] = rows_ders[(size_t) i * row_count + k];
        }
    }

    int size = 0;          // number of radial functions
    double min_cutoff = 0;
    double max_cutoff = 0;
    double scaling = 1.0;  // factor on every radial function

    // [size] values and derivatives at the last distance of calc_radial_basis_ders
    std::vector<double> radial_basis_vals;
    std::vector<double> radial_basis_ders;

    // from the last calc_radial_basis_rows: its count of distances, and the values and derivatives [size * row_count]
    int row_count = 0;
    std::vector<double> rows_vals, rows_ders;
};

// phi_0 = (r - r_cut)^2, phi_1 = ksi (r - r_cut)^2, then phi_i = 2 ksi phi_{i-1} - phi_{i-2}, all times scaling.
// ksi = (2 r - (min_cutoff + max_cutoff)) / (max_cutoff - min_cutoff).
class RBChebyshev : public RadialMTPBasis {
  public:
    RBChebyshev(double min_cutoff, double max_cutoff, int size) : RadialMTPBasis(min_cutoff, max_cutoff, size) {}
    void calc_radial_basis_ders(double dist) override;
};
