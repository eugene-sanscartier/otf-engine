/* -*- c++ -*- ----------------------------------------------------------
   Standalone MTP radial basis — no LAMMPS dependency.
   Ported from lammps-mtp/src/ML-MTP/mtp_radial_basis.h
   Original author: Richard Meng, Queen's University at Kingston, 22.11.24
------------------------------------------------------------------------- */

#pragma once

#include <string>
#include <vector>

class TextFileReader;

class RadialMTPBasis {
  public:
    // Reads scaling/min_dist/max_dist/radial_basis_size from the current position.
    explicit RadialMTPBasis(TextFileReader& tfr);
    virtual ~RadialMTPBasis() = default;

    virtual void calc_radial_basis_ders(double dist) = 0;

    int size = 0;          // The size of the radial basis functions
    double min_cutoff = 0; // Minimum radius value
    double max_cutoff = 0; // Cutoff radius
    double scaling = 1.0;  // All radial functions are multiplied by scaling

    // Values and derivatives for radial basis functions
    std::vector<double> radial_basis_vals;
    std::vector<double> radial_basis_ders;

  private:
    // Reads the basis properties (cutoffs and size), not the radial parameters
    void read_basis_properties(TextFileReader& tfr);
};

// phi_0 = (r - r_cut)^2, phi_1 = ksi (r - r_cut)^2, Chebyshev recursion above.
class RBChebyshev : public RadialMTPBasis {
  public:
    explicit RBChebyshev(TextFileReader& tfr) : RadialMTPBasis(tfr) {}
    void calc_radial_basis_ders(double dist) override;
};
