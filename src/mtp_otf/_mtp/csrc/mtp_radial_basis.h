/* -*- c++ -*- ----------------------------------------------------------
   Standalone MTP radial basis — no LAMMPS dependency.
   Refactored from lammps-mtp/src/ML-MTP/mtp_radial_basis.h
   Original author: Richard Meng, Queen's University at Kingston, 22.11.24
------------------------------------------------------------------------- */

#pragma once
#include <iosfwd>

void read_basis_properties(std::istream& is, class RadialMTPBasis& basis);

class RadialMTPBasis {
  public:
    // Construct by reading min_dist/max_dist/radial_basis_size from an open stream
    explicit RadialMTPBasis(std::istream& is);
    virtual ~RadialMTPBasis(); // Needed to clear memory

    virtual void calc_radial_basis(double dist) = 0;
    virtual void calc_radial_basis_ders(double dist) = 0;

    int size;             // The size of the radial basis functions
    double min_cutoff;    // Minimum radius value
    double max_cutoff;    // Cutoff radius
    double scaling = 1.0; // All functions are multiplied by scaling

    // Values and derivatives for radial basis functions
    double* radial_basis_vals = nullptr;
    double* radial_basis_ders = nullptr;

    friend void read_basis_properties(std::istream& is, RadialMTPBasis& basis);
};

class RBChebyshev : public RadialMTPBasis {
  public:
    explicit RBChebyshev(std::istream& is) : RadialMTPBasis(is) {}
    void calc_radial_basis(double val) override;
    void calc_radial_basis_ders(double val) override;
};
