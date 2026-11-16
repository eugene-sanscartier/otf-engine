/* -*- c++ -*- ----------------------------------------------------------
   The base of PairMTP and PairETN, as LAMMPS's Pair is of its pair styles:
   the members they share, and what the trainers and the selection call.
------------------------------------------------------------------------- */

#pragma once

#include "mtp_radial_basis.h"
#include "neigh_list.h"

#include <algorithm>
#include <array>
#include <string>
#include <vector>

class TextFileReader;

class Pair {
  public:
    virtual ~Pair() { delete radial_basis; }

    // Energy, per-atom energies, forces and virial of the central atoms in list,
    // left in the members below.
    virtual void compute(const NeighList& list) = 0;

    double get_energy() const { return energy; }
    const std::vector<double>& get_eatom() const { return eatom; }       // [n_atoms]
    const std::vector<double>& get_forces() const { return forces; }     // [n_atoms * 3]
    const std::array<double, 6>& get_virial() const { return virial; }  // xx,yy,zz,xy,xz,yz

    // The coefficients as one vector, of coeff_count(), in the order the potential defines.
    virtual int coeff_count() const = 0;
    virtual void get_coeffs(double* c) const = 0;
    virtual void set_coeffs(const double* c) = 0;

    // d(E_i, F, virial)/dc. force_grad and virial_grad may be null, and virial_grad
    // is filled only with force_grad. Throws unless PairMTPGrad or PairETNGrad implements it.
    //   site_energy_grad : [inum * coeff_count]      per-atom dE_i/dc
    //   force_grad       : [n_atoms * 3 * coeff_count]
    //   virial_grad      : [6 * coeff_count]
    virtual void eval_grad(const NeighList& list, double* site_energy_grad, double* force_grad, double* virial_grad);

    // Adds dL/dc to loss_grad [coeff_count], for a loss L whose derivatives
    // with respect to the energy, forces and virial of this structure are
    // given; dloss_dforces [n_atoms * 3] and dloss_dvirial [6] may be null.
    // Throws unless PairMTPGrad or PairETNGrad implements it.
    virtual void eval_loss_grad(const NeighList& list, double dloss_denergy, const double* dloss_dforces, const double* dloss_dvirial, double* loss_grad);

    // Radial basis values and derivatives at dist, into arrays of radial_basis_size.
    void eval_radial_basis(double dist, double* vals_out, double* ders_out);

    int get_species_count() const { return species_count; }
    int get_radial_basis_size() const { return radial_basis_size; }
    double get_min_cutoff() const { return min_cutoff; }
    double get_max_cutoff() const { return max_cutoff; }
    double get_scaling() const { return scaling; }
    // False for a file without coefficients, which holds the reference code's initial values.
    bool is_trained() const { return trained; }
    const std::string& get_potential_name() const { return potential_name; }
    const std::string& get_radial_basis_type() const { return radial_basis_type; }
    const std::vector<double>& get_species_coeffs() const { return species_coeffs; }    // [species_count]

    void set_species_coeffs(const double* c) { std::copy(c, c + species_count, species_coeffs.begin()); }
    void set_scaling(double s) { scaling = s; radial_basis->scaling = s; }
    void set_trained(bool t) { trained = t; }

  protected:
    // Reads radial_basis_type at the current line of tfr and the basis it names, RBChebyshev or RBCinf,
    // and sets the radial basis members from it; scaling must be read first.
    void read_radial_basis(TextFileReader& tfr);

    std::string potential_name = "Untitled";
    std::string radial_basis_type;

    int species_count = 0;
    double scaling = 1.0;
    bool trained = true;

    // Radial basis
    RadialMTPBasis* radial_basis = nullptr;
    int radial_basis_size = 0;           // Number of elements in bases

    // One cutoff set for all species combinations
    double min_cutoff = 0.0;
    double max_cutoff = 0.0;
    double max_cutoff_sq = 0.0;

    std::vector<double> species_coeffs;  // [species_count] site energy offsets

    // Results of compute
    double energy = 0.0;
    std::vector<double> eatom;   // [n_atoms] site energies
    std::vector<double> forces;  // [n_atoms * 3]
    std::array<double, 6> virial{};    // xx,yy,zz,xy,xz,yz

    // Cache values between forwards and backwards pass
    int cache_size = 0;
    std::vector<int> cached_j;                    // [cache_size]
    std::vector<std::array<double, 3>> valid_dr;  // [cache_size] displacement of each cached neighbor
    int cache_stride = 0;                         // values per cached neighbor
    std::vector<double> neighbor_cache;           // [cache_size * cache_stride] 1/dist, then the potential's per-neighbor values
};
