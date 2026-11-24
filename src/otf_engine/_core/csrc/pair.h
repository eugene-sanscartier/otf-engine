/* -*- c++ -*- ----------------------------------------------------------
   The base of PairMTP and PairETN, as LAMMPS's Pair is of its pair styles:
   the members they share, and what the fits and the selection call.
------------------------------------------------------------------------- */

#pragma once

#include "mtp_radial_basis.h"
#include "neigh_list.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <vector>

class TextFileReader;

// Central atoms per chunk: 8 vectors of the widest SIMD registers the build targets.
#if defined(__AVX512F__)
constexpr int chunk_size = 64;
#elif defined(__AVX__)
constexpr int chunk_size = 32;
#else
constexpr int chunk_size = 16;
#endif

// One value per central atom of a chunk, with the arithmetic of a double applied atom by atom. An operation that
// combines products, as x += s * y * z, runs as one loop over the atoms, as the expression for one atom would.
struct ScaledLanes;
struct alignas(64) Lanes {
    double v[chunk_size] = {};

    Lanes() = default;
    Lanes(const ScaledLanes& e);    // w = s * x

    double& operator[](int a) { return v[a]; }
    double operator[](int a) const { return v[a]; }
    Lanes& operator=(double s) {
        std::fill(v, v + chunk_size, s);
        return *this;
    }
};

// The products a Lanes expression is made of, evaluated when added
struct ScaledLanes {
    double s;
    const Lanes& x;
};
struct LanesProduct {
    const Lanes& x;
    const Lanes& y;
};
struct ScaledLanesProduct {
    double s;
    const Lanes& x;
    const Lanes& y;
};

inline Lanes::Lanes(const ScaledLanes& e) {
    for (int a = 0; a < chunk_size; a++) v[a] = e.s * e.x.v[a];
}

inline ScaledLanes operator*(double s, const Lanes& x) { return {s, x}; }
inline ScaledLanes operator*(const Lanes& x, double s) { return {s, x}; }
inline LanesProduct operator*(const Lanes& x, const Lanes& y) { return {x, y}; }
inline ScaledLanesProduct operator*(const ScaledLanes& sx, const Lanes& y) { return {sx.s, sx.x, y}; }

inline Lanes& operator+=(Lanes& out, const Lanes& x) {
    for (int a = 0; a < chunk_size; a++) out.v[a] += x.v[a];
    return out;
}
inline Lanes& operator+=(Lanes& out, const ScaledLanes& e) {
    for (int a = 0; a < chunk_size; a++) out.v[a] += e.s * e.x.v[a];
    return out;
}
inline Lanes& operator+=(Lanes& out, const LanesProduct& e) {
    for (int a = 0; a < chunk_size; a++) out.v[a] += e.x.v[a] * e.y.v[a];
    return out;
}
inline Lanes& operator+=(Lanes& out, const ScaledLanesProduct& e) {
    for (int a = 0; a < chunk_size; a++) out.v[a] += e.s * e.x.v[a] * e.y.v[a];
    return out;
}

// The sum of x * y over the chunk's atoms, in an order the compiler may vectorize (-fopenmp-simd).
inline double sum_of_products(const Lanes& x, const Lanes& y) {
    double sum = 0.0;
#pragma omp simd reduction(+ : sum)
    for (int a = 0; a < chunk_size; a++) sum += x.v[a] * y.v[a];
    return sum;
}

// The sum of x over the chunk's atoms, in an order the compiler may vectorize (-fopenmp-simd).
inline double sum_over_atoms(const Lanes& x) {
    double sum = 0.0;
#pragma omp simd reduction(+ : sum)
    for (int a = 0; a < chunk_size; a++) sum += x.v[a];
    return sum;
}

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
    // is filled only with force_grad. Throws unless MTP or ETN implements it.
    //   site_energy_grad : [inum * coeff_count]      per-atom dE_i/dc
    //   force_grad       : [n_atoms * 3 * coeff_count]
    //   virial_grad      : [6 * coeff_count]
    virtual void eval_grad(const NeighList& list, double* site_energy_grad, double* force_grad, double* virial_grad);

    // Adds dL/dc to loss_grad [coeff_count], for a loss L whose derivatives
    // with respect to the energy, forces and virial of this structure are
    // given; dloss_dforces [n_atoms * 3] and dloss_dvirial [6] may be null.
    // Throws unless MTP or ETN implements it.
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
    // Sets the minimum distance of the radial basis.
    void set_min_cutoff(double d) { min_cutoff = d; radial_basis->min_cutoff = d; }
    void set_trained(bool t) { trained = t; }

  protected:
    Pair() = default;
    // The start of a potential built from its shape: species_count species, 0 leaving the count to training, and the
    // radial basis radial_basis_type names from its cutoffs and size.
    Pair(int species_count, const std::string& radial_basis_type, double min_dist, double max_dist, int radial_basis_size);

    // Reads a file's first lines: model (MTP, ETN) and version, then potential_name, scaling and species_count,
    // each optional, and leaves tfr at the next line.
    void read_header(TextFileReader& tfr, const std::string& model, const std::string& version);
    // Reads radial_basis_type at the current line of tfr and the cutoffs and size below it, and sets the basis.
    void read_radial_basis(TextFileReader& tfr);
    // Builds the radial basis type names, RBChebyshev or RBCinf, and sets the radial basis members
    // from it; scaling must be set first.
    void set_radial_basis(const std::string& type, double min_dist, double max_dist, int size);

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

    // Grows the neighbor cache to hold jnum neighbors, and points cached_j, valid_dr and neighbor_cache at its start.
    void grow_cache(int jnum) {
        if (cache_size < jnum) {
            cached_j_storage.resize(jnum);
            valid_dr_storage.resize(jnum);
            neighbor_cache_storage.resize((size_t) jnum * cache_stride);
            cache_size = jnum;
        }
        select_atom_cache(0);
    }

    // Grows the neighbor cache to a block for each atom of a chunk, each as large as list's largest neighbor count.
    void grow_chunk_cache(const NeighList& list) {
        atom_cache_size = 0;
        for (int ii = 0; ii < list.inum; ii++) atom_cache_size = std::max(atom_cache_size, list.numneigh[ii]);
        grow_cache(chunk_size * atom_cache_size);
    }

    // The radial basis at each of an atom's jnum neighbors at displacements dr within the cutoff, in order, in
    // radial_basis's rows, and their unit vectors in neighbor_units, x, y and z each in a row of jnum.
    virtual void calc_neighbor_rows(int jnum, const double* dr) {
        neighbor_dists.resize(jnum);
        neighbor_units.resize(3 * (size_t) jnum);
        int count = 0;
        for (int jj = 0; jj < jnum; jj++) {
            const double r[3] = {dr[jj * 3 + 0], dr[jj * 3 + 1], dr[jj * 3 + 2]};
            const double rsq = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
            if (rsq > max_cutoff_sq) continue;
            const double dist = std::sqrt(rsq);
            const double inv_dist = 1.0 / dist;
            neighbor_dists[count] = dist;
            for (int a = 0; a < 3; a++) neighbor_units[(size_t) a * jnum + count] = r[a] * inv_dist;
            count++;
        }
        radial_basis->calc_radial_basis_rows(count, neighbor_dists.data());
    }

    // Points cached_j, valid_dr and neighbor_cache at the block of the chunk's atom a.
    void select_atom_cache(int a) {
        const size_t first = (size_t) a * atom_cache_size;
        cached_j = cached_j_storage.data() + first;
        valid_dr = valid_dr_storage.data() + first;
        neighbor_cache = neighbor_cache_storage.data() + first * cache_stride;
    }

    // Cache values between forwards and backwards pass, in storage that cached_j, valid_dr and neighbor_cache point into
    int cache_size = 0;
    int atom_cache_size = 0;                              // neighbors per atom block, for compute's chunk
    std::vector<int> cached_j_storage;                    // [cache_size]
    std::vector<std::array<double, 3>> valid_dr_storage;  // [cache_size]
    std::vector<double> neighbor_cache_storage;           // [cache_size * cache_stride]
    int* cached_j = nullptr;                    // the current atom's cached neighbors
    std::array<double, 3>* valid_dr = nullptr;  // displacement of each cached neighbor
    int cache_stride = 0;                       // values per cached neighbor
    double* neighbor_cache = nullptr;           // [cache_stride] per neighbor: 1/dist, then the potential's per-neighbor values
    std::array<int, chunk_size> cached_counts{};  // compute's chunk: the neighbors each atom has cached

    // calc_neighbor_rows's distances [jnum] and unit vectors [3 * jnum] of the neighbors within the cutoff
    std::vector<double> neighbor_dists, neighbor_units;
};
