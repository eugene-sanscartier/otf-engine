/* -*- c++ -*- ----------------------------------------------------------
   Pair, the base of PairMTP and PairETN: their shared members, and the
   methods the fits and the selection call.
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

// Central atoms per chunk: 8 times the doubles of the widest SIMD register the build targets.
#if defined(__AVX512F__)
constexpr int chunk_size = 64;
#elif defined(__AVX__)
constexpr int chunk_size = 32;
#else
constexpr int chunk_size = 16;
#endif

// One value per central atom of a chunk. Its operators apply the arithmetic of a double atom by atom.
// An update such as x += s * y * z runs as one loop over the atoms.
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

// Products of Lanes, which operator+= and the Lanes constructor evaluate
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

// Returns the sum of x * y over the atoms of the chunk, in an order the compiler may vectorize (-fopenmp-simd).
inline double sum_of_products(const Lanes& x, const Lanes& y) {
    double sum = 0.0;
#pragma omp simd reduction(+ : sum)
    for (int a = 0; a < chunk_size; a++) sum += x.v[a] * y.v[a];
    return sum;
}

// Returns the sum of x over the atoms of the chunk, in an order the compiler may vectorize (-fopenmp-simd).
inline double sum_over_atoms(const Lanes& x) {
    double sum = 0.0;
#pragma omp simd reduction(+ : sum)
    for (int a = 0; a < chunk_size; a++) sum += x.v[a];
    return sum;
}

class Pair {
  public:
    virtual ~Pair() { delete radial_basis; }

    // Computes the energy, site energies, forces and virial of the central atoms in list into the members below.
    virtual void compute(const NeighList& list) = 0;

    double get_energy() const { return energy; }
    const std::vector<double>& get_eatom() const { return eatom; }       // [n_atoms]
    const std::vector<double>& get_forces() const { return forces; }     // [n_atoms * 3]
    const std::array<double, 6>& get_virial() const { return virial; }  // xx,yy,zz,xy,xz,yz

    // The coefficients as one vector of coeff_count(), in the order of the potential.
    virtual int coeff_count() const = 0;
    virtual void get_coeffs(double* c) const = 0;
    virtual void set_coeffs(const double* c) = 0;

    // Computes d(E_i, F, virial)/dc. Throws unless MTP or ETN overrides it.
    //   site_energy_grad : [inum * coeff_count] dE_i/dc of each central atom
    //   force_grad       : [n_atoms * 3 * coeff_count], or null
    //   virial_grad      : [6 * coeff_count], or null; filled only with force_grad
    virtual void eval_grad(const NeighList& list, double* site_energy_grad, double* force_grad, double* virial_grad);

    // Adds dL/dc to loss_grad [coeff_count], given the derivatives of the loss L with respect to the energy, forces
    // and virial of the structure. dloss_dforces [n_atoms * 3] and dloss_dvirial [6] may be null.
    // Throws unless MTP or ETN overrides it.
    virtual void eval_loss_grad(const NeighList& list, double dloss_denergy, const double* dloss_dforces, const double* dloss_dvirial, double* loss_grad);

    // Writes the radial basis values and derivatives at dist into arrays of radial_basis_size.
    void eval_radial_basis(double dist, double* vals_out, double* ders_out);

    int get_species_count() const { return species_count; }
    int get_radial_basis_size() const { return radial_basis_size; }
    double get_min_cutoff() const { return min_cutoff; }
    double get_max_cutoff() const { return max_cutoff; }
    double get_scaling() const { return scaling; }
    // False for a potential without fitted coefficients, which holds initial values.
    bool is_trained() const { return trained; }
    const std::string& get_potential_name() const { return potential_name; }
    const std::string& get_radial_basis_type() const { return radial_basis_type; }
    const std::vector<std::string>& get_species() const { return species; }
    const std::vector<double>& get_species_coeffs() const { return species_coeffs; }    // [species_count]

    void set_species(const std::vector<std::string>& s) { species = s; }
    void set_species_coeffs(const double* c) { std::copy(c, c + species_count, species_coeffs.begin()); }
    void set_scaling(double s) { scaling = s; radial_basis->scaling = s; }
    // Rebuilds the radial basis at the minimum distance d.
    void set_min_cutoff(double d) { set_radial_basis(radial_basis_type, d, max_cutoff, radial_basis_size); }
    void set_trained(bool t) { trained = t; }

  protected:
    Pair() = default;
    // Starts a potential of species_count species, where 0 leaves the count to training.
    // Builds its radial basis of type radial_basis_type from min_dist, max_dist and radial_basis_size.
    Pair(int species_count, const std::string& radial_basis_type, double min_dist, double max_dist, int radial_basis_size);

    // Reads the first lines of a file: model (MTP, ETN) and version, then the optional potential_name, scaling,
    // species_count and potential_tag. Leaves tfr at the next line.
    void read_header(TextFileReader& tfr, const std::string& model, const std::string& version);
    // Reads radial_basis_type at the current line of tfr, then the cutoffs and size below it, and sets the radial basis.
    void read_radial_basis(TextFileReader& tfr);
    // Builds a radial basis of type RBChebyshev or RBCinf with the current scaling, and sets the radial basis
    // members from it.
    void set_radial_basis(const std::string& type, double min_dist, double max_dist, int size);

    std::string potential_name = "Untitled";
    std::string radial_basis_type;
    std::vector<std::string> species;    // the element of each species index

    int species_count = 0;
    double scaling = 1.0;
    bool trained = true;

    RadialMTPBasis* radial_basis = nullptr;
    int radial_basis_size = 0;

    // cutoffs shared by every species pair
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

    // Grows the neighbor cache to chunk_size blocks, each of the largest neighbor count in list.
    void grow_chunk_cache(const NeighList& list) {
        atom_cache_size = 0;
        for (int ii = 0; ii < list.inum; ii++) atom_cache_size = std::max(atom_cache_size, list.numneigh[ii]);
        grow_cache(chunk_size * atom_cache_size);
    }

    // Computes the radial basis at the neighbors within the cutoff, in order, into the rows of radial_basis, and
    // their unit vectors into neighbor_units.
    // dr: [jnum * 3] displacements of the jnum neighbors of an atom. neighbor_units: [3 * jnum], a row per x, y, z.
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

    // Points cached_j, valid_dr and neighbor_cache at the block of atom a of the chunk.
    void select_atom_cache(int a) {
        const size_t first = (size_t) a * atom_cache_size;
        cached_j = cached_j_storage.data() + first;
        valid_dr = valid_dr_storage.data() + first;
        neighbor_cache = neighbor_cache_storage.data() + first * cache_stride;
    }

    // Neighbor cache between the forwards and backwards passes. cached_j, valid_dr and neighbor_cache point into its storage.
    int cache_size = 0;
    int atom_cache_size = 0;                              // neighbors per atom block of a chunk
    std::vector<int> cached_j_storage;                    // [cache_size]
    std::vector<std::array<double, 3>> valid_dr_storage;  // [cache_size]
    std::vector<double> neighbor_cache_storage;           // [cache_size * cache_stride]
    int* cached_j = nullptr;
    std::array<double, 3>* valid_dr = nullptr;  // displacement of each cached neighbor
    int cache_stride = 0;
    double* neighbor_cache = nullptr;           // [cache_stride] per neighbor: 1/dist, then the values the potential caches
    std::array<int, chunk_size> cached_counts{};  // number of cached neighbors of each atom of the chunk

    // distances [jnum] and unit vectors [3 * jnum] of the neighbors within the cutoff, from calc_neighbor_rows
    std::vector<double> neighbor_dists, neighbor_units;
};
