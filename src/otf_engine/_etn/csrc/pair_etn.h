/* -*- c++ -*- ----------------------------------------------------------
   Standalone ETN potential — no LAMMPS dependency.
   Computes mlip-4's ETN (src/pots/etn/etn.cpp at 79a39bf), which has no
   LAMMPS pair style; hand-written passes replace mlip-4's program.
   Original author of mlip-4's ETN: Max Hodapp
------------------------------------------------------------------------- */

#pragma once

#include "etn_harmonics.h"
#include "pair.h"

#include <algorithm>
#include <array>
#include <istream>
#include <string>
#include <vector>

// Layout of a covariant vector with ranks[l] channels per l: entry (l, m, n) at
// offsets[l] + (m + l) * ranks[l] + n.
struct CovariantShape {
    std::vector<int> ranks;
    std::vector<int> offsets;
    int size = 0;

    explicit CovariantShape(std::vector<int> ranks);
    int index(int l, int m) const { return offsets[l] + (m + l) * ranks[l]; }
};

// One (l1, l2, l3) block of a tensor-train core.
// Coefficient (n1, n2, n3) is at base + n3 * stride3 + n2 * stride2 + n1.
// wigner[w_begin, w_end) holds the 3-j symbols of the block.
// The products of the block start at products_offset within those of the core.
struct CoreBlock {
    int l1, l2, l3;
    int base, stride2, stride3;
    int w_begin, w_end;
    int products_offset;
};

// The real or imaginary part of core i, which couples bond i and the input into bond i + 1.
// coeffs is ordered l3, n3, l2, n2, l1, n1, and products_size is the total size of the block products.
struct Core {
    std::vector<CoreBlock> blocks;
    std::vector<double> coeffs;
    int products_size = 0;
};

// One term of core i: its real or imaginary part applied to part in_part of bond i and added into part out_part of bond i + 1.
// Part 0 is the real part and part 1 the imaginary part. The products of the term start at products_offset in core_products.
struct CoreTerm {
    bool imaginary;
    int in_part, out_part;
    int products_offset;
};

// ETN potential: energy, forces and virial of a configuration.
class PairETN : public Pair {
  public:
    // Reads an ETN potential file (version 1.0.0).
    explicit PairETN(const std::string& filename);
    // Builds the untrained ETN of this shape, equal to the one read from a file that stops after bond_ranks.
    // With species_count 0, training sets the species count.
    PairETN(std::vector<int> species_ranks, std::vector<int> radial_func_counts, std::vector<int> radial_ranks, std::vector<std::vector<int>> bond_ranks, int species_count, const std::string& radial_basis_type, double min_dist, double max_dist, int radial_basis_size);

    // Computes the energy, per-atom energies, forces and virial of the central atoms in list into energy, eatom, forces and virial.
    void compute(const NeighList& list) override;

    // Layout of bond b, and the offset of its real part within the bond vectors of one atom (ETN::eval_bonds).
    // The imaginary part follows the real part, and is zero outside bonds 2 .. d-2.
    const CovariantShape& get_bond_shape(int b) const { return bond_shapes[b]; }
    int get_bond_offset(int b) const { return bond_offsets[b]; }
    int get_bonds_size() const { return bonds_size; }
    const CovariantShape& get_input_shape() const { return input_shape; }

    // Length of the coefficient vector.
    int coeff_count() const override;
    // Copies the coefficients into c: B_l per l, A_l per l, the real core parts, the imaginary core parts, the site energies.
    void get_coeffs(double* c) const override;
    void set_coeffs(const double* c) override;

    int get_core_count() const { return (int) bond_ranks.size() - 1; }
    bool has_imaginary_cores() const { return get_core_count() > 3; }
    const std::vector<int>& get_species_ranks() const { return species_ranks; }
    const std::vector<int>& get_radial_func_counts() const { return radial_func_counts; }
    const std::vector<int>& get_radial_ranks() const { return radial_ranks; }
    const std::vector<std::vector<int>>& get_bond_ranks() const { return bond_ranks; }
    // radial_basis_coeffs[l]: [species_rank * radial_func_count * radial_rank], column-major (k, n', n)
    const std::vector<double>& get_radial_basis_coeffs(int l) const { return radial_basis_coeffs[l]; }
    // species_pair_coeffs[l]: [species_count^2 * species_rank], row-major (s_i, s_j, k)
    const std::vector<double>& get_species_pair_coeffs(int l) const { return species_pair_coeffs[l]; }
    // The coefficients of the real or imaginary part of core i, ordered l3, n3, l2, n2, l1, n1
    const std::vector<double>& get_core_coeffs(int i, bool imaginary) const { return (imaginary ? cores_im[i] : cores[i]).coeffs; }

    void set_radial_basis_coeffs(int l, const double* c);
    void set_species_pair_coeffs(int l, const double* c);
    void set_core_coeffs(int i, bool imaginary, const double* c);
    // Resizes the coefficients to n species, keeping the A_l entries and site energies of the remaining species.
    // The new entries are zero until the caller sets them.
    void set_species_count(int n);
    // Sets the k-th coefficient to sin(10 + k^2).
    void reset_coeffs();

  protected:
    // Computes the radial rows of Pair, and the harmonic rows at the unit vectors of the neighbors.
    void calc_neighbor_rows(int jnum, const double* dr) override {
        Pair::calc_neighbor_rows(jnum, dr);
        const double* units = neighbor_units.data();
        harmonics.compute_rows(radial_basis->row_count, units, units + jnum, units + 2 * jnum);
    }

    void read_file(std::istream& is);
    void prepare_cores();
    Core prepare_core(int i, bool imaginary);
    void prepare_columns();
    void prepare_radial_mixing();

    // Shape: the species rank, radial function count and radial rank per l of the input;
    // the rank per l of each bond, {1} at both ends
    std::vector<int> species_ranks;
    std::vector<int> radial_func_counts;
    std::vector<int> radial_ranks;
    std::vector<std::vector<int>> bond_ranks;

    // Coefficients in the get_coeffs order: these two, then the cores, then species_coeffs
    std::vector<std::vector<double>> radial_basis_coeffs;        // [l] B_l
    std::vector<std::vector<double>> species_pair_coeffs;  // [l] A_l

    // First column of each coefficient block in the get_coeffs order
    std::vector<int> radial_columns;        // [l]
    std::vector<int> species_pair_columns;  // [l]
    std::vector<int> core_columns;          // [core]
    std::vector<int> core_im_columns;       // [core]
    int species_columns = 0;

    // Tensor-train layout
    std::vector<Core> cores;                // [core_count]
    std::vector<Core> cores_im;             // [core_count]
    std::vector<std::vector<CoreTerm>> core_terms;  // [core_count] the terms of each core
    std::vector<WignerEntry> wigner;        // the 3-j symbols the blocks point into
    CovariantShape input_shape{{1}};        // v, with the constant 1 ahead of the l = 0 channels
    std::vector<CovariantShape> bond_shapes;
    std::vector<int> bond_offsets;          // [core_count + 1] offset of the real part of bond b in the bond vectors of one atom
    int bonds_size = 0;
    std::vector<int> mixed_offsets;         // [l_count + 1] first radial channel of each l

    // S_l = A_l B_l for each species pair: [l][(s_i * species_count + s_j) * N_rad * R_rad + n' * R_rad + n]
    std::vector<std::vector<double>> radial_mixing;

    RealHarmonics harmonics{1};

    // Tensor-train values and derivatives for the chunk of central atoms in compute, one lane per atom
    std::vector<Lanes> input_vals;         // [input_shape.size] the input vector v, the constant 1 first
    std::vector<Lanes> bond_vals;          // [bonds_size] the bond vectors, bond 0 = 1
    std::vector<Lanes> input_energy_ders;  // [input_shape.size] dE_i/dv
    std::vector<Lanes> bond_energy_ders;   // [bonds_size] dE_i/dbond

    // Products of each block of each core term with the left bond, sum_n1 c[n3, n2, n1] u[m1, n1].
    // A block starts at the products_offset of its term plus its own, and is indexed [(m1 + l1) * n3_count * n2_count + n3 * n2_count + n2].
    std::vector<Lanes> core_products;
    // Derivatives of the products of one block, in the same layout, sized for the largest block
    std::vector<Lanes> core_product_ders;

    // The same for the kernels that take one atom at a time
    std::vector<double> atom_input_vals;
    std::vector<double> atom_bond_vals;
    std::vector<double> atom_input_energy_ders;
    std::vector<double> atom_bond_energy_ders;
    std::vector<double> atom_core_products;
    std::vector<double> atom_core_product_ders;

    // neighbor_cache holds per neighbor 1/dist, Y, grad Y, the radial channels and their derivatives.
    // cache_stride is 1 + 4 * l_count^2 + 2 * radial-rank channels
};
