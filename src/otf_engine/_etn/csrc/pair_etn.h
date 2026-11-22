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

// One (l1, l2, l3) block of a tensor-train core: coefficient (n1, n2, n3) sits at
// base + n3 * stride3 + n2 * stride2 + n1, wigner[w_begin, w_end) holds its 3-j symbols,
// and its products start at products_offset within its core's.
struct CoreBlock {
    int l1, l2, l3;
    int base, stride2, stride3;
    int w_begin, w_end;
    int products_offset;
};

// The real or the imaginary part of one core, between bond i, the input and bond i + 1: its blocks, its
// coefficients, ordered l3, n3, l2, n2, l1, n1, and the size of its blocks' products.
struct Core {
    std::vector<CoreBlock> blocks;
    std::vector<double> coeffs;
    int products_size = 0;
};

// One term of a core: its real or imaginary part applied to one part (0 real, 1 imaginary) of
// bond i, added into one part of bond i + 1; its products start at products_offset in core_products.
struct CoreTerm {
    bool imaginary;
    int in_part, out_part;
    int products_offset;
};

// The members PairETN shares with PairMTP are Pair's.
class PairETN : public Pair {
  public:
    // Reads an ETN potential file (version 1.0.0).
    explicit PairETN(const std::string& filename);
    // The untrained ETN of this shape, as a file that stops after bond_ranks gives it;
    // species_count 0 leaves the count to training.
    PairETN(std::vector<int> species_ranks, std::vector<int> radial_func_counts, std::vector<int> radial_ranks, std::vector<std::vector<int>> bond_ranks, int species_count, const std::string& radial_basis_type, double min_dist, double max_dist, int radial_basis_size);

    // Energy, per-atom energies, forces and virial of the central atoms in list,
    // left in the members below.
    void compute(const NeighList& list) override;

    // Bond b of the tensor train: its layout, and the offset of its real part within one atom's
    // bond vectors (ETN::eval_bonds); the imaginary part follows it, and is zero outside bonds 2 .. d-2.
    const CovariantShape& get_bond_shape(int b) const { return bond_shapes[b]; }
    int get_bond_offset(int b) const { return bond_offsets[b]; }
    int get_bonds_size() const { return bonds_size; }
    const CovariantShape& get_input_shape() const { return input_shape; }

    // Length of the coefficient vector: every block below, in their order.
    int coeff_count() const override;
    // The coefficients as one vector in mlip-4's params order.
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
    // The coefficients of core i's real or imaginary part, ordered l3, n3, l2, n2, l1, n1
    const std::vector<double>& get_core_coeffs(int i, bool imaginary) const { return (imaginary ? cores_im[i] : cores[i]).coeffs; }

    void set_radial_basis_coeffs(int l, const double* c);
    void set_species_pair_coeffs(int l, const double* c);
    void set_core_coeffs(int i, bool imaginary, const double* c);
    // Sizes the coefficients for n species, keeping each kept pair's A_l entries and each kept species'
    // site energy; the new entries are zero. Set them afterwards.
    void set_species_count(int n);
    // mlip-4's initial values, ETN::PostConstruct's: sin(10 + k^2) for the k-th of the params.
    void reset_coeffs();

  protected:
    void read_file(std::istream& is);
    void prepare_cores();
    Core prepare_core(int i, bool imaginary);
    void prepare_columns();
    void prepare_radial_mixing();

    // Shape: per l of the input, the species rank, radial functions and radial rank;
    // per bond, its rank per l, {1} at both ends
    std::vector<int> species_ranks;
    std::vector<int> radial_func_counts;
    std::vector<int> radial_ranks;
    std::vector<std::vector<int>> bond_ranks;

    // Coefficients, in get_coeffs' order: these, the cores' and species_coeffs
    std::vector<std::vector<double>> radial_basis_coeffs;        // [l] B_l
    std::vector<std::vector<double>> species_pair_coeffs;  // [l] A_l

    // First column of each coefficient block in get_coeffs' order
    std::vector<int> radial_columns;        // [l]
    std::vector<int> species_pair_columns;  // [l]
    std::vector<int> core_columns;          // [core]
    std::vector<int> core_im_columns;       // [core]
    int species_columns = 0;

    // Tensor-train layout
    std::vector<Core> cores;                // [core_count]
    std::vector<Core> cores_im;             // [core_count]
    std::vector<std::vector<CoreTerm>> core_terms;  // [core_count] the parts each core applies
    std::vector<WignerEntry> wigner;        // the 3-j symbols the blocks point into
    CovariantShape input_shape{{1}};        // v, with the constant 1 ahead of the l = 0 channels
    std::vector<CovariantShape> bond_shapes;
    std::vector<int> bond_offsets;          // [core_count + 1] real part of bond b in one atom's block
    int bonds_size = 0;
    std::vector<int> mixed_offsets;         // [l_count + 1] radial-rank channels of each l

    // A_l contracted with B_l for each species pair: [l][(s_i * species_count + s_j) * N_rad * R_rad + n' * R_rad + n]
    std::vector<std::vector<double>> radial_mixing;

    RealHarmonics harmonics{1};

    // Tensor train, forwards and backwards pass, for compute's chunk of central atoms, a value per atom
    std::vector<Lanes> input_vals;         // [input_shape.size] the input vector v, the constant 1 first
    std::vector<Lanes> bond_vals;          // [bonds_size] the bond vectors, bond 0 = 1
    std::vector<Lanes> input_energy_ders;  // [input_shape.size] dE_i/dv
    std::vector<Lanes> bond_energy_ders;   // [bonds_size] dE_i/dbond

    // Each block of each core term contracted with the left bond, sum_n1 c[n3, n2, n1] u[m1, n1]:
    // at products_offset of the term and of the block, [(m1 + l1) * n3_count * n2_count + n3 * n2_count + n2]
    std::vector<Lanes> core_products;
    // The derivative of one block's products, laid out as one block's, sized for the largest
    std::vector<Lanes> core_product_ders;

    // The same for the kernels that take one atom at a time
    std::vector<double> atom_input_vals;
    std::vector<double> atom_bond_vals;
    std::vector<double> atom_input_energy_ders;
    std::vector<double> atom_bond_energy_ders;
    std::vector<double> atom_core_products;
    std::vector<double> atom_core_product_ders;

    // neighbor_cache holds per neighbor 1/dist, Y, grad Y, the radial channels and their ders:
    // cache_stride is 1 + 4 * l_count^2 + 2 * radial-rank channels
};
