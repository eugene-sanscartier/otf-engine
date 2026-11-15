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
// base + n3 * stride3 + n2 * stride2 + n1, and wigner[w_begin, w_end) holds its 3-j symbols.
struct CoreBlock {
    int l1, l2, l3;
    int base, stride2, stride3;
    int w_begin, w_end;
};

// The blocks of the real or the imaginary part of one core, between bond i, the input and bond i + 1.
struct Core {
    int coeff_count = 0;
    std::vector<CoreBlock> blocks;
};

// One term of a core: its real or imaginary part applied to one part (0 real, 1 imaginary) of
// bond i, added into one part of bond i + 1.
struct CoreTerm {
    bool imaginary;
    int in_part, out_part;
};

// The members PairETN shares with PairMTP are Pair's.
class PairETN : public Pair {
  public:
    // Reads an ETN potential file (version 1.0.0).
    explicit PairETN(const std::string& filename);

    // Energy, per-atom energies, forces and virial of the central atoms in list,
    // and each central atom's input and bond vectors, left in the members below.
    void compute(const NeighList& list) override;

    const double* get_inputs() const { return inputs.data(); }    // [inum * input_shape.size]
    const double* get_bonds() const { return bonds.data(); }      // [inum * bonds_size]

    // Bond b of the tensor train: its layout, and the offset of its real part within one atom's
    // block of get_bonds(); the imaginary part follows it, and is zero outside bonds 2 .. d-2.
    const CovariantShape& get_bond_shape(int b) const { return bond_shapes[b]; }
    int get_bond_offset(int b) const { return bond_offsets[b]; }
    int get_bonds_size() const { return bonds_size; }
    const CovariantShape& get_input_shape() const { return input_shape; }

    // Length of mlip-4's params: every block below, in their order.
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
    // radial_coeffs[l]: [species_rank * radial_func_count * radial_rank], column-major (k, n', n)
    const std::vector<double>& get_radial_coeffs(int l) const { return radial_coeffs[l]; }
    // species_pair_coeffs[l]: [species_count^2 * species_rank], row-major (s_i, s_j, k)
    const std::vector<double>& get_species_pair_coeffs(int l) const { return species_pair_coeffs[l]; }
    // core_coeffs[i], core_coeffs_im[i]: the core's blocks, ordered l3, n3, l2, n2, l1, n1
    const std::vector<double>& get_core_coeffs(int i, bool imaginary) const { return imaginary ? core_coeffs_im[i] : core_coeffs[i]; }

    void set_radial_coeffs(int l, const double* c);
    void set_species_pair_coeffs(int l, const double* c);
    void set_core_coeffs(int i, bool imaginary, const double* c);

  protected:
    void read_file(std::istream& is);
    void prepare_cores();
    Core prepare_core(int i, bool imaginary);
    void prepare_radial_mixing();

    // The input vector v of central atom ii, whose neighbors start at nbr_offset in list; fills
    // cached_j, valid_dr and neighbor_cache, and returns the count of neighbors within the cutoff.
    int site_input(const NeighList& list, int ii, int nbr_offset, double* v);
    // The bond vectors of v, into bond [bonds_size], from bond 0 = 1.
    void train_forward(const double* v, double* bond) const;
    // From bond_adj, the adjoint of bond, adds the adjoints of v into v_adj and, when core_adj and
    // core_im_adj are given ([core] pointers to each part's coefficient adjoints), of the cores.
    void train_backward(const double* v, const double* bond, double* bond_adj, double* v_adj, double* const* core_adj, double* const* core_im_adj) const;

    // One core part applied to (u, v), added into out, and its adjoint; c_adj may be null.
    void contract(const Core& core, const double* c, const CovariantShape& left, const CovariantShape& right, const double* u, const double* v, double* out) const;
    void contract_adjoint(const Core& core, const double* c, const CovariantShape& left, const CovariantShape& right, const double* u, const double* v, const double* out_adj, double* u_adj, double* v_adj, double* c_adj) const;

    // Shape: per l of the input, the species rank, radial functions and radial rank;
    // per bond, its rank per l, {1} at both ends
    std::vector<int> species_ranks;
    std::vector<int> radial_func_counts;
    std::vector<int> radial_ranks;
    std::vector<std::vector<int>> bond_ranks;

    // Coefficients, in mlip-4's params order
    std::vector<std::vector<double>> radial_coeffs;        // [l] B_l
    std::vector<std::vector<double>> species_pair_coeffs;  // [l] A_l
    std::vector<std::vector<double>> core_coeffs;          // [core] real part
    std::vector<std::vector<double>> core_coeffs_im;       // [core] imaginary part, empty but for 1 .. d-2 when d > 3

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

    // Results of compute beside Pair's
    std::vector<double> inputs;  // [inum * input_shape.size]
    std::vector<double> bonds;   // [inum * bonds_size]

    // Adjoints of the backwards pass
    std::vector<double> input_adj;  // [input_shape.size]
    std::vector<double> bond_adj;   // [bonds_size]

    // neighbor_cache holds per neighbor 1/dist, Y, grad Y, the radial channels and their ders:
    // cache_stride is 1 + 4 * l_count^2 + 2 * radial-rank channels
};
