/* -*- c++ -*- ----------------------------------------------------------
   Standalone ETN potential — no LAMMPS dependency.
   Computes mlip-4's ETN (src/pots/etn/etn.cpp at 79a39bf), which has no
   LAMMPS pair style; hand-written passes replace mlip-4's program.
   Original author of mlip-4's ETN: Max Hodapp
------------------------------------------------------------------------- */

#pragma once

#include "etn_harmonics.h"
#include "mtp_radial_basis.h"
#include "neigh_list.h"

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

class PairETN {
  public:
    // Reads an ETN potential file (version 1.0.0).
    explicit PairETN(const std::string& filename);
    virtual ~PairETN();

    // Energy, per-atom energies, forces and virial of the central atoms in list,
    // and each central atom's input and bond vectors, left in the members below.
    void compute(const NeighList& list);

    double get_energy() const { return energy; }
    const double* get_eatom() const { return eatom.data(); }      // [n_atoms]
    const double* get_forces() const { return forces.data(); }    // [n_atoms * 3]
    const double* get_virial() const { return virial; }           // [6] xx,yy,zz,xy,xz,yz
    const double* get_inputs() const { return inputs.data(); }    // [inum * input_shape.size]
    const double* get_bonds() const { return bonds.data(); }      // [inum * bonds_size]

    // Bond b of the tensor train: its layout, and the offset of its real part within one atom's
    // block of get_bonds(); the imaginary part follows it, and is zero outside bonds 2 .. d-2.
    const CovariantShape& get_bond_shape(int b) const { return bond_shapes[b]; }
    int get_bond_offset(int b) const { return bond_offsets[b]; }
    int get_bonds_size() const { return bonds_size; }
    const CovariantShape& get_input_shape() const { return input_shape; }

    // Radial basis values and derivatives at dist.
    void eval_radial_basis(double dist, double* vals_out, double* ders_out);

    // Length of mlip-4's params: every block below, in their order.
    int coeff_count() const;
    int get_species_count() const { return species_count; }
    int get_core_count() const { return (int) bond_ranks.size() - 1; }
    bool has_imaginary_cores() const { return get_core_count() > 3; }
    const std::vector<int>& get_species_ranks() const { return species_ranks; }
    const std::vector<int>& get_radial_func_counts() const { return radial_func_counts; }
    const std::vector<int>& get_radial_ranks() const { return radial_ranks; }
    const std::vector<std::vector<int>>& get_bond_ranks() const { return bond_ranks; }
    int get_radial_basis_size() const { return radial_basis_size; }
    double get_min_cutoff() const { return min_cutoff; }
    double get_max_cutoff() const { return max_cutoff; }
    double get_scaling() const { return scaling; }
    // False for a file without coefficients, which holds mlip-4's initial values.
    bool is_trained() const { return trained; }
    const std::string& get_potential_name() const { return potential_name; }
    const std::string& get_radial_basis_type() const { return radial_basis_type; }

    // radial_coeffs[l]: [species_rank * radial_func_count * radial_rank], column-major (k, n', n)
    const std::vector<double>& get_radial_coeffs(int l) const { return radial_coeffs[l]; }
    // species_pair_coeffs[l]: [species_count^2 * species_rank], row-major (s_i, s_j, k)
    const std::vector<double>& get_species_pair_coeffs(int l) const { return species_pair_coeffs[l]; }
    // core_coeffs[i], core_coeffs_im[i]: the core's blocks, ordered l3, n3, l2, n2, l1, n1
    const std::vector<double>& get_core_coeffs(int i, bool imaginary) const { return imaginary ? core_coeffs_im[i] : core_coeffs[i]; }
    const std::vector<double>& get_species_coeffs() const { return species_coeffs; }

    void set_radial_coeffs(int l, const double* c);
    void set_species_pair_coeffs(int l, const double* c);
    void set_core_coeffs(int i, bool imaginary, const double* c);
    void set_species_coeffs(const double* c) { std::copy(c, c + species_count, species_coeffs.begin()); }
    void set_scaling(double s) { scaling = s; radial_basis->scaling = s; }
    void set_trained(bool t) { trained = t; }

  protected:
    void read_file(std::istream& is);
    void prepare_cores();
    Core prepare_core(int i, bool imaginary);
    void prepare_radial_mixing();
    void contract(const Core& core, const double* c, const CovariantShape& left, const CovariantShape& right, const double* u, const double* v, double* out) const;
    void contract_adjoint(const Core& core, const double* c, const CovariantShape& left, const CovariantShape& right, const double* u, const double* v, const double* out_adj, double* u_adj, double* v_adj) const;

    std::string potential_name = "Untitled";
    std::string radial_basis_type;

    int species_count = 0;
    double scaling = 1.0;
    bool trained = true;

    // Radial basis
    RadialMTPBasis* radial_basis = nullptr;
    int radial_basis_size = 0;
    double min_cutoff = 0.0;
    double max_cutoff = 0.0;
    double max_cutoff_sq = 0.0;

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
    std::vector<double> species_coeffs;                    // [species] site energy offsets

    // Tensor-train layout
    std::vector<Core> cores;                // [core_count]
    std::vector<Core> cores_im;             // [core_count]
    std::vector<WignerEntry> wigner;        // the 3-j symbols the blocks point into
    CovariantShape input_shape{{1}};        // v, with the constant 1 ahead of the l = 0 channels
    std::vector<CovariantShape> bond_shapes;
    std::vector<int> bond_offsets;          // [core_count + 1] real part of bond b in one atom's block
    int bonds_size = 0;
    std::vector<int> mixed_offsets;         // [l_count + 1] radial-rank channels of each l

    // A_l contracted with B_l for each species pair: [l][(s_i * species_count + s_j) * N_rad * R_rad + n' * R_rad + n]
    std::vector<std::vector<double>> radial_mixing;

    RealHarmonics harmonics{1};

    // Results of compute
    double energy = 0.0;
    std::vector<double> eatom;   // [n_atoms] site energies
    std::vector<double> forces;  // [n_atoms * 3]
    double virial[6] = {};       // xx,yy,zz,xy,xz,yz
    std::vector<double> inputs;  // [inum * input_shape.size]
    std::vector<double> bonds;   // [inum * bonds_size]

    // Adjoints of the backwards pass
    std::vector<double> input_adj;  // [input_shape.size]
    std::vector<double> bond_adj;   // [bonds_size]

    // Cache values between forwards and backwards pass
    int cache_size = 0;
    std::vector<int> cached_j;                    // [cache_size]
    std::vector<std::array<double, 3>> valid_dr;  // [cache_size] displacement of each cached neighbor
    std::vector<double> neighbor_cache;           // [cache_size * stride] 1/dist, Y, grad Y, radial channels and their ders
};
