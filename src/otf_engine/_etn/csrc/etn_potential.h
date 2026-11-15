/* -*- c++ -*- ----------------------------------------------------------
   Equivariant tensor network potential: the computation of mlip-4's ETN
   (src/pots/etn/etn.cpp at 79a39bf) in PairMTP's structure.
------------------------------------------------------------------------- */

#pragma once

#include "etn_harmonics.h"
#include "etn_radial_basis.h"
#include "neigh_list.h"

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

// The real or the imaginary part of one core, between bond i (left), the input and bond i + 1.
struct Core {
    int coeff_offset = 0;
    int coeff_count = 0;
    std::vector<CoreBlock> blocks;
};

class ETN {
  public:
    // Reads an .etn file.
    explicit ETN(const std::string& filename);
    // An ETN of the given shape with every coefficient zero.
    ETN(int species_count, std::vector<int> species_ranks, std::vector<int> radial_func_counts, std::vector<int> radial_ranks, std::vector<std::vector<int>> bond_ranks, int radial_basis_size, double min_dist, double max_dist);

    // Energy, per-atom energies, forces and virial of the central atoms in list, and each
    // central atom's input vector and bond vectors, left in the members below.
    void compute(const NeighList& list);

    double get_energy() const { return energy; }
    const double* get_eatom() const { return eatom.data(); }      // [n_atoms]
    const double* get_forces() const { return forces.data(); }    // [n_atoms * 3]
    const double* get_virial() const { return virial; }           // [6] xx,yy,zz,xy,xz,yz
    const double* get_inputs() const { return inputs.data(); }    // [inum * input_shape.size]
    const double* get_bonds() const { return bonds.data(); }      // [inum * bonds_size]

    // Bond b of the tensor train: its layout, and the offsets of its real and imaginary parts
    // within one atom's block of get_bonds(); the imaginary part is zero outside bonds 2 .. d-2.
    const CovariantShape& bond_shape(int b) const { return bond_shapes[b]; }
    int bond_offset(int b) const { return bond_offsets[b]; }
    int get_bonds_size() const { return bonds_size; }
    const CovariantShape& get_input_shape() const { return input_shape; }

    int coeff_count() const { return (int) coeffs.size(); }
    const double* get_coeffs() const { return coeffs.data(); }
    void set_coeffs(const double* c);

    int get_species_count() const { return species_count; }
    int get_core_count() const { return (int) bond_ranks.size() - 1; }
    bool has_imaginary_cores() const { return get_core_count() > 3; }
    const std::vector<int>& get_species_ranks() const { return species_ranks; }
    const std::vector<int>& get_radial_func_counts() const { return radial_func_counts; }
    const std::vector<int>& get_radial_ranks() const { return radial_ranks; }
    const std::vector<std::vector<int>>& get_bond_ranks() const { return bond_ranks; }
    const RadialBasisCinf& get_radial_basis() const { return radial_basis; }
    const std::string& get_potential_name() const { return potential_name; }
    double get_scaling() const { return radial_basis.scaling; }
    void set_scaling(double s) { radial_basis.scaling = s; }

    // Offsets into the coefficient vector, in mlip-4's params order.
    int radial_coeffs_offset(int l) const { return radial_offsets[l]; }
    int species_pair_coeffs_offset(int l) const { return species_pair_offsets[l]; }
    const Core& core(int i, bool imaginary) const { return imaginary ? cores_im[i] : cores_re[i]; }
    int species_coeffs_offset() const { return coeff_count() - species_count; }

  protected:
    void read_file(std::istream& is);
    void build();
    void prepare_radial_mixing();
    Core build_core(int i, bool imaginary, int& offset);
    void contract(const Core& core, const CovariantShape& left, const CovariantShape& right, const double* u, const double* v, double* out) const;
    void contract_adjoint(const Core& core, const CovariantShape& left, const CovariantShape& right, const double* u, const double* v, const double* out_adj, double* u_adj, double* v_adj) const;

    std::string potential_name = "Untitled";
    int species_count = 0;
    RadialBasisCinf radial_basis{1, 0.5, 1.0};

    // Shape, per l of the input: species rank, radial functions, radial rank
    std::vector<int> species_ranks;
    std::vector<int> radial_func_counts;
    std::vector<int> radial_ranks;
    std::vector<std::vector<int>> bond_ranks;  // [core_count + 1][l]; bonds 0 and d are {1}

    std::vector<double> coeffs;
    std::vector<int> radial_offsets;        // [l] B_l, column-major (k, n', n)
    std::vector<int> species_pair_offsets;  // [l] A_l, [s_i][s_j][k]
    std::vector<Core> cores_re;             // [core_count]
    std::vector<Core> cores_im;             // [core_count], empty but for 1 .. d-2 when d > 3
    std::vector<WignerEntry> wigner;        // the 3-j symbols the blocks point into

    CovariantShape input_shape{{1}};
    std::vector<CovariantShape> bond_shapes;
    std::vector<int> bond_offsets;  // [core_count + 1] real part of bond b; imaginary part after it
    int bonds_size = 0;

    // A_l contracted with B_l for each species pair: [l][(s_i * species_count + s_j) * N_rad * R_rad + n' * R_rad + n]
    std::vector<std::vector<double>> radial_mixing;

    RealHarmonics harmonics{1};

    // Results of compute
    double energy = 0.0;
    std::vector<double> eatom;   // [n_atoms]
    std::vector<double> forces;  // [n_atoms * 3]
    double virial[6] = {};       // xx,yy,zz,xy,xz,yz
    std::vector<double> inputs;  // [inum * input_shape.size]
    std::vector<double> bonds;   // [inum * bonds_size]

    // Scratch of compute
    std::vector<double> radial_vals, radial_ders;  // [radial_basis.size]
    std::vector<double> neighbor_cache;            // per neighbor: dr, 1/r, Y, grad Y, b, db/dr
    std::vector<double> input_adj;                 // [input_shape.size]
    std::vector<double> bond_adj;                  // [bonds_size]
    std::vector<double> mixed_adj;                 // [sum of radial ranks]
};
