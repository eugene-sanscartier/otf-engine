/* -*- c++ -*- ----------------------------------------------------------
   MaxVol (maximum-volume) D-optimality selection and extrapolation grading.
   Algorithm of mlip-3's MaxVol (mlip-3/src/maxvol.{h,cpp}) and of
   cfg_selection.cpp's PrepareMatrix, on a Pair's gradients.
   Copyright (c) 2023, Alexander Shapeev (Skoltech). BSD 2-Clause, see LICENSE.mlip-3.

   One equation is one row of mlip-3's matrix B: the gradient of an
   observable with respect to the coefficients. The grade of an equation v
   against the active set A is max_i |v . invA[i]|, with invA = A^-T stored
   row-major, as lammps-mtp's pair_mtp_extrapolation grades.

   Grading multiplies blocks of equations by invA^T on one thread, through
   Eigen.
------------------------------------------------------------------------- */

#pragma once

#include "comm.h"
#include "pair.h"

#include <optional>
#include <vector>

// The MaxVol equations of a set of structures, under one set of selection weights.
class Equations {
  public:
    Equations(int coeff_count, double energy_weight, double force_weight, double stress_weight, double site_en_weight, int weight_scaling);

    // Appends the equations of one structure, as PrepareMatrix builds them:
    //   energy only           : the total energy, scaled by energy_weight / n_atoms^(weight_scaling/2)
    //   with forces or stress : the raw total energy (with energy_weight), the forces times
    //                           force_weight, the 9 stress components times stress_weight / n_atoms^(weight_scaling/2)
    //   site_en_weight > 0    : the raw site energies, last
    // structure numbers it; by default the next after the highest so far.
    void add(Pair& pot, const NeighList& list, int structure = -1);
    // The same from the gradients of the site energies [inum * coeff_count] and, when needs_forces(),
    // of the forces [n_atoms * 3 * coeff_count] and of the virial [6 * coeff_count].
    void add(const NeighList& list, const double* site_energy_grad, const double* force_grad, const double* virial_grad, int structure = -1);

    bool needs_forces() const { return force_weight != 0.0 || stress_weight != 0.0; }

    // The equations whose keep entry is true [size()], under the same structure numbering.
    Equations subset(const bool* keep) const;

    int size() const { return (int) structure_indices.size(); }

    const int coeff_count;
    const double energy_weight, force_weight, stress_weight, site_en_weight;
    const int weight_scaling;
    int structure_count = 0;

    std::vector<double> grads;          // [size * coeff_count]
    std::vector<int> structure_indices; // [size] which added structure each equation belongs to
    std::vector<int> equation_indices;  // [size] mlip-3's equation index within its structure

  private:
    void append(const double* grad, double factor, int structure, int equation);

    std::vector<double> site_energy_grad, force_grad, virial_grad;
};

// grades[r] = max_i |rows[r] . invA[i]| for count rows of n, invA [n * n] row-major, by blocks of rows; block is scratch.
void grade_rows(const double* invA, int n, const double* rows, int count, double* grades, std::vector<double>& block);

class MaxVol {
  public:
    // A = init_scale * I; the defaults are mlip-3's.
    explicit MaxVol(int n, double init_scale = 1e-6, double threshold = 1.001, std::optional<double> threshold_swap = std::nullopt);
    // A and invA [n * n] as stored in an #MVS_v1.1 block.
    MaxVol(int n, const double* A, const double* invA, double threshold = 1.001, std::optional<double> threshold_swap = std::nullopt);

    // grades[r] = max_i |rows[r] . invA[i]| for count rows of n.
    void grade(const double* rows, int count, double* grades);

    // MaximizeVol: swaps in the highest-grading equation of pool until none grades above threshold
    // (threshold_swap once it has swapped, when set) or
    // max_swaps are made. Each swap puts the equation it displaces from A into pool in its place, with its
    // provenance, so later sweeps grade the displaced one; pool is restored when the search ends.
    // pool_id is recorded as the pool of each of pool's equations that enters A.
    // Over the ranks of comm, each holding the same active set, pool is this rank's share; the swaps, and so A
    // and its provenance, are those of the whole pool searched on one rank, with its structures added in order.
    void maximize_volume(Equations& pool, int pool_id, Comm& comm, int max_swaps = 99999);

    // Provenance of each active equation, from indices saved beside A.
    void restore_active(const int* structure_indices, const int* eqn_indices, int pool_id);

    const int n;
    double threshold;
    // Once a search has swapped, it goes on while a grade exceeds threshold_swap, as mlip-4's MaximizeVol
    // does; unset, threshold.
    std::optional<double> threshold_swap;
    std::vector<double> A, invA;                // [n * n] row k of A is active equation k
    std::vector<int> active_pool_ids;           // [n] -1 where row k is still init_scale * e_k
    std::vector<int> active_structure_indices;     // [n]
    std::vector<int> active_eqn_indices;        // [n]

  private:
    // UpdateInvA: swaps v into the row of A it grades highest against, when that grade exceeds bound,
    // by a rank-1 update of invA; the displaced row goes to displaced [n].
    bool try_swap(const double* v, double bound, int pool_id, int structure_index, int eqn_index, double* displaced, int& displaced_pool_id, int& displaced_structure_index, int& displaced_eqn_index);

    std::vector<double> w, dv, buf3, row_k;     // [n] try_swap's
    std::vector<double> block;                  // [grade_block_rows * n] grade's product block
    std::vector<double> sweep_grades;           // [pool size] maximize_volume's
};
