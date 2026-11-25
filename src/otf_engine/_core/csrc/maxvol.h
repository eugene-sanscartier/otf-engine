/* -*- c++ -*- ----------------------------------------------------------
   MaxVol (maximum-volume) D-optimality selection and extrapolation grading.
   Algorithm of mlip-3's MaxVol (mlip-3/src/maxvol.{h,cpp}) and of
   cfg_selection.cpp's PrepareMatrix, on a Pair's gradients.
   Copyright (c) 2023, Alexander Shapeev (Skoltech). BSD 2-Clause, see LICENSE.mlip-3.

   An equation is the gradient of an observable with respect to the
   coefficients. The grade of an equation v against the active set A is
   max_i |v . invA[i]|, where invA = A^-T is stored row-major.

   Grading multiplies blocks of equations by invA^T with Eigen, on one
   thread.
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

    // Appends the equations of one structure:
    //   energy only           : the total energy, scaled by energy_weight / n_atoms^(weight_scaling/2)
    //   with forces or stress : the unscaled total energy (if energy_weight != 0), the forces times
    //                           force_weight, the 9 stress components times stress_weight / n_atoms^(weight_scaling/2)
    //   site_en_weight > 0    : the unscaled site energies, last
    // structure: the number of the structure, by default one more than the highest so far.
    void add(Pair& pot, const NeighList& list, int structure = -1);
    // Appends the same equations from site_energy_grad [inum * coeff_count] and, when needs_forces(),
    // force_grad [n_atoms * 3 * coeff_count] and virial_grad [6 * coeff_count].
    void add(const NeighList& list, const double* site_energy_grad, const double* force_grad, const double* virial_grad, int structure = -1);

    bool needs_forces() const { return force_weight != 0.0 || stress_weight != 0.0; }

    // Returns the equations whose entry of keep [size()] is true, with the same structure numbers.
    Equations subset(const bool* keep) const;

    int size() const { return (int) structure_indices.size(); }

    const int coeff_count;
    const double energy_weight, force_weight, stress_weight, site_en_weight;
    const int weight_scaling;
    int structure_count = 0;

    std::vector<double> grads;          // [size * coeff_count]
    std::vector<int> structure_indices; // [size] structure number of each equation
    std::vector<int> equation_indices;  // [size] index within the structure: 0 energy, 1 + r force row r, 1 + 3 n_atoms + q stress q, 10 + 3 n_atoms + ii site energy ii

  private:
    void append(const double* grad, double factor, int structure, int equation);

    std::vector<double> site_energy_grad, force_grad, virial_grad;
};

// Sets grades[r] = max_i |rows[r] . invA[i]| for count rows of length n, by blocks of rows.
// invA: [n * n] row-major. block: scratch.
void grade_rows(const double* invA, int n, const double* rows, int count, double* grades, std::vector<double>& block);

class MaxVol {
  public:
    // Starts from A = init_scale * I.
    explicit MaxVol(int n, double init_scale = 1e-6, double threshold = 1.001, std::optional<double> threshold_swap = std::nullopt);
    // Restores A and invA [n * n], as stored in an #MVS_v1.1 block.
    MaxVol(int n, const double* A, const double* invA, double threshold = 1.001, std::optional<double> threshold_swap = std::nullopt);

    // Sets grades[r] = max_i |rows[r] . invA[i]| for count rows of length n.
    void grade(const double* rows, int count, double* grades);

    // Swaps the highest-grading equation of pool into A until no equation grades above the bound, or max_swaps swaps.
    // The bound is threshold, and threshold_swap after the first swap when it is set.
    // Each swap puts the displaced row of A, with its provenance, into pool in place of the entering equation, and
    // later sweeps grade it. The search restores pool at the end.
    // pool_id: the pool recorded for each equation of pool that enters A.
    // comm: each rank holds the same active set and its own share of pool. The swaps, A and its provenance are those
    // of a search of the whole pool on one rank, with its structures added in order.
    void maximize_volume(Equations& pool, int pool_id, Comm& comm, int max_swaps = 99999);

    // Sets the provenance of each active row from indices saved with A.
    void restore_active(const int* structure_indices, const int* eqn_indices, int pool_id);

    const int n;
    double threshold;
    // After the first swap, a search goes on while a grade exceeds threshold_swap, or threshold when it is unset.
    std::optional<double> threshold_swap;
    std::vector<double> A, invA;                // [n * n] row k of A is active equation k
    std::vector<int> active_pool_ids;           // [n] -1 where row k is still init_scale * e_k
    std::vector<int> active_structure_indices;     // [n]
    std::vector<int> active_eqn_indices;        // [n]

  private:
    // Swaps v into the row of A where v has its highest grade, when that grade exceeds bound, and updates invA by
    // rank 1. Writes the displaced row and its provenance to displaced [n] and the displaced_ arguments.
    // Returns whether it swapped.
    bool try_swap(const double* v, double bound, int pool_id, int structure_index, int eqn_index, double* displaced, int& displaced_pool_id, int& displaced_structure_index, int& displaced_eqn_index);

    std::vector<double> w, dv, buf3, row_k;     // [n] scratch of try_swap
    std::vector<double> block;                  // [grade_block_rows * n] product block of grade
    std::vector<double> sweep_grades;           // [pool size] grades of a sweep of maximize_volume
};
