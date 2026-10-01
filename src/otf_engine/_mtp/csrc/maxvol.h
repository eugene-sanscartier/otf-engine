/* -*- c++ -*- ----------------------------------------------------------
   MaxVol (maximum-volume) D-optimality selection and extrapolation grading.
   Algorithm of mlip-3's MaxVol (mlip-3/src/maxvol.{h,cpp}) and of
   cfg_selection.cpp's PrepareMatrix, on MTPTraining's kernels.
   Copyright (c) 2023, Alexander Shapeev (Skoltech). BSD 2-Clause, see LICENSE.mlip-3.

   One equation is one row of mlip-3's matrix B: the gradient of an
   observable with respect to the coefficients. The grade of an equation v
   against the active set A is max_i |v . invA[i]|, with invA = A^-T stored
   row-major, as lammps-mtp's pair_mtp_extrapolation grades.

   Grading multiplies blocks of equations by invA^T on one thread (see
   maxvol_sweep.h).
------------------------------------------------------------------------- */

#pragma once

#include "mtp_training.h"

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
    void add(MTPTraining& pot, const NeighList& list);

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

class MaxVol {
  public:
    // A = init_scale * I, mlip-3's INIT_VALUE 1e-6; threshold is mlip-3's SELECT_THRESHOLD.
    explicit MaxVol(int n, double init_scale = 1e-6, double threshold = 1.001);
    // A and invA [n * n] as stored in an #MVS_v1.1 block.
    MaxVol(int n, const double* A, const double* invA, double threshold = 1.001);

    // grades[r] = max_i |rows[r] . invA[i]| for count rows of n.
    void grade(const double* rows, int count, double* grades);

    // MaximizeVol: swaps in the highest-grading equation of pool until none grades above threshold or
    // max_swaps are made. Each swap puts the equation it displaces from A into pool in its place, with its
    // provenance, so later sweeps grade the displaced one; pool is restored when the search ends.
    // pool_id is recorded as the pool of each of pool's equations that enters A.
    void maximize_volume(Equations& pool, int pool_id, int max_swaps = 99999);

    // Provenance of each active equation, from indices saved beside A.
    void restore_active(const int* cfg_indices, const int* eqn_indices, int pool_id);

    const int n;
    double threshold;
    std::vector<double> A, invA;                // [n * n] row k of A is active equation k
    std::vector<int> active_pool_ids;           // [n] -1 where row k is still init_scale * e_k
    std::vector<int> active_struct_indices;     // [n]
    std::vector<int> active_eqn_indices;        // [n]

  private:
    // UpdateInvA: swaps v into the row of A it grades highest against, when that grade exceeds threshold,
    // by a rank-1 update of invA; the displaced row goes to displaced [n].
    bool try_swap(const double* v, int pool_id, int struct_index, int eqn_index, double* displaced, int& displaced_pool_id, int& displaced_struct_index, int& displaced_eqn_index);

    std::vector<double> w, dv, buf3, row_k;     // [n] try_swap's
    std::vector<double> block;                  // [sweep_block_rows * n] grade's product block
    std::vector<double> sweep_grades;           // [pool size] maximize_volume's
};
