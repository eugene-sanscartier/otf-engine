/* -*- c++ -*- ----------------------------------------------------------
   MaxVol selection and grading (see maxvol.h).
------------------------------------------------------------------------- */

#include "maxvol.h"
#include "maxvol_sweep.h"

#include <algorithm>
#include <cmath>

// ---------------------------------------------------------------------------
// Equations
// ---------------------------------------------------------------------------

Equations::Equations(int coeff_count, double energy_weight, double force_weight, double stress_weight, double site_en_weight, int weight_scaling)
    : coeff_count(coeff_count), energy_weight(energy_weight), force_weight(force_weight), stress_weight(stress_weight), site_en_weight(site_en_weight), weight_scaling(weight_scaling) {}

void Equations::append(const double* grad, double factor, int structure, int equation) {
    const size_t start = grads.size();
    grads.resize(start + coeff_count);
    for (int c = 0; c < coeff_count; c++) grads[start + c] = grad[c] * factor;
    structure_indices.push_back(structure);
    equation_indices.push_back(equation);
}

void Equations::add(MTPTraining& pot, const NeighList& list) {
    const int n_atoms = list.n_atoms, cc = coeff_count;
    const double scale = std::max(std::pow((double) n_atoms, weight_scaling / 2.0), 1e-30);
    const bool forces = force_weight != 0.0, stress = stress_weight != 0.0;

    site_energy_grad.resize((size_t) list.inum * cc);
    if (forces || stress) force_grad.resize((size_t) n_atoms * 3 * cc);
    if (stress) virial_grad.resize(6 * cc);
    pot.eval_grad(list, site_energy_grad.data(), forces || stress ? force_grad.data() : nullptr, stress ? virial_grad.data() : nullptr);

    const int s = structure_count++;
    if (energy_weight != 0.0) {
        std::vector<double> total(cc, 0.0);
        for (int ii = 0; ii < list.inum; ii++)
            for (int c = 0; c < cc; c++) total[c] += site_energy_grad[(size_t) ii * cc + c];
        append(total.data(), forces || stress ? 1.0 : energy_weight / scale, s, 0);
    }
    if (forces)
        for (int r = 0; r < 3 * n_atoms; r++) append(&force_grad[(size_t) r * cc], force_weight, s, 1 + r);
    if (stress) {
        // mlip-3 stores the full 3x3 stress, not its 6 Voigt components
        const int voigt[9] = {0, 3, 4, 3, 1, 5, 4, 5, 2};
        for (int q = 0; q < 9; q++) append(&virial_grad[(size_t) voigt[q] * cc], stress_weight / scale, s, 1 + 3 * n_atoms + q);
    }
    if (site_en_weight != 0.0)
        for (int ii = 0; ii < list.inum; ii++) append(&site_energy_grad[(size_t) ii * cc], 1.0, s, 1 + 3 * n_atoms + 9 + ii);
}

Equations Equations::subset(const bool* keep) const {
    Equations kept(coeff_count, energy_weight, force_weight, stress_weight, site_en_weight, weight_scaling);
    kept.structure_count = structure_count;
    for (int r = 0; r < size(); r++)
        if (keep[r]) kept.append(&grads[(size_t) r * coeff_count], 1.0, structure_indices[r], equation_indices[r]);
    return kept;
}

// ---------------------------------------------------------------------------
// MaxVol
// ---------------------------------------------------------------------------

MaxVol::MaxVol(int n, double init_scale, double threshold)
    : n(n), threshold(threshold), A((size_t) n * n, 0.0), invA((size_t) n * n, 0.0), active_pool_ids(n, -1), active_struct_indices(n, -1), active_eqn_indices(n, -1), w(n), dv(n), buf3(n), row_k(n) {
    for (int k = 0; k < n; k++) {
        A[(size_t) k * n + k] = init_scale;
        invA[(size_t) k * n + k] = 1.0 / init_scale;
    }
}

MaxVol::MaxVol(int n, const double* A, const double* invA, double threshold)
    : n(n), threshold(threshold), A(A, A + (size_t) n * n), invA(invA, invA + (size_t) n * n), active_pool_ids(n, -1), active_struct_indices(n, -1), active_eqn_indices(n, -1), w(n), dv(n), buf3(n), row_k(n) {}

void MaxVol::grade(const double* rows, int count, double* grades) {
    block.resize((size_t) std::min(sweep_block_rows, count) * n);
#ifdef MTP_SWEEP_V3
    static const bool v3 = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    if (v3) return sweep_grades_v3(rows, count, invA.data(), n, block.data(), grades);
#endif
    sweep_grades_portable(rows, count, invA.data(), n, block.data(), grades);
}

void MaxVol::maximize_volume(Equations& pool, int pool_id, int max_swaps) {
    const int count = pool.size();
    if (count == 0) return;
    sweep_grades.resize(count);

    // The pool rows each swap overwrote, to put back when the search ends
    struct Replaced {
        int row, struct_index, eqn_index;
        std::vector<double> grad;
    };
    std::vector<Replaced> replaced;
    std::vector<double> displaced(n);
    // A displaced equation keeps its own pool, as mlip-3 swaps an equation's link to its configuration with it
    std::vector<int> pool_ids(count, pool_id);

    for (int swaps = 0; swaps < max_swaps; swaps++) {
        grade(pool.grads.data(), count, sweep_grades.data());
        const int best = (int) (std::max_element(sweep_grades.begin(), sweep_grades.end()) - sweep_grades.begin());
        if (sweep_grades[best] <= threshold) break;

        double* row = &pool.grads[(size_t) best * n];
        int displaced_pool_id, displaced_struct_index, displaced_eqn_index;
        if (!try_swap(row, pool_ids[best], pool.structure_indices[best], pool.equation_indices[best], displaced.data(), displaced_pool_id, displaced_struct_index, displaced_eqn_index)) break;

        replaced.push_back({best, pool.structure_indices[best], pool.equation_indices[best], std::vector<double>(row, row + n)});
        std::copy(displaced.begin(), displaced.end(), row);
        pool_ids[best] = displaced_pool_id;
        pool.structure_indices[best] = displaced_struct_index;
        pool.equation_indices[best] = displaced_eqn_index;
    }

    for (auto it = replaced.rbegin(); it != replaced.rend(); ++it) {
        std::copy(it->grad.begin(), it->grad.end(), &pool.grads[(size_t) it->row * n]);
        pool.structure_indices[it->row] = it->struct_index;
        pool.equation_indices[it->row] = it->eqn_index;
    }
}

void MaxVol::restore_active(const int* cfg_indices, const int* eqn_indices, int pool_id) {
    for (int k = 0; k < n; k++) {
        active_pool_ids[k] = cfg_indices[k] >= 0 ? pool_id : -1;
        active_struct_indices[k] = cfg_indices[k];
        active_eqn_indices[k] = eqn_indices[k];
    }
}

bool MaxVol::try_swap(const double* v, int pool_id, int struct_index, int eqn_index, double* displaced, int& displaced_pool_id, int& displaced_struct_index, int& displaced_eqn_index) {
    // w = invA v, the grade elements; row k of A is the one v replaces
    for (int i = 0; i < n; i++) {
        double s = 0.0;
        for (int j = 0; j < n; j++) s += invA[(size_t) i * n + j] * v[j];
        w[i] = s;
    }
    int k = 0;
    for (int i = 1; i < n; i++)
        if (std::fabs(w[i]) > std::fabs(w[k])) k = i;
    if (std::fabs(w[k]) <= threshold) return false;

    // invA -= (invA (v - A[k]) / w[k]) invA[k]^T
    for (int j = 0; j < n; j++) dv[j] = v[j] - A[(size_t) k * n + j];
    const double tmp = 1.0 / w[k];
    for (int i = 0; i < n; i++) {
        double s = 0.0;
        for (int j = 0; j < n; j++) s += invA[(size_t) i * n + j] * dv[j];
        buf3[i] = tmp * s;
    }
    std::copy_n(&invA[(size_t) k * n], n, row_k.begin());
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) invA[(size_t) i * n + j] -= buf3[i] * row_k[j];

    std::copy_n(&A[(size_t) k * n], n, displaced);
    displaced_pool_id = active_pool_ids[k];
    displaced_struct_index = active_struct_indices[k];
    displaced_eqn_index = active_eqn_indices[k];
    std::copy_n(v, n, &A[(size_t) k * n]);
    active_pool_ids[k] = pool_id;
    active_struct_indices[k] = struct_index;
    active_eqn_indices[k] = eqn_index;
    return true;
}
