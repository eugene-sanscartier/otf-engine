/* -*- c++ -*- ----------------------------------------------------------
   The error table of a potential on training structures, which MTPFit
   and ETNFit log.
------------------------------------------------------------------------- */

#pragma once

#include "pair.h"
#include "fit_structure.h"

#include "comm.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

// Returns the error table of potential on the structures of every rank of comm, under title, on rank 0. Returns ""
// on the other ranks.
// Rows: the energy, the energy per atom, the force per atom, the force of each species (named from species, or
// "type t" past its end), and the virial as the Frobenius norm of the 3x3 tensor.
// Columns: n, MAE, RMSE, max |diff|, max |diff| / max |value| and RMSE / RMS(value).
// The value of an energy is its residual after per-species energies fitted to the structures by least squares.
// Its relative max divides by the range of the residuals.
// The table ends with the structures of the largest energy-per-atom and force errors.
inline std::string error_table(Pair& potential, const std::vector<FitStructure>& structures, const std::vector<std::string>& species, Comm& comm, const std::string& title) {
    enum { ENE, EPA, FRC, VIR, KINDS };    // rows; the force row of species t is KINDS + t
    const int species_count = potential.get_species_count();
    const int rows = KINDS + species_count;
    const int C = species_count;

    // normal equations of the per-species energies: [C * C] sum of counts counts^T, [C] sum of counts * energy,
    // [1] number of energies
    std::vector<double> fit(C * C + C + 1, 0.0), total_fit(C * C + C + 1), counts(C);
    for (const FitStructure& s : structures) {
        if (!s.has_energy) continue;
        std::fill(counts.begin(), counts.end(), 0.0);
        for (int t : s.types)
            if (t < C) counts[t] += 1.0;
        for (int a = 0; a < C; a++) {
            for (int b = 0; b < C; b++) fit[a * C + b] += counts[a] * counts[b];
            fit[C * C + a] += counts[a] * s.energy;
        }
        fit[C * C + C] += 1.0;
    }
    comm.allreduce(fit.data(), total_fit.data(), (int) fit.size(), ReduceOp::sum);

    // Gauss-Jordan elimination, pivoting on the largest remaining diagonal. Species past the independent
    // compositions keep an energy of 0.
    double* G = total_fit.data();
    double* g = G + C * C;
    std::vector<int> pivots;
    std::vector<double> species_energy(C, 0.0);
    double largest = 0.0;
    for (int a = 0; a < C; a++) largest = std::max(largest, G[a * C + a]);
    for (int step = 0; step < C; step++) {
        int k = -1;
        for (int a = 0; a < C; a++)
            if (std::find(pivots.begin(), pivots.end(), a) == pivots.end() && (k < 0 || G[a * C + a] > G[k * C + k])) k = a;
        if (G[k * C + k] <= 1e-12 * largest) break;
        pivots.push_back(k);
        for (int i = 0; i < C; i++) {
            if (i == k) continue;
            const double ratio = G[i * C + k] / G[k * C + k];
            for (int j = 0; j < C; j++) G[i * C + j] -= ratio * G[k * C + j];
            g[i] -= ratio * g[k];
        }
    }
    for (int k : pivots) species_energy[k] = g[k] / G[k * C + k];
    // false when the energies are no more than the independent compositions; every residual is then 0
    const bool residuals = total_fit[C * C + C] > (double) pivots.size();

    // sums [rows * 4]: count, sum |diff|, sum diff^2 and sum value^2 of each row
    // maxima [rows * 3]: max |diff|, max value and max -value of each row
    // at [rows]: the place of the max |diff|, structure number * 1e6 + atom number
    // The value is the energy residual, or the norm of the reference force or virial.
    std::vector<double> sums(rows * 4, 0.0), maxima(rows * 3, -1e300), at(rows, 0.0);
    auto add = [&](int row, double dltsq, double value, double where) {
        double* sum = &sums[row * 4];
        double* max = &maxima[row * 3];
        const double delta = std::sqrt(dltsq);
        sum[0] += 1.0;
        sum[1] += delta;
        sum[2] += dltsq;
        sum[3] += value * value;
        if (delta > max[0]) at[row] = where;
        max[0] = std::max(max[0], delta);
        max[1] = std::max(max[1], value);
        max[2] = std::max(max[2], -value);
    };

    for (const FitStructure& s : structures) {
        const NeighList list = s.list();
        potential.compute(list);
        const double* forces = potential.get_forces().data();
        const double* virial = potential.get_virial().data();

        if (s.has_energy) {
            const double diff = potential.get_energy() - s.energy;
            double residual = s.energy;
            for (int t : s.types)
                if (t < C) residual -= species_energy[t];
            add(ENE, diff * diff, residual, s.number * 1e6);
            add(EPA, diff * diff / ((double) s.n_atoms() * s.n_atoms()), residual / s.n_atoms(), s.number * 1e6);
        }
        if (s.has_forces)
            for (int ii = 0; ii < list.inum; ii++) {
                const int i = list.ilist[ii];
                double dltsq = 0.0, valsq = 0.0;
                for (int a = 0; a < 3; a++) {
                    dltsq += (forces[i * 3 + a] - s.forces[i * 3 + a]) * (forces[i * 3 + a] - s.forces[i * 3 + a]);
                    valsq += s.forces[i * 3 + a] * s.forces[i * 3 + a];
                }
                add(FRC, dltsq, std::sqrt(valsq), s.number * 1e6 + i + 1);
                if (s.types[i] < species_count) add(KINDS + s.types[i], dltsq, std::sqrt(valsq), s.number * 1e6 + i + 1);
            }
        if (s.has_virial) {
            double dltsq = 0.0, valsq = 0.0;
            for (int a = 0; a < 3; a++)
                for (int b = 0; b < 3; b++) {
                    const int k = VIRIAL_INDEX[a][b];
                    dltsq += (virial[k] - s.virial[k]) * (virial[k] - s.virial[k]);
                    valsq += s.virial[k] * s.virial[k];
                }
            add(VIR, dltsq, std::sqrt(valsq), s.number * 1e6);
        }
    }

    std::vector<double> total_sums(rows * 4), total_maxima(rows * 3), total_at(rows);
    comm.reduce(sums.data(), total_sums.data(), rows * 4, ReduceOp::sum, 0);
    comm.allreduce(maxima.data(), total_maxima.data(), rows * 3, ReduceOp::max);
    // at: the lowest place among the ranks that hold the max |diff|
    for (int r = 0; r < rows; r++)
        if (maxima[r * 3] != total_maxima[r * 3]) at[r] = 1e300;
    comm.reduce(at.data(), total_at.data(), rows, ReduceOp::min, 0);
    if (comm.rank() != 0) return "";

    char line[160];
    std::ostringstream out;
    out << title << "\n";
    std::snprintf(line, sizeof line, "%-12s %7s %9s %9s %9s %9s %9s\n", "", "n", "MAE", "RMSE", "max", "rel max", "rel RMSE");
    out << line;
    auto row = [&](int r, const std::string& name, const char* unit) {
        const double* sum = &total_sums[r * 4];
        const double* max = &total_maxima[r * 3];
        if (sum[0] == 0) return;
        const double rmse = std::sqrt(sum[2] / sum[0]);
        const bool energy = r == ENE || r == EPA;
        const bool scaled = !energy || residuals;
        // formats value / scale, or a dash when scale is 0
        auto relative = [](double value, double scale) {
            char field[16];
            if (scale > 0) std::snprintf(field, sizeof field, " %9.2e", value / scale);
            else std::snprintf(field, sizeof field, " %8s—", "");
            return std::string(field);
        };
        const std::string rel_max = relative(max[0], !scaled ? 0.0 : energy ? max[1] + max[2] : max[1]);
        const std::string rel_rmse = relative(rmse, scaled ? std::sqrt(sum[3] / sum[0]) : 0.0);
        std::snprintf(line, sizeof line, "%-12s %7ld %9.2e %9.2e %9.2e%s%s  %s\n", name.c_str(), (long) sum[0], sum[1] / sum[0], rmse, max[0], rel_max.c_str(), rel_rmse.c_str(), unit);
        out << line;
    };
    row(ENE, "energy", "eV");
    row(EPA, "energy/atom", "eV/atom");
    row(FRC, "force", "eV/A");
    for (int t = 0; t < species_count; t++)
        row(KINDS + t, "  " + (t < (int) species.size() ? species[t] : "type " + std::to_string(t)), "eV/A");
    row(VIR, "virial", "eV");
    if (total_sums[EPA * 4]) {
        std::snprintf(line, sizeof line, "max energy/atom error: %.2e eV/atom (structure %ld)\n", total_maxima[EPA * 3], (long) (total_at[EPA] / 1e6));
        out << line;
    }
    if (total_sums[FRC * 4]) {
        std::snprintf(line, sizeof line, "max force error: %.2e eV/A (structure %ld, atom %ld)\n", total_maxima[FRC * 3], (long) (total_at[FRC] / 1e6), (long) std::fmod(total_at[FRC], 1e6));
        out << line;
    }
    return out.str();
}
