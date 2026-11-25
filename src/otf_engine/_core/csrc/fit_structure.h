/* -*- c++ -*- ----------------------------------------------------------
   The training structure of MTPFit and ETNFit.
------------------------------------------------------------------------- */

#pragma once

#include "neigh_list.h"

#include <algorithm>
#include <vector>

// Virial component of each entry of the symmetric 3x3 stress.
constexpr int VIRIAL_INDEX[3][3] = {{0, 3, 4}, {3, 1, 5}, {4, 5, 2}};

// A training structure: its neighbor list at the cutoff of the potential, and its reference values.
struct FitStructure {
    std::vector<int> types, ilist, numneigh, firstneigh;
    std::vector<double> displacements;    // [sum(numneigh) * 3]

    bool has_energy = false, has_forces = false, has_virial = false;
    double energy = 0.0;
    std::vector<double> forces;    // [n_atoms * 3]
    double virial[6] = {};         // xx,yy,zz,xy,xz,yz in eV
    int number = 0;                // number of the structure in the error table
    std::vector<double> species_factors;    // [species] force-weight factor of each species under balance_species

    // Sets species_factors: for N atoms of k species, n_s of them of species s, an atom of species s weighs N / (k n_s).
    void balance_species() {
        std::vector<int> counts;
        for (int i : ilist) {
            if (types[i] >= (int) counts.size()) counts.resize(types[i] + 1, 0);
            counts[types[i]]++;
        }
        const int k = (int) std::count_if(counts.begin(), counts.end(), [](int n) { return n > 0; });
        species_factors.assign(counts.size(), 0.0);
        for (size_t t = 0; t < counts.size(); t++)
            if (counts[t] > 0) species_factors[t] = (double) ilist.size() / (k * counts[t]);
    }

    int n_atoms() const { return (int) types.size(); }
    NeighList list() const {
        return NeighList{n_atoms(), types.data(), (int) ilist.size(), ilist.data(), numneigh.data(), firstneigh.data(), displacements.data()};
    }
};
