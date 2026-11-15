/* -*- c++ -*- ----------------------------------------------------------
   A structure to fit, which MTPTrainer and ETNTrainer take.
------------------------------------------------------------------------- */

#pragma once

#include "neigh_list.h"

#include <vector>

// Virial component of each entry of the symmetric 3x3 stress.
constexpr int VIRIAL_INDEX[3][3] = {{0, 3, 4}, {3, 1, 5}, {4, 5, 2}};

// A structure to fit: its neighbor list at the potential's cutoff and its reference values.
struct TrainingStructure {
    std::vector<int> types, ilist, numneigh, firstneigh;
    std::vector<double> displacements;    // [sum(numneigh) * 3]

    bool has_energy = false, has_forces = false, has_virial = false;
    double energy = 0.0;
    std::vector<double> forces;    // [n_atoms * 3]
    double virial[6] = {};         // xx,yy,zz,xy,xz,yz in eV, mlip-3's PlusStress
    int number = 0;                // the caller's number for it, which the error table names
    std::vector<double> species_factors;    // force-weight factor by species index under balance_species

    int n_atoms() const { return (int) types.size(); }
    NeighList list() const {
        return NeighList{n_atoms(), types.data(), (int) ilist.size(), ilist.data(), numneigh.data(), firstneigh.data(), displacements.data()};
    }
};
