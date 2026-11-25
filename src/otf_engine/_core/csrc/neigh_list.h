/* -*- c++ -*- ----------------------------------------------------------
   The neighbor list that Pair::compute takes.
------------------------------------------------------------------------- */

#pragma once

// The neighbors of the central atoms are contiguous in ilist order: those of ilist[ii] start at offset
// sum(numneigh[0..ii)) in firstneigh and in displacements.
// The caller supplies the displacements, already corrected for periodic images. The list has no ghost atoms.
struct NeighList {
    int n_atoms;
    const int* types;             // [n_atoms] 0-based species
    int inum;
    const int* ilist;             // [inum] indices of the central atoms
    const int* numneigh;          // [inum] neighbor count of each central atom
    const int* firstneigh;        // [sum(numneigh)] neighbor atom indices
    const double* displacements;  // [sum(numneigh) * 3] r_j - r_i, corrected for periodic images
};
