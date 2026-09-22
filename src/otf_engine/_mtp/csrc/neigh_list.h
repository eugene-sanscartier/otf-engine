/* -*- c++ -*- ----------------------------------------------------------
   Neighbor list passed to PairMTP::compute.
   Stands in for LAMMPS NeighList + Atom, which the reference reads through
   `list->` and `atom->`.
------------------------------------------------------------------------- */

#pragma once

// Neighbor blocks are flat and contiguous in ilist order: the block for
// central atom ilist[ii] starts at offset sum(numneigh[0..ii)) in both
// firstneigh and displacements.
//
// Unlike LAMMPS, displacements are supplied by the caller rather than derived
// from ghost-atom positions, so there is no minimum-image convention here.
struct NeighList {
    int n_atoms;                  // total number of atoms
    const int* types;             // [n_atoms] 0-indexed MTP species
    int inum;                     // number of central atoms
    const int* ilist;             // [inum] indices of central atoms
    const int* numneigh;          // [inum] neighbor count per central atom
    const int* firstneigh;        // [sum(numneigh)] neighbor atom indices
    const double* displacements;  // [sum(numneigh) * 3] r_j - r_i, PBC-corrected
};
