/* -*- c++ -*- ----------------------------------------------------------
   MaxVol's sweep: the extrapolation grade of a block of equations,
   grades[r] = max_i |rows[r] . invA[i]|, single-threaded through Eigen.

   maxvol_sweep.cpp is compiled twice: for x86-64-v3 (AVX2, FMA) as
   sweep_grades_v3, and with the portable flags as sweep_grades_portable,
   each under its own name for the Eigen namespace so the two never share
   an instantiation. MaxVol::grade picks one when the process starts.
------------------------------------------------------------------------- */

#pragma once

// block holds sweep_block_rows * n doubles.
constexpr int sweep_block_rows = 1024;

void sweep_grades_portable(const double* rows, int count, const double* invA, int n, double* block, double* grades);
#ifdef MTP_SWEEP_V3
void sweep_grades_v3(const double* rows, int count, const double* invA, int n, double* block, double* grades);
#endif
