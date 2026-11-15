/* -*- c++ -*- ----------------------------------------------------------
   The base of PairMTP and PairETN (see pair.h).
------------------------------------------------------------------------- */

#include "pair.h"

#include <stdexcept>

void Pair::eval_grad(const NeighList&, double*, double*, double*) {
    throw std::runtime_error("eval_grad needs a training class, MTPTraining or ETNTraining");
}

void Pair::eval_loss_grad(const NeighList&, double, const double*, const double*, double*) {
    throw std::runtime_error("eval_loss_grad needs a training class, MTPTraining or ETNTraining");
}

void Pair::eval_radial_basis(double dist, double* vals_out, double* ders_out) {
    radial_basis->calc_radial_basis_ders(dist);
    std::copy(radial_basis->radial_basis_vals.begin(), radial_basis->radial_basis_vals.end(), vals_out);
    std::copy(radial_basis->radial_basis_ders.begin(), radial_basis->radial_basis_ders.end(), ders_out);
}
