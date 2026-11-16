/* -*- c++ -*- ----------------------------------------------------------
   The base of PairMTP and PairETN (see pair.h).
------------------------------------------------------------------------- */

#include "pair.h"

#include "etn_radial_basis.h"
#include "text_file_reader.h"

#include <stdexcept>

void Pair::eval_grad(const NeighList&, double*, double*, double*) {
    throw std::runtime_error("eval_grad needs a PairMTPGrad or PairETNGrad");
}

void Pair::eval_loss_grad(const NeighList&, double, const double*, const double*, double*) {
    throw std::runtime_error("eval_loss_grad needs a PairMTPGrad or PairETNGrad");
}

void Pair::read_radial_basis(TextFileReader& tfr) {
    if (tfr.keyword() != "radial_basis_type")
        throw std::runtime_error("potential file: no radial basis set type is specified");
    tfr.rest() >> radial_basis_type;

    if (radial_basis_type == "RBChebyshev")
        radial_basis = new RBChebyshev(tfr);
    else if (radial_basis_type == "RBCinf")
        radial_basis = new RBCinf(tfr);
    else
        throw std::runtime_error("potential file: the radial basis type '" + radial_basis_type + "' is neither RBChebyshev nor RBCinf");

    radial_basis->scaling = scaling;
    radial_basis_size = radial_basis->size;
    min_cutoff = radial_basis->min_cutoff;
    max_cutoff = radial_basis->max_cutoff;
    max_cutoff_sq = max_cutoff * max_cutoff;
}

void Pair::eval_radial_basis(double dist, double* vals_out, double* ders_out) {
    radial_basis->calc_radial_basis_ders(dist);
    std::copy(radial_basis->radial_basis_vals.begin(), radial_basis->radial_basis_vals.end(), vals_out);
    std::copy(radial_basis->radial_basis_ders.begin(), radial_basis->radial_basis_ders.end(), ders_out);
}
