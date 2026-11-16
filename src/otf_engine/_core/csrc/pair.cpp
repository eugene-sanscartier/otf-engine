/* -*- c++ -*- ----------------------------------------------------------
   The base of PairMTP and PairETN (see pair.h).
------------------------------------------------------------------------- */

#include "pair.h"

#include "etn_radial_basis.h"
#include "text_file_reader.h"

#include <stdexcept>

void Pair::eval_grad(const NeighList&, double*, double*, double*) {
    throw std::runtime_error("eval_grad needs an MTP or ETN");
}

void Pair::eval_loss_grad(const NeighList&, double, const double*, const double*, double*) {
    throw std::runtime_error("eval_loss_grad needs an MTP or ETN");
}

void Pair::read_radial_basis(TextFileReader& tfr) {
    if (tfr.keyword() != "radial_basis_type")
        throw std::runtime_error("potential file: no radial basis set type is specified");
    std::string type;
    tfr.rest() >> type;

    // A scaling line in the basis block is accepted and ignored: the potential's scaling applies
    tfr.advance();
    if (tfr.keyword() == "scaling") tfr.advance();

    // Lower cutoff — accepts both 'min_dist' and 'min_val'
    if (tfr.keyword() != "min_val" && tfr.keyword() != "min_dist")
        throw std::runtime_error("potential file: expected min_dist, got '" + tfr.keyword() + "'");
    double min_dist = 0;
    tfr.rest() >> min_dist;

    // Upper cutoff — accepts both 'max_dist' and 'max_val'
    tfr.advance();
    if (tfr.keyword() != "max_val" && tfr.keyword() != "max_dist")
        throw std::runtime_error("potential file: expected max_dist, got '" + tfr.keyword() + "'");
    double max_dist = 0;
    tfr.rest() >> max_dist;

    const int size = tfr.next_int("radial_basis_size");
    set_radial_basis(type, min_dist, max_dist, size);
}

void Pair::set_radial_basis(const std::string& type, double min_dist, double max_dist, int size) {
    RadialMTPBasis* basis;
    if (type == "RBChebyshev")
        basis = new RBChebyshev(min_dist, max_dist, size);
    else if (type == "RBCinf")
        basis = new RBCinf(min_dist, max_dist, size);
    else
        throw std::runtime_error("the radial basis type '" + type + "' is neither RBChebyshev nor RBCinf");

    delete radial_basis;
    radial_basis = basis;
    radial_basis_type = type;
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
