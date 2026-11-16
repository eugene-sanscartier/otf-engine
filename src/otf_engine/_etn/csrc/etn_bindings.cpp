/* -*- c++ -*- ----------------------------------------------------------
   Python bindings for ETN (see etn.h). Its eval_grad and
   eval_loss_grad are Pair's bindings.
------------------------------------------------------------------------- */

#include "etn.h"
#include "pair_etn_bindings.h"

void bind_etn(py::module_& m) {
    auto cls = py::class_<ETN, PairETN>(m, "ETN", "An ETN potential that also differentiates site energies, and losses on its energy, forces and virial, w.r.t. its coefficients.");
    def_pair_etn_init(cls);
}
