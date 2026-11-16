/* -*- c++ -*- ----------------------------------------------------------
   Python bindings for PairETNGrad (see pair_etn_grad.h). Its eval_grad and
   eval_loss_grad are Pair's bindings.
------------------------------------------------------------------------- */

#include "bindings_common.h"
#include "pair_etn_grad.h"

void bind_pair_etn_grad(py::module_& m) {
    py::class_<PairETNGrad, PairETN>(m, "PairETNGrad", "An ETN potential that also differentiates site energies, and losses on its energy, forces and virial, w.r.t. its coefficients.")
        .def(py::init<const std::string&>(), py::arg("filename"));
}
