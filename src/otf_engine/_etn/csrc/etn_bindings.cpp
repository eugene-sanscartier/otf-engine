/* -*- c++ -*- ----------------------------------------------------------
   Python bindings for ETN (see etn.h). Its eval_grad and
   eval_loss_grad are Pair's bindings.
------------------------------------------------------------------------- */

#include "bindings_common.h"
#include "etn.h"

void bind_etn(py::module_& m) {
    py::class_<ETN, PairETN>(m, "ETN", "An ETN potential that also differentiates site energies, and losses on its energy, forces and virial, w.r.t. its coefficients.")
        .def(py::init<const std::string&>(), py::arg("filename"));
}
