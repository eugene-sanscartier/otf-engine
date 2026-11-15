/* -*- c++ -*- ----------------------------------------------------------
   Python bindings for ETNTraining (see etn_training.h). Its eval_grad and
   eval_loss_grad are Pair's bindings.

   mlip-4 takes these derivatives by differentiating its program; nothing
   here binds a function of its.
------------------------------------------------------------------------- */

#include "bindings_common.h"
#include "etn_training.h"

void bind_etn_training(py::module_& m) {
    py::class_<ETNTraining, PairETN>(m, "ETNTraining", "An ETN potential that also differentiates site energies, and losses on its energy, forces and virial, w.r.t. its coefficients.")
        .def(py::init<const std::string&>(), py::arg("filename"));
}
