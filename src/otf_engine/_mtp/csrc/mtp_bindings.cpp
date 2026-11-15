/* -*- c++ -*- ----------------------------------------------------------
   The _mtp_ext module: one binding unit per source pair, in source order.
   Pair and NeighList are _core_ext's, which it imports first.
------------------------------------------------------------------------- */

#include "bindings_common.h"

PYBIND11_MODULE(_mtp_ext, m) {
    m.doc() = "Standalone MTP potential bindings (no LAMMPS).";
    py::module_::import("otf_engine._core._core_ext");

    bind_pair_mtp(m);
    bind_pair_mtp_extrapolation(m);
    bind_mtp_training(m);
    bind_mtp_trainer(m);
}
