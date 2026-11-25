/* -*- c++ -*- ----------------------------------------------------------
   The _mtp_ext module, with one binding unit per source pair, in source
   order. It first imports _core_ext, which defines Pair and NeighList.
------------------------------------------------------------------------- */

#include "bindings_common.h"

PYBIND11_MODULE(_mtp_ext, m) {
    m.doc() = "MTP potential, its extrapolation grade and its fit.";
    py::module_::import("otf_engine._core._core_ext");

    bind_pair_mtp(m);
    bind_pair_mtp_extrapolation(m);
    bind_mtp(m);
    bind_mtp_fit(m);
}
