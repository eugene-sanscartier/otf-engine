/* -*- c++ -*- ----------------------------------------------------------
   The _etn_ext module: one binding unit per source pair, in source order.
   Pair and NeighList are _core_ext's, which it imports first.
------------------------------------------------------------------------- */

#include "bindings_common.h"

PYBIND11_MODULE(_etn_ext, m) {
    m.doc() = "Standalone ETN potential bindings (no LAMMPS, no mlip-4).";
    py::module_::import("otf_engine._core._core_ext");

    bind_etn_harmonics(m);
    bind_pair_etn(m);
    bind_pair_etn_extrapolation(m);
    bind_etn(m);
    bind_etn_fit(m);
}
