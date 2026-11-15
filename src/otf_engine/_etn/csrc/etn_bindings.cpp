/* -*- c++ -*- ----------------------------------------------------------
   The _etn_ext module: one binding unit per source pair, in source order.
   The NeighList type is _mtp_ext's, which must be imported first.
------------------------------------------------------------------------- */

#include "bindings_common.h"

PYBIND11_MODULE(_etn_ext, m) {
    m.doc() = "Standalone ETN potential bindings (no LAMMPS, no mlip-4).";

    bind_etn_harmonics(m);
    bind_etn_potential(m);
    bind_etn_training(m);
}
