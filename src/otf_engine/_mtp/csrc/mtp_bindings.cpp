/* -*- c++ -*- ----------------------------------------------------------
   The _mtp_ext module: the neighbor list, then one binding unit per source
   pair, in source order.
------------------------------------------------------------------------- */

#include "bindings_common.h"

void bind_neigh_list(py::module_& m) {
    py::class_<PyNeighbors>(m, "NeighList", R"doc(
Neighbor list for one structure.

types         : ndarray int32   (n_atoms)         — 0-indexed MTP species
ilist         : ndarray int32   (inum)            — indices of central atoms
numneigh      : ndarray int32   (inum)            — neighbor count per central atom
firstneigh    : ndarray int32   (sum(numneigh))   — flat neighbor indices
displacements : ndarray float64 (sum(numneigh),3) — PBC-corrected r_j - r_i
)doc")
        .def(py::init<IntArray, IntArray, IntArray, IntArray, DoubleArray>(),
             py::arg("types"), py::arg("ilist"), py::arg("numneigh"), py::arg("firstneigh"), py::arg("displacements"))
        .def_property_readonly("n_atoms", &PyNeighbors::n_atoms)
        .def_property_readonly("inum", &PyNeighbors::inum)
        .def_property_readonly("types", &PyNeighbors::types)
        .def_property_readonly("displacements", &PyNeighbors::displacements);
}

PYBIND11_MODULE(_mtp_ext, m) {
    m.doc() = "Standalone MTP potential bindings (no LAMMPS).";

    bind_neigh_list(m);
    bind_potential(m);
    bind_extrapolation(m);
    bind_training(m);
}
