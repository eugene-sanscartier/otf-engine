/* -*- c++ -*- ----------------------------------------------------------
   The constructors of PairETN's Python classes, which ETN shares.
------------------------------------------------------------------------- */

#pragma once

#include "bindings_common.h"

#include <string>
#include <vector>

// From an .etn file, or the untrained ETN of a shape given by the file's fields.
template <class T, class... Options>
void def_pair_etn_init(py::class_<T, Options...>& cls) {
    cls.def(py::init<const std::string&>(), py::arg("filename"), "Load an ETN potential from an .etn file (version 1.0.0).")
        .def(py::init<std::vector<int>, std::vector<int>, std::vector<int>, std::vector<std::vector<int>>, int, const std::string&, double, double, int>(), py::kw_only(), py::arg("species_ranks"), py::arg("radial_funcs_counts"), py::arg("radial_ranks"), py::arg("bond_ranks"), py::arg("species_count") = 0, py::arg("radial_basis_type") = "RBCinf", py::arg("min_dist"), py::arg("max_dist"), py::arg("radial_basis_size"),
             R"doc(
The untrained ETN of this shape, as an .etn file that stops after bond_ranks gives it.

species_ranks, radial_funcs_counts and radial_ranks hold one entry per l of the input;
bond_ranks one list per bond, the ranks per l, [1] at both ends, so core_count is
len(bond_ranks) - 1. species_count 0 leaves the count to training.
)doc");
}
