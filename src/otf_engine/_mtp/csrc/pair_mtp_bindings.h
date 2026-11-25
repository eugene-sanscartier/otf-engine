/* -*- c++ -*- ----------------------------------------------------------
   Constructors shared by the Python classes PairMTP, PairMTPExtrapolation
   and MTP.
------------------------------------------------------------------------- */

#pragma once

#include "bindings_common.h"
#include "mtp_basis.h"

#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

// Defines the constructors from an .mtp file, and of the untrained MTP of a level or of basis functions.
template <class T, class... Options>
void def_pair_mtp_init(py::class_<T, Options...>& cls) {
    cls.def(py::init<const std::string&>(), py::arg("filename"), "Load an MTP potential from a .mtp file (version 1.1.0).")
        .def(py::init([](std::optional<int> level, std::optional<std::vector<std::vector<int>>> basis_functions, int species_count, const std::string& radial_basis_type, double min_dist, double max_dist, int radial_basis_size) {
                 if (level.has_value() == basis_functions.has_value())
                     throw std::invalid_argument("give either level or basis_functions");
                 return new T(level ? level_basis_functions(*level) : *basis_functions, species_count, radial_basis_type, min_dist, max_dist, radial_basis_size);
             }),
             py::kw_only(), py::arg("level") = py::none(), py::arg("basis_functions") = py::none(), py::arg("species_count") = 0, py::arg("radial_basis_type") = "RBChebyshev", py::arg("min_dist"), py::arg("max_dist"), py::arg("radial_basis_size"),
             R"doc(
Build the untrained MTP of a level, 6 to 28, or of basis_functions, with the defaults of an untrained .mtp file.

A basis function holds n radial indices, then an n x n symmetric matrix row by row.
Entry (i, i) is the tensor rank of moment i, and entry (i, j) the number of indices
moments i and j contract. Every index is contracted. level_basis_functions(level) gives
the basis functions of a level. radial_funcs_count is 1 + the largest radial index.

Args:
    species_count: number of species; 0 leaves the count to training.
)doc");
}
