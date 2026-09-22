/* -*- c++ -*- ----------------------------------------------------------
   Shared pieces of the pybind11 layer.

   One binding unit per source pair, bound in the same order:
     mtp_potential_bindings.cpp     <- mtp_potential.{h,cpp}
     mtp_extrapolation_bindings.cpp <- mtp_extrapolation.{h,cpp}
     mtp_training_bindings.cpp      <- mtp_training.{h,cpp}
------------------------------------------------------------------------- */

#pragma once

#include "mtp_training.h"

#include <stdexcept>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

using IntArray = py::array_t<int, py::array::c_style | py::array::forcecast>;
using DoubleArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

// Owns the numpy arrays a NeighList points into.
class PyNeighbors {
  public:
    PyNeighbors(IntArray types, IntArray ilist, IntArray numneigh, IntArray firstneigh, DoubleArray displacements)
        : types_(types), ilist_(ilist), numneigh_(numneigh), firstneigh_(firstneigh), displacements_(displacements) {
        if (ilist_.shape(0) != numneigh_.shape(0))
            throw std::runtime_error("NeighList: ilist and numneigh must have the same length");
        if (displacements_.ndim() != 2 || displacements_.shape(1) != 3)
            throw std::runtime_error("NeighList: displacements must have shape (n_pairs, 3)");
        if (firstneigh_.shape(0) != displacements_.shape(0))
            throw std::runtime_error("NeighList: firstneigh and displacements must have the same length");

        list_.n_atoms = (int) types_.shape(0);
        list_.types = types_.data();
        list_.inum = (int) ilist_.shape(0);
        list_.ilist = ilist_.data();
        list_.numneigh = numneigh_.data();
        list_.firstneigh = firstneigh_.data();
        list_.displacements = displacements_.data();
    }

    const NeighList& view() const { return list_; }
    int n_atoms() const { return list_.n_atoms; }
    int inum() const { return list_.inum; }
    IntArray types() const { return types_; }
    DoubleArray displacements() const { return displacements_; }

  private:
    IntArray types_, ilist_, numneigh_, firstneigh_;
    DoubleArray displacements_;
    NeighList list_;
};

inline DoubleArray zeros(std::vector<py::ssize_t> shape) {
    DoubleArray a(shape);
    std::fill(a.mutable_data(), a.mutable_data() + a.size(), 0.0);
    return a;
}

// Binds fn under two signatures: one taking a NeighList, and one taking the
// five loose arrays the older API passes. `extra` carries the py::arg entries
// for any trailing parameters.
template <class Cls, class Self, class Ret, class... Args, class... Extra>
void def_neighbors(Cls& cls, const char* name, Ret (*fn)(Self&, const PyNeighbors&, Args...), const char* doc, Extra... extra) {
    cls.def(name, fn, py::arg("neighbors"), extra..., doc);
    cls.def(
        name,
        [fn](Self& self, IntArray types, IntArray ilist, IntArray numneigh, IntArray firstneigh, DoubleArray displacements, Args... args) {
            return fn(self, PyNeighbors(types, ilist, numneigh, firstneigh, displacements), args...);
        },
        py::arg("types"), py::arg("ilist"), py::arg("numneigh"), py::arg("firstneigh"), py::arg("displacements"), extra...);
}

void bind_neigh_list(py::module_& m);
void bind_potential(py::module_& m);
void bind_extrapolation(py::module_& m);
void bind_training(py::module_& m);
