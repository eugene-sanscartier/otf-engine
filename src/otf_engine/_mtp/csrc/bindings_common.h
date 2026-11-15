/* -*- c++ -*- ----------------------------------------------------------
   Shared pieces of the pybind11 layer.

   One binding unit per source pair, bound in the same order:
     mtp_potential_bindings.cpp     <- mtp_potential.{h,cpp}
     mtp_extrapolation_bindings.cpp <- mtp_extrapolation.{h,cpp}
     mtp_training_bindings.cpp      <- mtp_training.{h,cpp}
     mtp_trainer_bindings.cpp       <- mtp_trainer.{h,cpp}
     maxvol_bindings.cpp            <- maxvol.{h,cpp}
   and, in the _etn_ext module under ../../_etn/csrc/:
     etn_harmonics_bindings.cpp     <- etn_harmonics.{h,cpp}
     etn_potential_bindings.cpp     <- etn_potential.{h,cpp}
     etn_training_bindings.cpp      <- etn_training.{h,cpp}
     etn_trainer_bindings.cpp       <- etn_trainer.{h,cpp}
   Built with MTP_MPI, mtp_trainer_bindings.cpp and maxvol_bindings.cpp are
   the _mtp_mpi module, and etn_trainer_bindings.cpp the _etn_mpi module.
------------------------------------------------------------------------- */

#pragma once

#include "mtp_extrapolation.h"
#include "mtp_training.h"
#include "training_structure.h"

#ifdef MTP_MPI
#include <mpi.h>
#endif

#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
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

// The results PairMTP::compute or PairETN::compute leaves in the potential, copied out:
// "energy", "forces" (n_atoms, 3), "virials" (6) and "eatom" (n_atoms).
template <class Potential>
py::dict potential_results(const Potential& self, int n_atoms) {
    py::dict result;
    result["energy"] = self.get_energy();
    result["forces"] = DoubleArray({n_atoms, 3}, self.get_forces());
    result["virials"] = DoubleArray({6}, self.get_virial());
    result["eatom"] = DoubleArray({n_atoms}, self.get_eatom());
    return result;
}

// A training structure from (NeighList, energy|None, forces (n_atoms, 3)|None, virial (6)|None, number).
inline TrainingStructure make_structure(py::handle item) {
    py::tuple t = py::reinterpret_borrow<py::tuple>(item);
    if (t.size() != 5)
        throw std::runtime_error("a training structure is (NeighList, energy, forces, virial, number)");
    const NeighList& list = t[0].cast<const PyNeighbors&>().view();
    py::object energy = t[1], forces = t[2], virial = t[3];
    const int n_pairs = std::accumulate(list.numneigh, list.numneigh + list.inum, 0);

    TrainingStructure s;
    s.types.assign(list.types, list.types + list.n_atoms);
    s.ilist.assign(list.ilist, list.ilist + list.inum);
    s.numneigh.assign(list.numneigh, list.numneigh + list.inum);
    s.firstneigh.assign(list.firstneigh, list.firstneigh + n_pairs);
    s.displacements.assign(list.displacements, list.displacements + (size_t) n_pairs * 3);

    if (!energy.is_none()) {
        s.energy = energy.cast<double>();
        s.has_energy = true;
    }
    if (!forces.is_none()) {
        DoubleArray f = forces.cast<DoubleArray>();
        if (f.ndim() != 2 || f.shape(0) != list.n_atoms || f.shape(1) != 3)
            throw std::runtime_error("a training structure's forces must have shape (n_atoms, 3)");
        s.forces.assign(f.data(), f.data() + f.size());
        s.has_forces = true;
    }
    if (!virial.is_none()) {
        DoubleArray v = virial.cast<DoubleArray>();
        if (v.size() != 6)
            throw std::runtime_error("a training structure's virial must have 6 components");
        std::copy(v.data(), v.data() + 6, s.virial);
        s.has_virial = true;
    }
    s.number = t[4].cast<int>();
    return s;
}

// The stream a trainer's log option names: stdout (or empty), none, or a file, opened into file.
inline std::ostream* open_log(const std::string& target, std::ofstream& file) {
    if (target == "stdout" || target.empty()) return &std::cout;
    if (target == "none") return nullptr;
    file.open(target);
    if (!file)
        throw std::runtime_error("cannot open log '" + target + "'");
    return &file;
}

#ifdef MTP_MPI
// Raises error on every rank of comm, or else the error of the lowest rank that has one.
inline void raise_together(MPI_Comm comm, const std::string& error) {
    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    const int mine = error.empty() ? size : rank;
    int failed = size;
    MPI_Allreduce(&mine, &failed, 1, MPI_INT, MPI_MIN, comm);
    if (failed == size) return;

    std::string message = rank == failed ? error : std::string();
    int length = (int) message.size();
    MPI_Bcast(&length, 1, MPI_INT, failed, comm);
    message.resize(length);
    MPI_Bcast(message.data(), length, MPI_CHAR, failed, comm);
    throw std::runtime_error(message);
}
#endif

// Raises unless every atom of nb has one of the potential's species, which the
// kernels index their coefficients with.
template <class Potential>
void check_species(const Potential& self, const PyNeighbors& nb) {
    const NeighList& list = nb.view();
    for (int i = 0; i < list.n_atoms; i++)
        if (list.types[i] < 0 || list.types[i] >= self.get_species_count())
            throw std::runtime_error("NeighList: atom " + std::to_string(i) + " has species " + std::to_string(list.types[i]) + ", outside the potential's " + std::to_string(self.get_species_count()) + " species");
}

// Binds fn under two signatures: one taking a NeighList, and one taking the
// five loose arrays the older API passes, both checking the species first.
// `extra` carries the py::arg entries for any trailing parameters.
template <class Cls, class Self, class Ret, class... Args, class... Extra>
void def_neighbors(Cls& cls, const char* name, Ret (*fn)(Self&, const PyNeighbors&, Args...), const char* doc, Extra... extra) {
    cls.def(
        name,
        [fn](Self& self, const PyNeighbors& nb, Args... args) {
            check_species(self, nb);
            return fn(self, nb, args...);
        },
        py::arg("neighbors"), extra..., doc);
    cls.def(
        name,
        [fn](Self& self, IntArray types, IntArray ilist, IntArray numneigh, IntArray firstneigh, DoubleArray displacements, Args... args) {
            const PyNeighbors nb(types, ilist, numneigh, firstneigh, displacements);
            check_species(self, nb);
            return fn(self, nb, args...);
        },
        py::arg("types"), py::arg("ilist"), py::arg("numneigh"), py::arg("firstneigh"), py::arg("displacements"), extra...);
}

void bind_neigh_list(py::module_& m);
void bind_potential(py::module_& m);
void bind_extrapolation(py::module_& m);
void bind_training(py::module_& m);
void bind_trainer(py::module_& m);
void bind_maxvol(py::module_& m);
void bind_etn_harmonics(py::module_& m);
void bind_etn_potential(py::module_& m);
void bind_etn_training(py::module_& m);
void bind_etn_trainer(py::module_& m);
