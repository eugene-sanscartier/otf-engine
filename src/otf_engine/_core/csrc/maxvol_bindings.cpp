/* -*- c++ -*- ----------------------------------------------------------
   Python bindings for Equations and MaxVol (see maxvol.h).
------------------------------------------------------------------------- */

#include "bindings_common.h"
#include "maxvol.h"

#include <cmath>

static void check_width(const MaxVol& self, const Equations& eqns) {
    if (eqns.coeff_count != self.n)
        throw std::runtime_error("MaxVol: equations of " + std::to_string(eqns.coeff_count) + " coefficients against an active set of " + std::to_string(self.n));
}

using BoolArray = py::array_t<bool, py::array::c_style | py::array::forcecast>;

void bind_maxvol(py::module_& m) {
    py::class_<Equations>(m, "Equations", "The MaxVol equations of a set of structures, under one set of selection weights.")
        .def(py::init<int, double, double, double, double, int>(),
             py::arg("coeff_count"), py::arg("energy_weight"), py::arg("force_weight"), py::arg("stress_weight"), py::arg("site_en_weight"), py::arg("weight_scaling"))
        .def("add",
             [](Equations& self, Pair& pot, const PyNeighbors& nb, int structure) {
                 check_species(pot, nb);
                 if (pot.coeff_count() != self.coeff_count)
                     throw std::runtime_error("Equations: a potential of " + std::to_string(pot.coeff_count()) + " coefficients, not " + std::to_string(self.coeff_count));
                 py::gil_scoped_release unlocked;
                 self.add(pot, nb.view(), structure);
             },
             py::arg("potential"), py::arg("neighbors"), py::arg("structure") = -1,
             "Append the equations of one structure, as the potential's current coefficients give them, numbered structure (by default the next).")
        .def("subset",
             [](const Equations& self, BoolArray keep) {
                 if (keep.size() != self.size())
                     throw std::runtime_error("Equations.subset: keep must have one entry per equation");
                 return self.subset(keep.data());
             },
             py::arg("keep"), "The equations whose keep entry is true, under the same structure numbering.")
        .def("__len__", &Equations::size)
        .def_readonly("structure_count", &Equations::structure_count)
        .def_property_readonly("structure_indices", [](const Equations& self) { return IntArray({self.size()}, self.structure_indices.data()); },
                               "Which added structure each equation belongs to.")
        .def_property_readonly("equation_indices", [](const Equations& self) { return IntArray({self.size()}, self.equation_indices.data()); },
                               "The index of each equation within its structure.");

    py::class_<MaxVol>(m, "MaxVol", "A MaxVol active set: A, invA = A^-T, and the structure and equation each active row came from.")
        .def(py::init<int, double, double, std::optional<double>>(), py::arg("n"), py::arg("init_scale") = 1e-6, py::arg("threshold") = 1.001, py::arg("threshold_swap") = py::none())
        .def(py::init([](DoubleArray A, DoubleArray invA, double threshold, std::optional<double> threshold_swap) {
                 if (A.ndim() != 2 || A.shape(0) != A.shape(1) || invA.ndim() != 2 || invA.shape(0) != A.shape(0) || invA.shape(1) != A.shape(0))
                     throw std::runtime_error("MaxVol: A and invA must be square and of one size");
                 return MaxVol((int) A.shape(0), A.data(), invA.data(), threshold, threshold_swap);
             }),
             py::arg("A"), py::arg("invA"), py::arg("threshold") = 1.001, py::arg("threshold_swap") = py::none(), "Restore an active set stored as A and invA.")
        .def("grade",
             [](MaxVol& self, const Equations& eqns) {
                 check_width(self, eqns);
                 auto grades = zeros({eqns.size()});
                 double* grades_ptr = grades.mutable_data();
                 {
                     py::gil_scoped_release unlocked;
                     self.grade(eqns.grads.data(), eqns.size(), grades_ptr);
                 }
                 return grades;
             },
             py::arg("equations"), "The extrapolation grade of each equation, max_i |v . invA[i]|.")
        .def("maximize_volume",
             [](MaxVol& self, Equations& pool, int pool_id, py::object comm, int max_swaps) {
                 check_width(self, pool);
                 ProcessRanks process("maximize_volume", 1, comm);
                 py::gil_scoped_release unlocked;
                 self.maximize_volume(pool, pool_id, process[0], max_swaps);
             },
             py::arg("equations"), py::arg("pool_id") = -1, py::arg("comm") = py::none(), py::arg("max_swaps") = 99999, R"doc(
Swap equations of the pool into the active set, the highest-grading first, until none grades above threshold.

comm : an mpi4py communicator to search over its ranks. Every rank passes the same active set and
       its own share of the pool, numbered by structure across the whole pool, and ends with the
       same A, invA and provenance.
)doc")
        .def("restore_active",
             [](MaxVol& self, IntArray cfg_indices, IntArray eqn_indices, int pool_id) {
                 if (cfg_indices.size() != self.n || eqn_indices.size() != self.n)
                     throw std::runtime_error("MaxVol.restore_active: one index per active row");
                 self.restore_active(cfg_indices.data(), eqn_indices.data(), pool_id);
             },
             py::arg("cfg_indices"), py::arg("eqn_indices"), py::arg("pool_id"), "Restore the provenance of each active row from saved indices.")
        .def_readwrite("threshold", &MaxVol::threshold)
        .def_readwrite("threshold_swap", &MaxVol::threshold_swap, "Once a search has swapped, it goes on while a grade exceeds this; None, threshold.")
        .def_property_readonly("A", [](const MaxVol& self) { return DoubleArray({self.n, self.n}, self.A.data()); })
        .def_property_readonly("invA", [](const MaxVol& self) { return DoubleArray({self.n, self.n}, self.invA.data()); })
        .def_property_readonly("active_pool_ids", [](const MaxVol& self) { return IntArray({self.n}, self.active_pool_ids.data()); })
        .def_property_readonly("active_struct_indices", [](const MaxVol& self) { return IntArray({self.n}, self.active_struct_indices.data()); })
        .def_property_readonly("active_eqn_indices", [](const MaxVol& self) { return IntArray({self.n}, self.active_eqn_indices.data()); });
}
