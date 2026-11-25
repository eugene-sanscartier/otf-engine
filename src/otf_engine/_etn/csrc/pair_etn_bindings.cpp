/* -*- c++ -*- ----------------------------------------------------------
   Python bindings for PairETN (see pair_etn.h).
------------------------------------------------------------------------- */

#include "pair_etn_bindings.h"
#include "pair_etn.h"

static py::dict compute(PairETN& self, const PyNeighbors& nb) {
    return compute_results(self, nb);
}

void bind_pair_etn(py::module_& m) {
    auto cls = py::class_<PairETN, Pair>(m, "PairETN", "ETN potential: energy, forces and virial.");
    def_pair_etn_init(cls);

    def_neighbors(cls, "compute", &compute,
                  R"doc(
Compute the energy, per-atom energies, forces and virial.

Returns:
    dict with "energy", "eatom" (n_atoms), "forces" (n_atoms, 3) and "virial" (6)
    as xx, yy, zz, xy, xz, yz.
)doc");

    cls.def("get_core_count", &PairETN::get_core_count)
        .def("get_species_ranks", [](const PairETN& self) { return to_array(self.get_species_ranks()); }, "Return the species rank per l, ndarray int32 (l_count).")
        .def("get_radial_func_counts", [](const PairETN& self) { return to_array(self.get_radial_func_counts()); }, "Return the radial function count per l, ndarray int32 (l_count).")
        .def("get_radial_ranks", [](const PairETN& self) { return to_array(self.get_radial_ranks()); }, "Return the radial rank per l, ndarray int32 (l_count).")
        .def("get_bond_ranks",
             [](const PairETN& self) {
                 py::list ranks;
                 for (const std::vector<int>& bond : self.get_bond_ranks()) ranks.append(to_array(bond));
                 return ranks;
             },
             "Return the ranks per l of bonds 0 .. core_count, a list of ndarray int32.")

        .def("get_radial_basis_coeffs",
             [](const PairETN& self, int l) { return to_array(self.get_radial_basis_coeffs(l), {self.get_radial_ranks()[l], self.get_radial_func_counts()[l], self.get_species_ranks()[l]}); },
             py::arg("l"), "Return B_l, ndarray float64 (radial_rank, radial_func_count, species_rank).")
        .def("get_species_pair_coeffs",
             [](const PairETN& self, int l) { return to_array(self.get_species_pair_coeffs(l), {self.get_species_count(), self.get_species_count(), self.get_species_ranks()[l]}); },
             py::arg("l"), "Return A_l, ndarray float64 (species, species, species_rank).")
        .def("get_core_coeffs", [](const PairETN& self, int i, bool imaginary) { return to_array(self.get_core_coeffs(i, imaginary)); }, py::arg("i"), py::arg("imaginary") = false,
             "Return the real or imaginary part of core i, a flat ndarray float64 ordered l3, n3, l2, n2, l1, n1.")

        .def("set_radial_basis_coeffs",
             [](PairETN& self, int l, DoubleArray arr) {
                 check_length(arr, self.get_radial_basis_coeffs(l).size(), "set_radial_basis_coeffs");
                 self.set_radial_basis_coeffs(l, arr.data());
             },
             py::arg("l"), py::arg("coeffs"))
        .def("set_species_pair_coeffs",
             [](PairETN& self, int l, DoubleArray arr) {
                 check_length(arr, self.get_species_pair_coeffs(l).size(), "set_species_pair_coeffs");
                 self.set_species_pair_coeffs(l, arr.data());
             },
             py::arg("l"), py::arg("coeffs"))
        .def("set_core_coeffs",
             [](PairETN& self, int i, DoubleArray arr, bool imaginary) {
                 check_length(arr, self.get_core_coeffs(i, imaginary).size(), "set_core_coeffs");
                 self.set_core_coeffs(i, imaginary, arr.data());
             },
             py::arg("i"), py::arg("coeffs"), py::arg("imaginary") = false);
}
