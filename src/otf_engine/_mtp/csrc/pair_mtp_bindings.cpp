/* -*- c++ -*- ----------------------------------------------------------
   Python bindings for PairMTP (see pair_mtp.h).
------------------------------------------------------------------------- */

#include "pair_mtp_bindings.h"
#include "pair_mtp.h"

static py::dict compute(PairMTP& self, const PyNeighbors& nb) {
    return compute_results(self, nb);
}

void bind_pair_mtp(py::module_& m) {
    m.def("level_basis_functions", &level_basis_functions, py::arg("level"), "Return the basis functions of an MTP level, 6 to 28, each a list of n radial indices followed by an n x n contraction matrix.");

    auto cls = py::class_<PairMTP, Pair>(m, "PairMTP", "MTP potential: energy, forces and virial.");
    def_pair_mtp_init(cls);

    def_neighbors(cls, "compute", &compute,
                  R"doc(
Compute the energy, per-atom energies, forces and virial.

Returns:
    dict with "energy", "eatom" (n_atoms), "forces" (n_atoms, 3) and
    "virial" (6) as xx, yy, zz, xy, xz, yz.
)doc");

    cls.def("get_radial_func_count", &PairMTP::get_radial_func_count)
        .def("get_radial_coeff_count", &PairMTP::get_radial_coeff_count, "Return the radial coefficient count, species^2 * radial_func_count * radial_basis_size.")
        .def("get_alpha_scalar_count", &PairMTP::get_alpha_scalar_count)
        .def("get_alpha_moment_count", &PairMTP::get_alpha_moment_count)
        .def("get_alpha_index_basic_count", &PairMTP::get_alpha_index_basic_count)
        .def("get_alpha_index_times_count", &PairMTP::get_alpha_index_times_count)

        .def("get_radial_basis_coeffs",
             [](const PairMTP& self) {
                 return to_array(self.get_radial_basis_coeffs(), {self.get_species_count() * self.get_species_count(), self.get_radial_func_count(), self.get_radial_basis_size()});
             },
             "Return the radial coefficients, an ndarray float64 (species^2, radial_func_count, radial_basis_size).")
        .def("get_alpha_index_basic",
             [](const PairMTP& self) { return py::array_t<int>({self.get_alpha_index_basic_count(), 4}, self.get_alpha_index_basic()[0].data()); },
             "Return the basic moments, an ndarray int32 (alpha_index_basic_count, 4) with columns mu, px, py, pz.")
        .def("get_alpha_index_times",
             [](const PairMTP& self) { return py::array_t<int>({self.get_alpha_index_times_count(), 4}, self.get_alpha_index_times()[0].data()); },
             "Return the moment products, an ndarray int32 (alpha_index_times_count, 4) with columns i0, i1, multiplier, out.")
        .def("get_alpha_moment_mapping",
             [](const PairMTP& self) { return to_array(self.get_alpha_moment_mapping()); },
             "Return the moment of each basis value, an ndarray int32 (alpha_scalar_count).")
        .def("get_linear_coeffs",
             [](const PairMTP& self) { return to_array(self.get_linear_coeffs()); },
             "Return the moment tensor basis coefficients, an ndarray float64 (alpha_scalar_count).")

        .def("set_linear_coeffs",
             [](PairMTP& self, DoubleArray arr) {
                 check_length(arr, self.get_alpha_scalar_count(), "set_linear_coeffs");
                 self.set_linear_coeffs(arr.data());
             },
             py::arg("coeffs"), "Set the linear (moment tensor) coefficients, shape (alpha_scalar_count,).")
        .def("set_radial_basis_coeffs",
             [](PairMTP& self, DoubleArray arr) {
                 check_length(arr, self.get_radial_coeff_count(), "set_radial_basis_coeffs");
                 self.set_radial_basis_coeffs(arr.data());
             },
             py::arg("coeffs"), "Set the radial basis coefficients, shape (species^2, radial_func_count, radial_basis_size) or flat.");
}
