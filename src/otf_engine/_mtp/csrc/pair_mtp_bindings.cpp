/* -*- c++ -*- ----------------------------------------------------------
   Python bindings for PairMTP (see pair_mtp.h).
------------------------------------------------------------------------- */

#include "pair_mtp_bindings.h"
#include "pair_mtp.h"

// --- ported from PairMTP --------------------------------------------------

static py::dict compute(PairMTP& self, const PyNeighbors& nb) {
    return compute_results(self, nb);
}

void bind_pair_mtp(py::module_& m) {
    m.def("level_basis_functions", &level_basis_functions, py::arg("level"), "The basis functions of an MTP level, 6 to 28, from mlip-4's mtp_levels.h: lists of n radial indices then an n x n contraction matrix.");

    auto cls = py::class_<PairMTP, Pair>(m, "PairMTP", "Energy, forces and virial for an MTP potential.");
    def_pair_mtp_init(cls);

    // ---- ported from PairMTP ----
    def_neighbors(cls, "compute", &compute,
                  R"doc(
Compute energy, per-atom energies, forces and virial.

Returns a dict with "energy", "eatom" (n_atoms), "forces" (n_atoms, 3) and
"virials" (6) as xx,yy,zz,xy,xz,yz.
)doc");

    // ---- no counterpart in the reference ----
    // PairMTP exposes no parameter accessors: LAMMPS owns the object and never
    // reads it back. Everything below and in Pair exists to serve the Python boundary.
    cls.def("get_radial_func_count", &PairMTP::get_radial_func_count)
        .def("get_radial_coeff_count", &PairMTP::get_radial_coeff_count, "species^2 * radial_func_count * radial_basis_size")
        .def("get_alpha_scalar_count", &PairMTP::get_alpha_scalar_count)
        .def("get_alpha_moment_count", &PairMTP::get_alpha_moment_count)
        .def("get_alpha_index_basic_count", &PairMTP::get_alpha_index_basic_count)
        .def("get_alpha_index_times_count", &PairMTP::get_alpha_index_times_count)
        .def("set_min_cutoff", &PairMTP::set_min_cutoff, py::arg("d"),
             "Set the radial basis's minimum distance.")

        .def("get_radial_basis_coeffs",
             [](const PairMTP& self) {
                 return to_array(self.get_radial_basis_coeffs(), {self.get_species_count() * self.get_species_count(), self.get_radial_func_count(), self.get_radial_basis_size()});
             },
             "ndarray float64 (species^2, radial_func_count, radial_basis_size)")
        .def("get_alpha_index_basic",
             [](const PairMTP& self) { return py::array_t<int>({self.get_alpha_index_basic_count(), 4}, self.get_alpha_index_basic()[0].data()); },
             "ndarray int32 (alpha_index_basic_count, 4)  columns: mu, px, py, pz")
        .def("get_alpha_index_times",
             [](const PairMTP& self) { return py::array_t<int>({self.get_alpha_index_times_count(), 4}, self.get_alpha_index_times()[0].data()); },
             "ndarray int32 (alpha_index_times_count, 4)  columns: i0, i1, multiplier, out")
        .def("get_alpha_moment_mapping",
             [](const PairMTP& self) { return to_array(self.get_alpha_moment_mapping()); },
             "ndarray int32 (alpha_scalar_count)")
        .def("get_linear_coeffs",
             [](const PairMTP& self) { return to_array(self.get_linear_coeffs()); },
             "ndarray float64 (alpha_scalar_count)  — moment tensor basis coefficients")

        .def("set_linear_coeffs",
             [](PairMTP& self, DoubleArray arr) {
                 if (arr.size() != self.get_alpha_scalar_count())
                     throw std::runtime_error("set_linear_coeffs: expected length " + std::to_string(self.get_alpha_scalar_count()));
                 self.set_linear_coeffs(arr.data());
             },
             py::arg("coeffs"), "Set linear (moment tensor) coefficients. Shape: (alpha_scalar_count,)")
        .def("set_radial_basis_coeffs",
             [](PairMTP& self, DoubleArray arr) {
                 if (arr.size() != self.get_radial_coeff_count())
                     throw std::runtime_error("set_radial_basis_coeffs: expected length " + std::to_string(self.get_radial_coeff_count()));
                 self.set_radial_basis_coeffs(arr.data());
             },
             py::arg("coeffs"), "Set radial basis coefficients. Shape: (species^2, radial_func_count, radial_basis_size) or flat.");
}
