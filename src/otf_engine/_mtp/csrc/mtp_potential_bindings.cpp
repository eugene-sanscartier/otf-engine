/* -*- c++ -*- ----------------------------------------------------------
   Python bindings for PairMTP (see mtp_potential.h).
------------------------------------------------------------------------- */

#include "bindings_common.h"

// --- ported from PairMTP --------------------------------------------------

static py::dict compute(PairMTP& self, const PyNeighbors& nb, bool compute_virials, bool compute_eatom) {
    const int n = nb.n_atoms();
    auto forces = zeros({n, 3});
    auto virials = zeros({6});
    auto eatom = zeros({compute_eatom ? n : 0});

    double energy = self.compute(nb.view(), forces.mutable_data(), compute_virials ? virials.mutable_data() : nullptr, compute_eatom ? eatom.mutable_data() : nullptr);

    py::dict result;
    result["energy"] = energy;
    result["forces"] = forces;
    result["virials"] = virials;
    if (compute_eatom)
        result["eatom"] = eatom;
    return result;
}

// --- no counterpart in the reference --------------------------------------

static DoubleArray eval_basis(PairMTP& self, const PyNeighbors& nb) {
    auto out = zeros({nb.inum(), self.get_alpha_scalar_count()});
    self.eval_basis(nb.view(), out.mutable_data());
    return out;
}

void bind_potential(py::module_& m) {
    auto cls = py::class_<PairMTP>(m, "PairMTP", "Energy, forces and virial for an MTP potential file.")
                   .def(py::init<const std::string&>(), py::arg("filename"), "Load an MTP potential from a .mtp file (version 1.1.0).");

    // ---- ported from PairMTP ----
    def_neighbors(cls, "compute", &compute,
                  R"doc(
Compute energy, forces and virial.

Returns a dict with "energy", "forces" (n_atoms, 3), "virials" (6) as
xx,yy,zz,xy,xz,yz, and "eatom" (n_atoms) when compute_eatom is True.
)doc",
                  py::arg("compute_virials") = true, py::arg("compute_eatom") = false);

    // ---- no counterpart in the reference ----
    // PairMTP exposes no parameter accessors: LAMMPS owns the object and never
    // reads it back. Everything below exists to serve the Python boundary.
    def_neighbors(cls, "eval_basis", &eval_basis,
                  "Basis values per central atom: ndarray float64 (inum, alpha_scalar_count).");

    cls.def("eval_radial_basis",
            [](PairMTP& self, double dist) {
                const int sz = self.get_radial_basis_size();
                auto vals = zeros({sz});
                auto ders = zeros({sz});
                self.eval_radial_basis(dist, vals.mutable_data(), ders.mutable_data());
                return py::make_tuple(vals, ders);
            },
            py::arg("dist"), "Return (vals, ders) of the radial basis at distance dist.")

        .def("get_species_count", &PairMTP::get_species_count)
        .def("get_radial_func_count", &PairMTP::get_radial_func_count)
        .def("get_radial_basis_size", &PairMTP::get_radial_basis_size)
        .def("get_radial_coeff_count", &PairMTP::get_radial_coeff_count, "species^2 * radial_func_count * radial_basis_size")
        .def("get_alpha_scalar_count", &PairMTP::get_alpha_scalar_count)
        .def("get_alpha_moment_count", &PairMTP::get_alpha_moment_count)
        .def("get_alpha_index_basic_count", &PairMTP::get_alpha_index_basic_count)
        .def("get_alpha_index_times_count", &PairMTP::get_alpha_index_times_count)
        .def("get_min_cutoff", &PairMTP::get_min_cutoff)
        .def("get_max_cutoff", &PairMTP::get_max_cutoff)
        .def("set_min_cutoff", &PairMTP::set_min_cutoff, py::arg("d"),
             "Update radial basis minimum distance. Mirrors mlip-3 AddSpecies(): min_val = 0.99 * min(training distances).")
        .def("get_scaling", &PairMTP::get_scaling)
        .def("set_scaling", &PairMTP::set_scaling, py::arg("scaling"), "Set the global scaling parameter.")
        .def("get_potential_name", &PairMTP::get_potential_name)

        .def("get_radial_basis_coeffs",
             [](const PairMTP& self) {
                 return DoubleArray({self.get_species_count() * self.get_species_count(), self.get_radial_func_count(), self.get_radial_basis_size()}, self.get_radial_basis_coeffs());
             },
             "ndarray float64 (species^2, radial_func_count, radial_basis_size)")
        .def("get_alpha_index_basic",
             [](const PairMTP& self) { return py::array_t<int>({self.get_alpha_index_basic_count(), 4}, self.get_alpha_index_basic()); },
             "ndarray int32 (alpha_index_basic_count, 4)  columns: mu, px, py, pz")
        .def("get_alpha_index_times",
             [](const PairMTP& self) { return py::array_t<int>({self.get_alpha_index_times_count(), 4}, self.get_alpha_index_times()); },
             "ndarray int32 (alpha_index_times_count, 4)  columns: i0, i1, multiplier, out")
        .def("get_alpha_moment_mapping",
             [](const PairMTP& self) { return py::array_t<int>({self.get_alpha_scalar_count()}, self.get_alpha_moment_mapping()); },
             "ndarray int32 (alpha_scalar_count)")
        .def("get_linear_coeffs",
             [](const PairMTP& self) { return DoubleArray({self.get_alpha_scalar_count()}, self.get_linear_coeffs()); },
             "ndarray float64 (alpha_scalar_count)  — moment tensor basis coefficients")
        .def("get_species_coeffs",
             [](const PairMTP& self) { return DoubleArray({self.get_species_count()}, self.get_species_coeffs()); },
             "ndarray float64 (species_count)  — per-species reference energies")

        .def("set_linear_coeffs",
             [](PairMTP& self, DoubleArray arr) {
                 if (arr.size() != self.get_alpha_scalar_count())
                     throw std::runtime_error("set_linear_coeffs: expected length " + std::to_string(self.get_alpha_scalar_count()));
                 self.set_linear_coeffs(arr.data());
             },
             py::arg("coeffs"), "Set linear (moment tensor) coefficients. Shape: (alpha_scalar_count,)")
        .def("set_species_coeffs",
             [](PairMTP& self, DoubleArray arr) {
                 if (arr.size() != self.get_species_count())
                     throw std::runtime_error("set_species_coeffs: expected length " + std::to_string(self.get_species_count()));
                 self.set_species_coeffs(arr.data());
             },
             py::arg("coeffs"), "Set per-species reference energies. Shape: (species_count,)")
        .def("set_radial_basis_coeffs",
             [](PairMTP& self, DoubleArray arr) {
                 if (arr.size() != self.get_radial_coeff_count())
                     throw std::runtime_error("set_radial_basis_coeffs: expected length " + std::to_string(self.get_radial_coeff_count()));
                 self.set_radial_basis_coeffs(arr.data());
             },
             py::arg("coeffs"), "Set radial basis coefficients. Shape: (species^2, radial_func_count, radial_basis_size) or flat.");
}
