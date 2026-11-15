/* -*- c++ -*- ----------------------------------------------------------
   Python bindings for PairETN (see etn_potential.h).
------------------------------------------------------------------------- */

#include "bindings_common.h"
#include "etn_potential.h"

// --- the computation of mlip-4's ETN ---------------------------------------

// potential_results, plus "inputs" (inum, input size) and "bonds" and "bonds_im",
// one (inum, bond size) array per bond.
static py::dict compute(PairETN& self, const PyNeighbors& nb) {
    {
        py::gil_scoped_release unlocked;
        self.compute(nb.view());
    }

    py::dict result = potential_results(self, nb.n_atoms());
    const int inum = nb.inum();
    result["inputs"] = DoubleArray({inum, self.get_input_shape().size}, self.get_inputs());

    py::list bonds, bonds_im;
    for (int b = 0; b <= self.get_core_count(); b++) {
        const int size = self.get_bond_shape(b).size;
        DoubleArray re({inum, size}), im({inum, size});
        for (int ii = 0; ii < inum; ii++) {
            const double* bond = self.get_bonds() + (size_t) ii * self.get_bonds_size() + self.get_bond_offset(b);
            std::copy(bond, bond + size, re.mutable_data() + (size_t) ii * size);
            std::copy(bond + size, bond + 2 * size, im.mutable_data() + (size_t) ii * size);
        }
        bonds.append(re);
        bonds_im.append(im);
    }
    result["bonds"] = bonds;
    result["bonds_im"] = bonds_im;
    return result;
}

static DoubleArray coeffs_array(const std::vector<double>& block) {
    return DoubleArray({(py::ssize_t) block.size()}, block.data());
}

// Raises unless arr has the length of block.
static void check_length(const DoubleArray& arr, const std::vector<double>& block, const char* what) {
    if ((size_t) arr.size() != block.size())
        throw std::runtime_error(std::string(what) + ": expected length " + std::to_string(block.size()));
}

void bind_etn_potential(py::module_& m) {
    auto cls = py::class_<PairETN>(m, "PairETN", "Energy, forces and virial for an ETN potential file.")
                   .def(py::init<const std::string&>(), py::arg("filename"), "Load an ETN potential from an .etn file (version 1.0.0).");

    // ---- the computation of mlip-4's ETN ----
    def_neighbors(cls, "compute", &compute,
                  R"doc(
Compute energy, per-atom energies, forces and virial, and each central atom's tensors.

Returns a dict with "energy", "eatom" (n_atoms), "forces" (n_atoms, 3), "virials"
(6) as xx,yy,zz,xy,xz,yz, "inputs" (inum, input size), and "bonds" and "bonds_im",
lists of (inum, bond size) arrays for bonds 0 .. core_count. A covariant vector holds
entry (l, m, n) at offset(l) + (m + l) * rank(l) + n.
)doc");

    cls.def("eval_radial_basis",
            [](PairETN& self, double dist) {
                const int sz = self.get_radial_basis_size();
                auto vals = zeros({sz});
                auto ders = zeros({sz});
                self.eval_radial_basis(dist, vals.mutable_data(), ders.mutable_data());
                return py::make_tuple(vals, ders);
            },
            py::arg("dist"), "Return (vals, ders) of the radial basis at distance dist.");

    // ---- no counterpart in mlip-4 ----
    // Accessors for the Python boundary: shape, hyperparameters and coefficient blocks.
    cls.def("get_coeff_count", &PairETN::coeff_count, "Length of mlip-4's params: every coefficient block")
        .def("get_species_count", &PairETN::get_species_count)
        .def("get_core_count", &PairETN::get_core_count)
        .def("get_species_ranks", &PairETN::get_species_ranks)
        .def("get_radial_funcs_counts", &PairETN::get_radial_func_counts)
        .def("get_radial_ranks", &PairETN::get_radial_ranks)
        .def("get_bond_ranks", &PairETN::get_bond_ranks)
        .def("get_radial_basis_size", &PairETN::get_radial_basis_size)
        .def("get_min_cutoff", &PairETN::get_min_cutoff)
        .def("get_max_cutoff", &PairETN::get_max_cutoff)
        .def("get_scaling", &PairETN::get_scaling)
        .def("set_scaling", &PairETN::set_scaling, py::arg("scaling"), "Set the global scaling parameter.")
        .def("is_trained", &PairETN::is_trained, "False for a file without coefficients, which holds mlip-4's initial values.")
        .def("set_trained", &PairETN::set_trained, py::arg("trained"))
        .def("get_potential_name", &PairETN::get_potential_name)
        .def("get_radial_basis_type", &PairETN::get_radial_basis_type)

        .def("get_radial_coeffs", [](const PairETN& self, int l) { return coeffs_array(self.get_radial_coeffs(l)); }, py::arg("l"),
             "ndarray float64 (species_rank * radial_func_count * radial_rank)  — B_l, column-major (k, n', n)")
        .def("get_species_pair_coeffs", [](const PairETN& self, int l) { return coeffs_array(self.get_species_pair_coeffs(l)); }, py::arg("l"),
             "ndarray float64 (species^2 * species_rank)  — A_l, row-major (s_i, s_j, k)")
        .def("get_core_coeffs", [](const PairETN& self, int i, bool imaginary) { return coeffs_array(self.get_core_coeffs(i, imaginary)); }, py::arg("i"), py::arg("imaginary") = false,
             "ndarray float64  — core i's real or imaginary part, ordered l3, n3, l2, n2, l1, n1")
        .def("get_species_coeffs", [](const PairETN& self) { return coeffs_array(self.get_species_coeffs()); },
             "ndarray float64 (species_count)  — per-species reference energies")

        .def("set_radial_coeffs",
             [](PairETN& self, int l, DoubleArray arr) {
                 check_length(arr, self.get_radial_coeffs(l), "set_radial_coeffs");
                 self.set_radial_coeffs(l, arr.data());
             },
             py::arg("l"), py::arg("coeffs"))
        .def("set_species_pair_coeffs",
             [](PairETN& self, int l, DoubleArray arr) {
                 check_length(arr, self.get_species_pair_coeffs(l), "set_species_pair_coeffs");
                 self.set_species_pair_coeffs(l, arr.data());
             },
             py::arg("l"), py::arg("coeffs"))
        .def("set_core_coeffs",
             [](PairETN& self, int i, DoubleArray arr, bool imaginary) {
                 check_length(arr, self.get_core_coeffs(i, imaginary), "set_core_coeffs");
                 self.set_core_coeffs(i, imaginary, arr.data());
             },
             py::arg("i"), py::arg("coeffs"), py::arg("imaginary") = false)
        .def("set_species_coeffs",
             [](PairETN& self, DoubleArray arr) {
                 check_length(arr, self.get_species_coeffs(), "set_species_coeffs");
                 self.set_species_coeffs(arr.data());
             },
             py::arg("coeffs"), "Set per-species reference energies. Shape: (species_count,)");
}
