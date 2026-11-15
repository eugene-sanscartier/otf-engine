/* -*- c++ -*- ----------------------------------------------------------
   The _etn_ext module: ETN, the radial basis, the real harmonics and their
   3-j symbols. The NeighList type is _mtp_ext's, which must be imported first.
------------------------------------------------------------------------- */

#include "bindings_common.h"
#include "etn_potential.h"

// compute's results: "energy", "forces" (n_atoms, 3), "virials" (6), "eatom" (n_atoms),
// "inputs" (inum, input size), and "bonds" and "bonds_im", one (inum, bond size) array per bond.
static py::dict compute(ETN& self, const PyNeighbors& nb) {
    {
        py::gil_scoped_release unlocked;
        self.compute(nb.view());
    }

    const int n_atoms = nb.n_atoms(), inum = nb.inum();
    py::dict result;
    result["energy"] = self.get_energy();
    result["forces"] = DoubleArray({n_atoms, 3}, self.get_forces());
    result["virials"] = DoubleArray({6}, self.get_virial());
    result["eatom"] = DoubleArray({n_atoms}, self.get_eatom());
    result["inputs"] = DoubleArray({inum, self.get_input_shape().size}, self.get_inputs());

    py::list bonds, bonds_im;
    const int stride = self.get_bonds_size();
    for (int b = 0; b <= self.get_core_count(); b++) {
        const int size = self.bond_shape(b).size;
        DoubleArray re({inum, size}), im({inum, size});
        for (int ii = 0; ii < inum; ii++) {
            const double* bond = self.get_bonds() + (size_t) ii * stride + self.bond_offset(b);
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

static DoubleArray block(const ETN& self, int offset, int count) {
    return DoubleArray({count}, self.get_coeffs() + offset);
}

PYBIND11_MODULE(_etn_ext, m) {
    m.doc() = "Equivariant tensor network potential (mlip-4's ETN, without its program).";

    auto cls = py::class_<ETN>(m, "ETN", "Energy, forces and virial for an equivariant tensor network potential.")
                   .def(py::init<const std::string&>(), py::arg("filename"), "Load an ETN potential from an .etn file.")
                   .def(py::init<int, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<std::vector<int>>, int, double, double>(),
                        py::arg("species_count"), py::arg("species_ranks"), py::arg("radial_funcs_counts"), py::arg("radial_ranks"), py::arg("bond_ranks"),
                        py::arg("radial_basis_size"), py::arg("min_dist"), py::arg("max_dist"),
                        "An ETN of the given shape with every coefficient zero. The three per-l lists share one length; bond_ranks holds one per-l list per bond, {1} at both ends.");

    def_neighbors(cls, "compute", &compute,
                  R"doc(
Compute energy, per-atom energies, forces and virial, and each central atom's tensors.

Returns a dict with "energy", "eatom" (n_atoms), "forces" (n_atoms, 3), "virials"
(6) as xx,yy,zz,xy,xz,yz, "inputs" (inum, input size), and "bonds" and "bonds_im",
lists of (inum, bond size) arrays for bonds 0 .. core_count. A covariant vector holds
entry (l, m, n) at offset(l) + (m + l) * rank(l) + n.
)doc");

    cls.def("eval_radial_basis",
            [](const ETN& self, double dist) {
                const RadialBasisCinf& basis = self.get_radial_basis();
                DoubleArray vals({basis.size}), ders({basis.size});
                basis.calc(dist, vals.mutable_data(), ders.mutable_data());
                return py::make_tuple(vals, ders);
            },
            py::arg("dist"), "Radial basis values times scaling, and their derivatives, at dist.")
        .def("get_coeff_count", &ETN::coeff_count)
        .def("get_coeffs", [](const ETN& self) { return DoubleArray({self.coeff_count()}, self.get_coeffs()); }, "Every coefficient, in mlip-4's params order.")
        .def("set_coeffs",
             [](ETN& self, DoubleArray arr) {
                 if (arr.size() != self.coeff_count())
                     throw std::runtime_error("set_coeffs: expected length " + std::to_string(self.coeff_count()));
                 self.set_coeffs(arr.data());
             },
             py::arg("coeffs"))
        .def("get_species_count", &ETN::get_species_count)
        .def("get_core_count", &ETN::get_core_count)
        .def("get_species_ranks", &ETN::get_species_ranks)
        .def("get_radial_funcs_counts", &ETN::get_radial_func_counts)
        .def("get_radial_ranks", &ETN::get_radial_ranks)
        .def("get_bond_ranks", &ETN::get_bond_ranks)
        .def("get_radial_basis_size", [](const ETN& self) { return self.get_radial_basis().size; })
        .def("get_min_cutoff", [](const ETN& self) { return self.get_radial_basis().min_dist; })
        .def("get_max_cutoff", [](const ETN& self) { return self.get_radial_basis().max_dist; })
        .def("get_scaling", &ETN::get_scaling)
        .def("set_scaling", &ETN::set_scaling, py::arg("scaling"))
        .def("get_potential_name", &ETN::get_potential_name)

        .def("get_radial_coeffs",
             [](const ETN& self, int l) {
                 return block(self, self.radial_coeffs_offset(l), self.get_species_ranks()[l] * self.get_radial_func_counts()[l] * self.get_radial_ranks()[l]);
             },
             py::arg("l"), "B_l, column-major over (species rank, radial function, radial rank).")
        .def("get_species_pair_coeffs",
             [](const ETN& self, int l) { return block(self, self.species_pair_coeffs_offset(l), self.get_species_count() * self.get_species_count() * self.get_species_ranks()[l]); },
             py::arg("l"), "A_l, row-major over (central species, neighbor species, species rank).")
        .def("get_core_coeffs",
             [](const ETN& self, int i, bool imaginary) {
                 const Core& core = self.core(i, imaginary);
                 return block(self, core.coeff_offset, core.coeff_count);
             },
             py::arg("i"), py::arg("imaginary") = false, "Coefficients of core i, real or imaginary part.")
        .def("get_species_coeffs", [](const ETN& self) { return block(self, self.species_coeffs_offset(), self.get_species_count()); }, "Per-species reference energies.");

    m.def("real_harmonics",
          [](int l_count, DoubleArray u) {
              RealHarmonics harmonics(l_count);
              harmonics.compute(u.data());
              return py::make_tuple(DoubleArray({l_count * l_count}, harmonics.vals.data()), DoubleArray({l_count * l_count, 3}, harmonics.grads.data()));
          },
          py::arg("l_count"), py::arg("u"), "Real harmonics Y_lm at the unit vector u, at l * l + l + m, and their gradients tangent to the sphere.");
    m.def("wigner_3j", &wigner_3j, py::arg("l1"), py::arg("l2"), py::arg("l3"), py::arg("m1"), py::arg("m2"), py::arg("m3"));
    m.def("real_wigner_3j", &real_wigner_3j, py::arg("l1"), py::arg("m1"), py::arg("l2"), py::arg("m2"), py::arg("l3"), py::arg("m3"),
          "3-j symbol of real harmonics: real part for an even l1 + l2 + l3, imaginary part for an odd one.");
}
