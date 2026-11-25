/* -*- c++ -*- ----------------------------------------------------------
   The _core_ext module: the neighbor list, Pair and the error table, then
   Equations and MaxVol.
------------------------------------------------------------------------- */

#include "bindings_common.h"
#include "error_table.h"

static py::object eval_grad(Pair& self, const PyNeighbors& nb, bool forces, bool virial) {
    return grad_arrays(nb, self.coeff_count(), forces, virial, [&](double* eg, double* fg, double* vg) { self.eval_grad(nb.view(), eg, fg, vg); });
}

static DoubleArray eval_loss_grad(Pair& self, const PyNeighbors& nb, double dloss_denergy, py::object dloss_dforces, py::object dloss_dvirial) {
    DoubleArray dF, dW;
    const double* dF_ptr = nullptr;
    const double* dW_ptr = nullptr;
    if (!dloss_dforces.is_none()) {
        dF = dloss_dforces.cast<DoubleArray>();
        if (dF.size() != (py::ssize_t) nb.n_atoms() * 3)
            throw std::runtime_error("eval_loss_grad: dloss_dforces must have shape (n_atoms, 3)");
        dF_ptr = dF.data();
    }
    if (!dloss_dvirial.is_none()) {
        dW = dloss_dvirial.cast<DoubleArray>();
        if (dW.size() != 6)
            throw std::runtime_error("eval_loss_grad: dloss_dvirial must have 6 components");
        dW_ptr = dW.data();
    }

    auto grad = zeros({self.coeff_count()});
    double* grad_ptr = grad.mutable_data();
    {
        py::gil_scoped_release unlocked;
        self.eval_loss_grad(nb.view(), dloss_denergy, dF_ptr, dW_ptr, grad_ptr);
    }
    return grad;
}

void bind_core(py::module_& m) {
    py::class_<PyNeighbors>(m, "NeighList", R"doc(
Neighbor list of one structure.

Args:
    types: int32 (n_atoms), the 0-based species of each atom.
    ilist: int32 (inum), the indices of the central atoms.
    numneigh: int32 (inum), the neighbor count of each central atom.
    firstneigh: int32 (sum(numneigh)), the flat neighbor indices.
    displacements: float64 (sum(numneigh), 3), r_j - r_i corrected for periodic images.
)doc")
        .def(py::init<IntArray, IntArray, IntArray, IntArray, DoubleArray>(),
             py::arg("types"), py::arg("ilist"), py::arg("numneigh"), py::arg("firstneigh"), py::arg("displacements"))
        .def_property_readonly("n_atoms", &PyNeighbors::n_atoms)
        .def_property_readonly("inum", &PyNeighbors::inum)
        .def_property_readonly("types", &PyNeighbors::types)
        .def_property_readonly("ilist", &PyNeighbors::ilist)
        .def_property_readonly("numneigh", &PyNeighbors::numneigh)
        .def_property_readonly("firstneigh", &PyNeighbors::firstneigh)
        .def_property_readonly("displacements", &PyNeighbors::displacements);

    py::class_<FitStructure>(m, "FitStructure", "A training structure: its neighbor list at the cutoff of the potential, and its reference values.")
        .def(py::init([](const PyNeighbors& nb, py::object energy, py::object forces, py::object virial, int number) {
                 const NeighList& list = nb.view();
                 const int n_pairs = std::accumulate(list.numneigh, list.numneigh + list.inum, 0);

                 FitStructure s;
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
                         throw std::runtime_error("FitStructure: forces must have shape (n_atoms, 3)");
                     s.forces.assign(f.data(), f.data() + f.size());
                     s.has_forces = true;
                 }
                 if (!virial.is_none()) {
                     DoubleArray v = virial.cast<DoubleArray>();
                     if (v.size() != 6)
                         throw std::runtime_error("FitStructure: virial must have 6 components");
                     std::copy(v.data(), v.data() + 6, s.virial);
                     s.has_virial = true;
                 }
                 s.number = number;
                 return s;
             }), py::arg("neighbors"), py::arg("energy") = py::none(), py::arg("forces") = py::none(), py::arg("virial") = py::none(), py::arg("number") = 0, R"doc(
Args:
    neighbors: the NeighList at the cutoff of the potential.
    energy: float or None.
    forces: (n_atoms, 3) or None.
    virial: (6) or None, in eV, ordered xx, yy, zz, xy, xz, yz.
    number: the number of the structure in the error tables.
)doc")
        .def_property_readonly("neighbors",
                               [](const FitStructure& s) {
                                   return PyNeighbors(to_array(s.types), to_array(s.ilist), to_array(s.numneigh), to_array(s.firstneigh), to_array(s.displacements, {(py::ssize_t) s.firstneigh.size(), 3}));
                               })
        .def_property_readonly("energy", [](const FitStructure& s) { return s.has_energy ? py::object(py::float_(s.energy)) : py::none(); })
        .def_property_readonly("forces", [](const FitStructure& s) { return s.has_forces ? py::object(to_array(s.forces, {s.n_atoms(), 3})) : py::none(); })
        .def_property_readonly("virial", [](const FitStructure& s) { return s.has_virial ? py::object(DoubleArray({6}, s.virial)) : py::none(); })
        .def_readonly("number", &FitStructure::number);

    auto cls = py::class_<Pair>(m, "Pair", "The base of PairMTP and PairETN: the methods of a potential of either kind.");
    cls.def("get_coeff_count", &Pair::coeff_count, "Return the length of the coefficient vector of get_coeffs.")
        .def("get_species_count", &Pair::get_species_count)
        .def("get_radial_basis_size", &Pair::get_radial_basis_size)
        .def("get_min_cutoff", &Pair::get_min_cutoff)
        .def("get_max_cutoff", &Pair::get_max_cutoff)
        .def("get_scaling", &Pair::get_scaling)
        .def("set_scaling", &Pair::set_scaling, py::arg("scaling"), "Set the global scaling parameter.")
        .def("set_min_cutoff", &Pair::set_min_cutoff, py::arg("d"), "Set the minimum distance of the radial basis and rebuild it.")
        .def("is_trained", &Pair::is_trained, "Return False for a potential without fitted coefficients, which holds initial values.")
        .def("set_trained", &Pair::set_trained, py::arg("trained"))
        .def("get_potential_name", &Pair::get_potential_name)
        .def("get_radial_basis_type", &Pair::get_radial_basis_type)
        .def("get_species", &Pair::get_species, "Return the element of each species index.")
        .def("set_species", &Pair::set_species, py::arg("species"))
        .def("get_species_coeffs",
             [](const Pair& self) { return to_array(self.get_species_coeffs()); },
             "Return the per-species reference energies, float64 (species_count).")
        .def("set_species_coeffs",
             [](Pair& self, DoubleArray arr) {
                 check_length(arr, self.get_species_count(), "set_species_coeffs");
                 self.set_species_coeffs(arr.data());
             },
             py::arg("coeffs"), "Set the per-species reference energies, of shape (species_count,).")
        .def("get_coeffs",
             [](const Pair& self) {
                 auto out = zeros({self.coeff_count()});
                 self.get_coeffs(out.mutable_data());
                 return out;
             },
             "Return every coefficient as one vector, in the order of the potential.")
        .def("set_coeffs",
             [](Pair& self, DoubleArray arr) {
                 check_length(arr, self.coeff_count(), "set_coeffs");
                 self.set_coeffs(arr.data());
             },
             py::arg("coeffs"))
        .def("eval_radial_basis",
             [](Pair& self, double dist) {
                 const int sz = self.get_radial_basis_size();
                 auto vals = zeros({sz});
                 auto ders = zeros({sz});
                 self.eval_radial_basis(dist, vals.mutable_data(), ders.mutable_data());
                 return py::make_tuple(vals, ders);
             },
             py::arg("dist"), "Return (vals, ders) of the radial basis at distance dist.");

    def_neighbors(cls, "eval_grad", &eval_grad,
                  R"doc(
Return d(E_i, F, virial)/dc, with c ordered as in get_coeffs. MTP and ETN only.

Args:
    forces: also return the force gradient.
    virial: also return the virial gradient; requires forces.

Returns:
    Without forces, site_energy_grad (inum, cc). Row ii is dE_i/dc of atom ilist[ii],
    the vector that extrapolation grades use.
    With forces, (site_energy_grad (inum, cc), force_grad (n_atoms, 3, cc), virial_grad (6, cc) or None).
)doc",
                  py::arg("forces") = false, py::arg("virial") = false);

    def_neighbors(cls, "eval_loss_grad", &eval_loss_grad,
                  R"doc(
Return dL/dc (cc,) for a loss L with the given derivatives. MTP and ETN only.

Args:
    dloss_denergy: dL/dE.
    dloss_dforces: dL/dF (n_atoms, 3), or None for zero.
    dloss_dvirial: dL/dvirial (6), ordered xx, yy, zz, xy, xz, yz, or None for zero.
)doc",
                  py::arg("dloss_denergy"), py::arg("dloss_dforces") = py::none(), py::arg("dloss_dvirial") = py::none());

    m.def("error_table", [](Pair& potential, py::iterable structures, const std::vector<std::string>& species, const std::string& title) {
        std::vector<FitStructure> share = std::move(split_structures(structures, 1)[0]);
        ProcessRanks process("error_table", 1, py::none());
        py::gil_scoped_release unlocked;
        return error_table(potential, share, species, process[0], title);
    }, py::arg("potential"), py::arg("structures"), py::arg("species"), py::arg("title"), R"doc(
Return the error table of potential on the structures under title, in the format of the fit_mtp and fit_etn logs.

Args:
    structures: iterable of FitStructure.
    species: the names of the species.
)doc");
}

PYBIND11_MODULE(_core_ext, m) {
    m.doc() = "The classes the MTP and ETN bindings share: NeighList, FitStructure, Pair, Equations and MaxVol, and error_table.";

    bind_core(m);
    bind_maxvol(m);
}
