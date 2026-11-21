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
Neighbor list for one structure.

types         : ndarray int32   (n_atoms)         — 0-indexed species
ilist         : ndarray int32   (inum)            — indices of central atoms
numneigh      : ndarray int32   (inum)            — neighbor count per central atom
firstneigh    : ndarray int32   (sum(numneigh))   — flat neighbor indices
displacements : ndarray float64 (sum(numneigh),3) — PBC-corrected r_j - r_i
)doc")
        .def(py::init<IntArray, IntArray, IntArray, IntArray, DoubleArray>(),
             py::arg("types"), py::arg("ilist"), py::arg("numneigh"), py::arg("firstneigh"), py::arg("displacements"))
        .def_property_readonly("n_atoms", &PyNeighbors::n_atoms)
        .def_property_readonly("inum", &PyNeighbors::inum)
        .def_property_readonly("types", &PyNeighbors::types)
        .def_property_readonly("displacements", &PyNeighbors::displacements);

    py::class_<FitStructure>(m, "FitStructure", "A structure to fit: its neighbor list at the potential's cutoff and its reference values, which the fits and error_table take.")
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
neighbors : NeighList at the potential's cutoff
energy    : float | None
forces    : (n_atoms, 3) | None
virial    : (6) | None, mlip-3's PlusStress in eV, xx,yy,zz,xy,xz,yz
number    : the number the error tables name it by
)doc")
        .def_property_readonly("neighbors",
                               [](const FitStructure& s) {
                                   return PyNeighbors(to_array(s.types), to_array(s.ilist), to_array(s.numneigh), to_array(s.firstneigh), to_array(s.displacements, {(py::ssize_t) s.firstneigh.size(), 3}));
                               })
        .def_property_readonly("energy", [](const FitStructure& s) { return s.has_energy ? py::object(py::float_(s.energy)) : py::none(); })
        .def_property_readonly("forces", [](const FitStructure& s) { return s.has_forces ? py::object(to_array(s.forces, {s.n_atoms(), 3})) : py::none(); })
        .def_property_readonly("virial", [](const FitStructure& s) { return s.has_virial ? py::object(DoubleArray({6}, s.virial)) : py::none(); })
        .def_readonly("number", &FitStructure::number);

    auto cls = py::class_<Pair>(m, "Pair", "The base of PairMTP and PairETN: what a potential gives whatever its kind.");
    cls.def("get_coeff_count", &Pair::coeff_count, "Length of the coefficient vector get_coeffs gives.")
        .def("get_species_count", &Pair::get_species_count)
        .def("get_radial_basis_size", &Pair::get_radial_basis_size)
        .def("get_min_cutoff", &Pair::get_min_cutoff)
        .def("get_max_cutoff", &Pair::get_max_cutoff)
        .def("get_scaling", &Pair::get_scaling)
        .def("set_scaling", &Pair::set_scaling, py::arg("scaling"), "Set the global scaling parameter.")
        .def("set_min_cutoff", &Pair::set_min_cutoff, py::arg("d"), "Set the radial basis's minimum distance.")
        .def("is_trained", &Pair::is_trained, "False for a file without coefficients, which holds the reference code's initial values.")
        .def("set_trained", &Pair::set_trained, py::arg("trained"))
        .def("get_potential_name", &Pair::get_potential_name)
        .def("get_radial_basis_type", &Pair::get_radial_basis_type)
        .def("get_species_coeffs",
             [](const Pair& self) { return to_array(self.get_species_coeffs()); },
             "ndarray float64 (species_count)  — per-species reference energies")
        .def("set_species_coeffs",
             [](Pair& self, DoubleArray arr) {
                 check_length(arr, self.get_species_count(), "set_species_coeffs");
                 self.set_species_coeffs(arr.data());
             },
             py::arg("coeffs"), "Set per-species reference energies. Shape: (species_count,)")
        .def("get_coeffs",
             [](const Pair& self) {
                 auto out = zeros({self.coeff_count()});
                 self.get_coeffs(out.mutable_data());
                 return out;
             },
             "Every coefficient as one vector, in the order of the potential's kind.")
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
d(E_i, F, virial)/dc, c as get_coeffs orders it; MTP and ETN only.

Without forces, returns site_energy_grad (inum, cc); row ii is dE_i/dc for
atom ilist[ii], the information vector extrapolation grades are taken from.
With forces, returns (site_energy_grad (inum, cc), force_grad (n_atoms, 3, cc), virial_grad (6, cc)|None).
)doc",
                  py::arg("forces") = false, py::arg("virial") = false);

    def_neighbors(cls, "eval_loss_grad", &eval_loss_grad,
                  R"doc(
dL/dc (cc,) for a loss L with the given derivatives w.r.t. the energy,
the forces (n_atoms, 3) and the virial (6, xx,yy,zz,xy,xz,yz); None is zero;
MTP and ETN only.
)doc",
                  py::arg("dloss_denergy"), py::arg("dloss_dforces") = py::none(), py::arg("dloss_dvirial") = py::none());

    m.def("error_table", [](Pair& potential, py::iterable structures, const std::vector<std::string>& species, const std::string& title) {
        std::vector<FitStructure> share = std::move(split_structures(structures, 1)[0]);
        ProcessRanks process("error_table", 1, py::none());
        py::gil_scoped_release unlocked;
        return error_table(potential, share, species, process[0], title);
    }, py::arg("potential"), py::arg("structures"), py::arg("species"), py::arg("title"), R"doc(
The errors of potential on the structures, as the table fit_mtp and fit_etn log, under title.

structures : iterable of FitStructure
species    : the species' names
)doc");
}

PYBIND11_MODULE(_core_ext, m) {
    m.doc() = "What the MTP and ETN bindings share: NeighList, Pair, the error table, Equations and MaxVol.";

    bind_core(m);
    bind_maxvol(m);
}
