/* -*- c++ -*- ----------------------------------------------------------
   The _core_ext module: the neighbor list, Pair and the error table, then
   Equations and MaxVol. Built with MTP_MPI, the _core_mpi module: the
   MaxVol search over MPI.
------------------------------------------------------------------------- */

#include "bindings_common.h"

#ifdef MTP_MPI

PYBIND11_MODULE(_core_mpi, m) {
    m.doc() = "The MaxVol search over MPI.";
    py::module_::import("otf_engine._core._core_ext");    // registers Equations and MaxVol
    bind_maxvol(m);
}

#else

#include "error_table.h"

static DoubleArray coeffs(const Pair& self) {
    auto out = zeros({self.coeff_count()});
    self.get_coeffs(out.mutable_data());
    return out;
}

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

    auto cls = py::class_<Pair>(m, "Pair", "The base of PairMTP and PairETN: what a potential gives whatever its kind.");
    cls.def("get_coeff_count", &Pair::coeff_count, "Length of the coefficient vector get_coeffs gives.")
        .def("get_species_count", &Pair::get_species_count)
        .def("get_radial_basis_size", &Pair::get_radial_basis_size)
        .def("get_min_cutoff", &Pair::get_min_cutoff)
        .def("get_max_cutoff", &Pair::get_max_cutoff)
        .def("get_scaling", &Pair::get_scaling)
        .def("set_scaling", &Pair::set_scaling, py::arg("scaling"), "Set the global scaling parameter.")
        .def("is_trained", &Pair::is_trained, "False for a file without coefficients, which holds the reference code's initial values.")
        .def("set_trained", &Pair::set_trained, py::arg("trained"))
        .def("get_potential_name", &Pair::get_potential_name)
        .def("get_radial_basis_type", &Pair::get_radial_basis_type)
        .def("get_species_coeffs",
             [](const Pair& self) { return to_array(self.get_species_coeffs()); },
             "ndarray float64 (species_count)  — per-species reference energies")
        .def("set_species_coeffs",
             [](Pair& self, DoubleArray arr) {
                 if (arr.size() != self.get_species_count())
                     throw std::runtime_error("set_species_coeffs: expected length " + std::to_string(self.get_species_count()));
                 self.set_species_coeffs(arr.data());
             },
             py::arg("coeffs"), "Set per-species reference energies. Shape: (species_count,)")
        .def("get_coeffs", &coeffs, "Every coefficient as one vector, in the order of the potential's kind.")
        .def("set_coeffs",
             [](Pair& self, DoubleArray arr) {
                 if (arr.size() != self.coeff_count())
                     throw std::runtime_error("set_coeffs: expected length " + std::to_string(self.coeff_count()));
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
d(E_i, F, virial)/dc, c as get_coeffs orders it; a training class only.

Without forces, returns site_energy_grad (inum, cc); row ii is dE_i/dc for
atom ilist[ii], the information vector extrapolation grades are taken from.
With forces, returns (site_energy_grad (inum, cc), force_grad (n_atoms, 3, cc), virial_grad (6, cc)|None).
)doc",
                  py::arg("forces") = false, py::arg("virial") = false);

    def_neighbors(cls, "eval_loss_grad", &eval_loss_grad,
                  R"doc(
dL/dc (cc,) for a loss L with the given derivatives w.r.t. the energy,
the forces (n_atoms, 3) and the virial (6, xx,yy,zz,xy,xz,yz); None is zero;
a training class only.
)doc",
                  py::arg("dloss_denergy"), py::arg("dloss_dforces") = py::none(), py::arg("dloss_dvirial") = py::none());

    m.def("error_table", [](Pair& potential, py::iterable structures, const std::vector<std::string>& species, const std::string& title) {
        std::vector<TrainingStructure> share;
        for (py::handle item : structures)
            share.push_back(make_structure(item));
        ThreadGroup group(1);
        ThreadComm comm{&group, 0};
        py::gil_scoped_release unlocked;
        return error_table(potential, share, species, &comm, title);
    }, py::arg("potential"), py::arg("structures"), py::arg("species"), py::arg("title"), R"doc(
The errors of potential on the structures, as the table train_mtp and train_etn log, under title.

structures : iterable of (NeighList, energy|None, forces(n_atoms,3)|None, virial(6)|None, number),
             neighbor lists at the potential's cutoff
species    : the species' names
)doc");
}

PYBIND11_MODULE(_core_ext, m) {
    m.doc() = "What the MTP and ETN bindings share: NeighList, Pair, the error table, Equations and MaxVol.";

    bind_core(m);
    bind_maxvol(m);
}

#endif
