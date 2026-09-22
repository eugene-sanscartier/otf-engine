/* -*- c++ -*- ----------------------------------------------------------
   Python bindings for MTPTraining (see mtp_training.h).

   Nothing here has a counterpart in the reference, which is a pair style and
   differentiates nothing w.r.t. the coefficients.
------------------------------------------------------------------------- */

#include "bindings_common.h"

static py::tuple grad_block(MTPTraining& self, const PyNeighbors& nb, MTPTraining::Cols cols, bool compute_virial_grad) {
    const int width = self.cols_width(cols);
    const int n = nb.n_atoms();
    const bool radial = (cols == MTPTraining::RADIAL);
    // The radial energy gradient is summed over atoms; the others are per-atom.
    auto eg = radial ? zeros({width}) : zeros({nb.inum(), width});
    auto fg = zeros({n, 3, width});
    DoubleArray vg;
    double* vg_ptr = nullptr;
    if (compute_virial_grad) {
        vg = zeros({6, width});
        vg_ptr = vg.mutable_data();
    }

    if (radial) {
        self.eval_grad_radial(nb.view(), eg.mutable_data(), fg.mutable_data(), vg_ptr);
    } else if (cols == MTPTraining::LINEAR) {
        self.eval_grad_linear(nb.view(), eg.mutable_data(), fg.mutable_data(), vg_ptr);
    } else {
        self.eval_grad_all(nb.view(), eg.mutable_data(), fg.mutable_data(), vg_ptr);
    }

    return py::make_tuple(eg, fg, compute_virial_grad ? py::object(vg) : py::none());
}

static py::tuple eval_grad_radial(MTPTraining& self, const PyNeighbors& nb, bool compute_virial_grad) {
    return grad_block(self, nb, MTPTraining::RADIAL, compute_virial_grad);
}

static py::tuple eval_grad_linear(MTPTraining& self, const PyNeighbors& nb, bool compute_virial_grad) {
    return grad_block(self, nb, MTPTraining::LINEAR, compute_virial_grad);
}

static py::tuple eval_grad_all(MTPTraining& self, const PyNeighbors& nb, bool compute_virial_grad) {
    return grad_block(self, nb, MTPTraining::ALL, compute_virial_grad);
}

static py::tuple compute_with_radial_grad(MTPTraining& self, const PyNeighbors& nb, bool compute_virial_grad) {
    const int n = nb.n_atoms();
    const int n_radial = self.get_radial_coeff_count();
    auto forces = zeros({n, 3});
    auto virials = zeros({6});
    auto eg = zeros({n_radial});
    auto fg = zeros({n, 3, n_radial});
    DoubleArray vg;
    double* vg_ptr = nullptr;
    if (compute_virial_grad) {
        vg = zeros({6, n_radial});
        vg_ptr = vg.mutable_data();
    }

    double energy = self.compute_with_radial_grad(nb.view(), forces.mutable_data(), virials.mutable_data(), eg.mutable_data(), fg.mutable_data(), vg_ptr);

    return py::make_tuple(energy, forces, virials, eg, fg, compute_virial_grad ? py::object(vg) : py::none());
}

void bind_training(py::module_& m) {
    auto cls = py::class_<MTPTraining, PairMTPExtrapolation>(m, "MTPTraining", "An MTP potential that also differentiates energy, forces and virial w.r.t. its coefficients.")
                   .def(py::init<const std::string&>(), py::arg("filename"));

    def_neighbors(cls, "eval_grad_radial", &eval_grad_radial,
                  "Returns (energy_grad(n_radial), force_grad(n,3,n_radial), virial_grad(6,n_radial)|None)",
                  py::arg("compute_virial_grad") = false);

    def_neighbors(cls, "eval_grad_linear", &eval_grad_linear,
                  "Returns (site_energy_grad(inum,n_lin), force_grad(n,3,n_lin), virial_grad(6,n_lin)|None)",
                  py::arg("compute_virial_grad") = false);

    def_neighbors(cls, "eval_grad_all", &eval_grad_all,
                  R"doc(
d(E_i, F, virial)/dc where c = [c_radial | c_species | beta_linear].

Returns (site_energy_grad(inum, cc), force_grad(n_atoms,3,cc), virial_grad(6,cc)|None)
)doc",
                  py::arg("compute_virial_grad") = false);

    def_neighbors(cls, "compute_with_radial_grad", &compute_with_radial_grad,
                  R"doc(
Fused compute plus energy/force/virial radial gradients in one pass.

Returns (energy, forces, virials, energy_grad, force_grad, virial_grad|None).
)doc",
                  py::arg("compute_virial_grad") = false);
}
