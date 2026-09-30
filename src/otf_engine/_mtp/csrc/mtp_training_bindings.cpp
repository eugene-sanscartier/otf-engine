/* -*- c++ -*- ----------------------------------------------------------
   Python bindings for MTPTraining (see mtp_training.h).

   Nothing here has a counterpart in the reference, which is a pair style and
   differentiates nothing w.r.t. the coefficients.
------------------------------------------------------------------------- */

#include "bindings_common.h"

static py::object eval_grad(MTPTraining& self, const PyNeighbors& nb, bool forces, bool virial, bool radial) {
    const int width = self.coeff_count();
    auto eg = zeros({nb.inum(), width});
    double* eg_ptr = eg.mutable_data();

    if (!forces) {
        if (virial)
            throw std::runtime_error("eval_grad: the virial gradient comes with the force gradient");
        {
            py::gil_scoped_release unlocked;
            if (radial)
                self.PairMTPExtrapolation::eval_grad(nb.view(), eg_ptr);
            else
                self.eval_grad(nb.view(), eg_ptr, nullptr, nullptr, false);
        }
        return eg;
    }

    auto fg = zeros({nb.n_atoms(), 3, width});
    DoubleArray vg;
    double* vg_ptr = nullptr;
    if (virial) {
        vg = zeros({6, width});
        vg_ptr = vg.mutable_data();
    }
    double* fg_ptr = fg.mutable_data();

    {
        py::gil_scoped_release unlocked;
        self.eval_grad(nb.view(), eg_ptr, fg_ptr, vg_ptr, radial);
    }

    return py::make_tuple(eg, fg, virial ? py::object(vg) : py::none());
}

static DoubleArray eval_loss_grad(MTPTraining& self, const PyNeighbors& nb, double dloss_denergy, py::object dloss_dforces, py::object dloss_dvirial) {
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

void bind_training(py::module_& m) {
    auto cls = py::class_<MTPTraining, PairMTPExtrapolation>(m, "MTPTraining", "An MTP potential that also differentiates site energies, forces and virial, and losses on them, w.r.t. its coefficients.")
                   .def(py::init<const std::string&>(), py::arg("filename"));

    def_neighbors(cls, "eval_grad", &eval_grad,
                  R"doc(
d(E_i, F, virial)/dc where c = [c_radial | c_species | beta_linear].

Without forces, returns site_energy_grad (inum, cc), as PairMTPExtrapolation.eval_grad.
With forces, returns (site_energy_grad (inum, cc), force_grad (n_atoms, 3, cc), virial_grad (6, cc)|None).
With radial False, the radial columns are zero.
)doc",
                  py::arg("forces") = false, py::arg("virial") = false, py::arg("radial") = true);

    def_neighbors(cls, "eval_loss_grad", &eval_loss_grad,
                  R"doc(
dL/dc (cc,) for a loss L with the given derivatives w.r.t. the energy,
the forces (n_atoms, 3) and the virial (6, xx,yy,zz,xy,xz,yz); None is zero.
)doc",
                  py::arg("dloss_denergy"), py::arg("dloss_dforces") = py::none(), py::arg("dloss_dvirial") = py::none());
}
