/* -*- c++ -*- ----------------------------------------------------------
   Python bindings for ETNTraining (see etn_training.h).

   mlip-4 takes these derivatives by differentiating its program; nothing
   here binds a function of its.
------------------------------------------------------------------------- */

#include "bindings_common.h"
#include "etn_training.h"

static DoubleArray eval_grad(ETNTraining& self, const PyNeighbors& nb) {
    auto out = zeros({nb.inum(), self.coeff_count()});
    double* out_ptr = out.mutable_data();
    {
        py::gil_scoped_release unlocked;
        self.eval_grad(nb.view(), out_ptr);
    }
    return out;
}

static DoubleArray eval_loss_grad(ETNTraining& self, const PyNeighbors& nb, double dloss_denergy, py::object dloss_dforces, py::object dloss_dvirial) {
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

void bind_etn_training(py::module_& m) {
    auto cls = py::class_<ETNTraining, PairETN>(m, "ETNTraining", "An ETN potential that also differentiates site energies, and losses on its energy, forces and virial, w.r.t. its coefficients.")
                   .def(py::init<const std::string&>(), py::arg("filename"));

    cls.def("get_coeffs",
            [](const ETNTraining& self) {
                auto out = zeros({self.coeff_count()});
                self.get_coeffs(out.mutable_data());
                return out;
            },
            "Every coefficient, in mlip-4's params order.")
        .def("set_coeffs",
             [](ETNTraining& self, DoubleArray arr) {
                 if (arr.size() != self.coeff_count())
                     throw std::runtime_error("set_coeffs: expected length " + std::to_string(self.coeff_count()));
                 self.set_coeffs(arr.data());
             },
             py::arg("coeffs"));

    def_neighbors(cls, "eval_grad", &eval_grad,
                  "dE_i/dc per central atom: ndarray float64 (inum, coeff_count), c in mlip-4's params order.");

    def_neighbors(cls, "eval_loss_grad", &eval_loss_grad,
                  R"doc(
dL/dc (coeff_count,) for a loss L with the given derivatives w.r.t. the energy,
the forces (n_atoms, 3) and the virial (6, xx,yy,zz,xy,xz,yz); None is zero.
)doc",
                  py::arg("dloss_denergy"), py::arg("dloss_dforces") = py::none(), py::arg("dloss_dvirial") = py::none());
}
