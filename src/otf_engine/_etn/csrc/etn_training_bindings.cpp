/* -*- c++ -*- ----------------------------------------------------------
   Python bindings for ETNTraining (see etn_training.h).

   mlip-4 takes these derivatives by differentiating its program; nothing
   here binds a function of its. The site-energy rows also give
   Equations.add an ETNTraining overload, on _mtp_ext's Equations class.
------------------------------------------------------------------------- */

#include "bindings_common.h"
#include "etn_training.h"
#include "maxvol.h"

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

    // mlip-4's CfgCalc selects on site energies (nbh) or the energy over N (cfg), and has no force rows.
    py::object equations = py::module_::import("otf_engine._mtp._mtp_ext").attr("Equations");
    equations.attr("add") = py::cpp_function(
        [](Equations& self, ETNTraining& pot, const PyNeighbors& nb, int structure) {
            check_species(pot, nb);
            if (pot.coeff_count() != self.coeff_count)
                throw std::runtime_error("Equations: a potential of " + std::to_string(pot.coeff_count()) + " coefficients, not " + std::to_string(self.coeff_count));
            if (self.needs_forces())
                throw std::runtime_error("Equations: an ETN selects on energies and site energies only; force_weight and stress_weight must be 0");
            py::gil_scoped_release unlocked;
            std::vector<double> site_energy_grad((size_t) nb.inum() * self.coeff_count);
            pot.eval_grad(nb.view(), site_energy_grad.data());
            self.add(nb.view(), site_energy_grad.data(), nullptr, nullptr, structure);
        },
        py::name("add"), py::is_method(equations), py::sibling(py::getattr(equations, "add", py::none())),
        py::arg("potential"), py::arg("neighbors"), py::arg("structure") = -1,
        "Append the equations of one structure under an ETN's current coefficients, numbered structure (by default the next).");
}
