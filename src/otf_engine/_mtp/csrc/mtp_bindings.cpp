/* -*- c++ -*- ----------------------------------------------------------
   Python bindings for MTP (see mtp.h).

   Nothing here has a counterpart in the reference, which is a pair style and
   differentiates nothing w.r.t. the coefficients.
------------------------------------------------------------------------- */

#include "mtp.h"
#include "pair_mtp_bindings.h"

static DoubleArray eval_basis(MTP& self, const PyNeighbors& nb) {
    auto out = zeros({nb.inum(), self.get_alpha_scalar_count()});
    double* out_ptr = out.mutable_data();
    {
        py::gil_scoped_release unlocked;
        self.eval_basis(nb.view(), out_ptr);
    }
    return out;
}

static py::object eval_grad(MTP& self, const PyNeighbors& nb, bool forces, bool virial, bool radial) {
    return grad_arrays(nb, self.coeff_count(), forces, virial, [&](double* eg, double* fg, double* vg) { self.eval_grad(nb.view(), eg, fg, vg, radial); });
}

void bind_mtp(py::module_& m) {
    auto cls = py::class_<MTP, PairMTP>(m, "MTP", "An MTP potential that also gives its basis values, and differentiates site energies, forces and virial, and losses on them, w.r.t. its coefficients.");
    def_pair_mtp_init(cls);

    def_neighbors(cls, "eval_basis", &eval_basis,
                  "Basis values per central atom: ndarray float64 (inum, alpha_scalar_count).");

    def_neighbors(cls, "eval_grad", &eval_grad,
                  R"doc(
d(E_i, F, virial)/dc where c = [c_radial | c_species | beta_linear].

Without forces, returns site_energy_grad (inum, cc); row ii is dE_i/dc for
atom ilist[ii], the information vector extrapolation grades are taken from.
With forces, returns (site_energy_grad (inum, cc), force_grad (n_atoms, 3, cc), virial_grad (6, cc)|None).
With radial False, the radial columns are zero.
)doc",
                  py::arg("forces") = false, py::arg("virial") = false, py::arg("radial") = true);
}
