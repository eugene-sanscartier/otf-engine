/* -*- c++ -*- ----------------------------------------------------------
   Python bindings for MTP (see mtp.h).
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
    auto cls = py::class_<MTP, PairMTP>(m, "MTP", "MTP potential with its basis values and the derivatives of site energies, forces, virial and losses with respect to its coefficients.");
    def_pair_mtp_init(cls);

    def_neighbors(cls, "eval_basis", &eval_basis,
                  "Return the basis values of each central atom, an ndarray float64 (inum, alpha_scalar_count).");

    def_neighbors(cls, "eval_grad", &eval_grad,
                  R"doc(
Return d(E_i, F, virial)/dc, where c = [c_radial | c_species | beta_linear].

Args:
    forces: also return the force gradient.
    virial: also return the virial gradient; requires forces.
    radial: fill the radial columns; False leaves them zero.

Returns:
    Without forces, site_energy_grad (inum, cc). Row ii is dE_i/dc for atom ilist[ii],
    the candidate vector of its extrapolation grade.
    With forces, (site_energy_grad (inum, cc), force_grad (n_atoms, 3, cc), virial_grad (6, cc) or None).
)doc",
                  py::arg("forces") = false, py::arg("virial") = false, py::arg("radial") = true);
}
