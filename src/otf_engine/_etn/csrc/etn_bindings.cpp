/* -*- c++ -*- ----------------------------------------------------------
   Python bindings for ETN (see etn.h). The Pair binding covers
   eval_loss_grad.
------------------------------------------------------------------------- */

#include "etn.h"
#include "pair_etn_bindings.h"

static py::object eval_grad(ETN& self, const PyNeighbors& nb, bool forces, bool virial, bool nonlinear) {
    return grad_arrays(nb, self.coeff_count(), forces, virial, [&](double* eg, double* fg, double* vg) { self.eval_grad(nb.view(), eg, fg, vg, nonlinear); });
}

static py::dict eval_bonds(ETN& self, const PyNeighbors& nb) {
    const int inum = nb.inum();
    auto inputs = zeros({inum, self.get_input_shape().size});
    std::vector<double> bonds((size_t) inum * self.get_bonds_size());
    double* inputs_ptr = inputs.mutable_data();
    {
        py::gil_scoped_release unlocked;
        self.eval_bonds(nb.view(), inputs_ptr, bonds.data());
    }

    py::list real_parts, imaginary_parts;
    for (int b = 0; b <= self.get_core_count(); b++) {
        const int size = self.get_bond_shape(b).size;
        DoubleArray re({inum, size}), im({inum, size});
        for (int ii = 0; ii < inum; ii++) {
            const double* bond = bonds.data() + (size_t) ii * self.get_bonds_size() + self.get_bond_offset(b);
            std::copy(bond, bond + size, re.mutable_data() + (size_t) ii * size);
            std::copy(bond + size, bond + 2 * size, im.mutable_data() + (size_t) ii * size);
        }
        real_parts.append(re);
        imaginary_parts.append(im);
    }

    py::dict result;
    result["inputs"] = inputs;
    result["bonds"] = real_parts;
    result["bonds_im"] = imaginary_parts;
    return result;
}

void bind_etn(py::module_& m) {
    auto cls = py::class_<ETN, PairETN>(m, "ETN", "An ETN potential that gives its input and bond vectors, and the gradients of its site energies and of a loss with respect to its coefficients.");
    def_pair_etn_init(cls);

    def_neighbors(cls, "eval_bonds", &eval_bonds,
                  R"doc(
Compute the input and bond vectors of each central atom.

Returns:
    dict with "inputs" (inum, input size), and "bonds" and "bonds_im", lists of
    (inum, bond size) arrays for bonds 0 .. core_count. Entry (l, m, n) of a
    covariant vector is at offset(l) + (m + l) * rank(l) + n.
)doc");
    def_neighbors(cls, "eval_grad", &eval_grad,
                  R"doc(
Compute d(E_i, F, virial)/dc, with c in the get_coeffs order.

Args:
    forces: also return the force and virial gradients; requires nonlinear False.
    virial: return the virial gradient; requires forces.
    nonlinear: if False, fill only the columns of the last core and of the species.

Returns:
    site_energy_grad (inum, cc), whose row ii is dE_i/dc for atom ilist[ii].
    With forces, (site_energy_grad, force_grad (n_atoms, 3, cc), virial_grad (6, cc) or None).
)doc",
                  py::arg("forces") = false, py::arg("virial") = false, py::arg("nonlinear") = true);
}
