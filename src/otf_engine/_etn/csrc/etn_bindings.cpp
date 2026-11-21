/* -*- c++ -*- ----------------------------------------------------------
   Python bindings for ETN (see etn.h). Its eval_grad and
   eval_loss_grad are Pair's bindings.
------------------------------------------------------------------------- */

#include "etn.h"
#include "pair_etn_bindings.h"

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
    auto cls = py::class_<ETN, PairETN>(m, "ETN", "An ETN potential that also gives its input and bond vectors, and differentiates site energies, and losses on its energy, forces and virial, w.r.t. its coefficients.");
    def_pair_etn_init(cls);

    def_neighbors(cls, "eval_bonds", &eval_bonds,
                  R"doc(
Each central atom's input and bond vectors.

Returns a dict with "inputs" (inum, input size), and "bonds" and "bonds_im",
lists of (inum, bond size) arrays for bonds 0 .. core_count. A covariant vector
holds entry (l, m, n) at offset(l) + (m + l) * rank(l) + n.
)doc");
}
