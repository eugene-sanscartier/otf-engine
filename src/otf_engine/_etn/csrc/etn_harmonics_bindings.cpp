/* -*- c++ -*- ----------------------------------------------------------
   Python bindings for the real harmonics and their 3-j symbols (see
   etn_harmonics.h).
------------------------------------------------------------------------- */

#include "bindings_common.h"
#include "etn_harmonics.h"

void bind_etn_harmonics(py::module_& m) {
    m.def("real_harmonics",
          [](int l_count, DoubleArray u) {
              RealHarmonics harmonics(l_count);
              harmonics.compute(u.data());
              return py::make_tuple(DoubleArray({l_count * l_count}, harmonics.vals.data()), DoubleArray({l_count * l_count, 3}, harmonics.grads.data()));
          },
          py::arg("l_count"), py::arg("u"), "Return the real harmonics Y_lm at the unit vector u, at index l * l + l + m, and their gradients tangent to the sphere.");
    m.def("wigner_3j", &wigner_3j, py::arg("l1"), py::arg("l2"), py::arg("l3"), py::arg("m1"), py::arg("m2"), py::arg("m3"));
    m.def("real_wigner_3j", &real_wigner_3j, py::arg("l1"), py::arg("m1"), py::arg("l2"), py::arg("m2"), py::arg("l3"), py::arg("m3"),
          "Return the 3-j symbol of real harmonics: the real part for an even l1 + l2 + l3, the imaginary part for an odd one.");
}
