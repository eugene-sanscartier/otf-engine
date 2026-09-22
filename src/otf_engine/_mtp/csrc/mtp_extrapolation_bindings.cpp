/* -*- c++ -*- ----------------------------------------------------------
   Python bindings for PairMTPExtrapolation (see mtp_extrapolation.h).
------------------------------------------------------------------------- */

#include "bindings_common.h"

// --- ported from PairMTPExtrapolation --------------------------------------

static py::tuple grade(PairMTPExtrapolation& self, const PyNeighbors& nb) {
    auto grades = zeros({nb.inum()});
    double cfg_grade = self.grade(nb.view(), grades.mutable_data());
    return py::make_tuple(grades, cfg_grade);
}

// --- no counterpart in the reference ---------------------------------------
// The reference builds energy_ders_wrt_coeffs and discards it after grading;
// select_add needs the per-atom rows themselves.

static DoubleArray eval_grad(PairMTPExtrapolation& self, const PyNeighbors& nb) {
    auto out = zeros({nb.inum(), self.coeff_count()});
    self.eval_grad(nb.view(), out.mutable_data());
    return out;
}

void bind_extrapolation(py::module_& m) {
    auto cls = py::class_<PairMTPExtrapolation, PairMTP>(m, "PairMTPExtrapolation", "An MTP potential that also grades configurations against a MaxVol active set.")
                   .def(py::init<const std::string&>(), py::arg("filename"));

    // ---- ported from PairMTPExtrapolation ----
    def_neighbors(cls, "grade", &grade,
                  R"doc(
Extrapolation grades against the active set set by set_active_set().

Returns (per_atom_grades (inum,), configuration_grade). In configuration mode
every per-atom entry is the configuration grade.
)doc");

    // ---- no counterpart in the reference ----
    def_neighbors(cls, "eval_grad", &eval_grad,
                  R"doc(
Per-atom information vector for extrapolation grade computation.

ndarray float64 (inum, coeff_count); row ii is dE_i/dc for atom ilist[ii],
laid out as [radial | species one-hot | basis values].
)doc");

    // The reference reads the active set from the file's #MVS_v1.1 block;
    // almtp_io.py parses and writes that block, and hands invA in here.
    cls.def("set_active_set",
            [](PairMTPExtrapolation& self, DoubleArray invA, bool configuration_mode) {
                const int n = self.coeff_count();
                if (invA.size() != (py::ssize_t) n * n)
                    throw std::runtime_error("set_active_set: invA must be coeff_count x coeff_count");
                self.set_active_set(invA.data(), configuration_mode);
            },
            py::arg("invA"), py::arg("configuration_mode") = false,
            "Install the MaxVol inverse active set used by grade().")
        .def_property_readonly("has_active_set", &PairMTPExtrapolation::has_active_set)
        .def_property_readonly("configuration_mode", &PairMTPExtrapolation::get_configuration_mode)
        // The reference declares coeff_count on this class, not on PairMTP.
        .def("get_coeff_count", &PairMTPExtrapolation::coeff_count, "radial_coeff_count + species_count + alpha_scalar_count");
}
