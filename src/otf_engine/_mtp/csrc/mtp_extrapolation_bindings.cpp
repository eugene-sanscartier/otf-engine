/* -*- c++ -*- ----------------------------------------------------------
   Python bindings for PairMTPExtrapolation (see mtp_extrapolation.h).
------------------------------------------------------------------------- */

#include "bindings_common.h"

// --- ported from PairMTPExtrapolation --------------------------------------

static py::dict compute(PairMTPExtrapolation& self, const PyNeighbors& nb) {
    {
        py::gil_scoped_release unlocked;
        self.compute(nb.view());
    }
    py::dict result = potential_results(self, nb.n_atoms());
    result["max_grade"] = self.get_max_grade();
    if (!self.get_configuration_mode())
        result["nbh_grades"] = DoubleArray({nb.n_atoms()}, self.get_nbh_extrapolation_grades());
    return result;
}

void bind_extrapolation(py::module_& m) {
    auto cls = py::class_<PairMTPExtrapolation, PairMTP>(m, "PairMTPExtrapolation", "An MTP potential that also grades configurations against a MaxVol active set.")
                   .def(py::init<const std::string&>(), py::arg("filename"));

    // ---- ported from PairMTPExtrapolation ----
    def_neighbors(cls, "compute", &compute,
                  R"doc(
PairMTP.compute's results, graded against the active set set by set_active_set().

Adds "max_grade", the largest neighborhood grade or the configuration grade,
and in neighborhood mode "nbh_grades" (n_atoms).
)doc");

    // ---- no counterpart in the reference ----
    // The reference reads the active set from the file's #MVS_v1.1 block;
    // almtp_io.py parses and writes that block, and hands invA in here.
    cls.def("set_active_set",
            [](PairMTPExtrapolation& self, DoubleArray invA, bool configuration_mode, int weight_scaling) {
                const int n = self.coeff_count();
                if (invA.size() != (py::ssize_t) n * n)
                    throw std::runtime_error("set_active_set: invA must be coeff_count x coeff_count");
                self.set_active_set(invA.data(), configuration_mode, weight_scaling);
            },
            py::arg("invA"), py::arg("configuration_mode") = false, py::arg("weight_scaling") = 2,
            "Install the MaxVol inverse active set, its mode and its weight_scaling, used by compute().")
        .def_property_readonly("has_active_set", &PairMTPExtrapolation::has_active_set)
        .def_property_readonly("configuration_mode", &PairMTPExtrapolation::get_configuration_mode);
}
