/* -*- c++ -*- ----------------------------------------------------------
   Python bindings for PairETNExtrapolation (see pair_etn_extrapolation.h).
------------------------------------------------------------------------- */

#include "pair_etn_bindings.h"
#include "pair_etn_extrapolation.h"

static py::dict compute(PairETNExtrapolation& self, const PyNeighbors& nb) {
    py::dict result = compute_results(self, nb);
    result["max_grade"] = self.get_max_grade();
    if (!self.get_configuration_mode())
        result["nbh_grades"] = to_array(self.get_nbh_extrapolation_grades());
    return result;
}

void bind_pair_etn_extrapolation(py::module_& m) {
    auto cls = py::class_<PairETNExtrapolation, PairETN>(m, "PairETNExtrapolation", "An ETN potential that also grades configurations against a MaxVol active set.");
    def_pair_etn_init(cls);

    def_neighbors(cls, "compute", &compute,
                  R"doc(
PairETN.compute's results, graded against the active set set by set_active_set().

Adds "max_grade", the largest neighborhood grade or the configuration grade,
and in neighborhood mode "nbh_grades" (n_atoms).
)doc");

    cls.def("set_active_set",
            [](PairETNExtrapolation& self, DoubleArray invA, bool configuration_mode, int weight_scaling) {
                const int n = self.coeff_count();
                if (invA.size() != (py::ssize_t) n * n)
                    throw std::runtime_error("set_active_set: invA must be coeff_count x coeff_count");
                self.set_active_set(invA.data(), configuration_mode, weight_scaling);
            },
            py::arg("invA"), py::arg("configuration_mode") = false, py::arg("weight_scaling") = 2,
            "Install the MaxVol inverse active set, its mode and its weight_scaling, used by compute().")
        .def_property_readonly("has_active_set", &PairETNExtrapolation::has_active_set)
        .def_property_readonly("configuration_mode", &PairETNExtrapolation::get_configuration_mode);
}
