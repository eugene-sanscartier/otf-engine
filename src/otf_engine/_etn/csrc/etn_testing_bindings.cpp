/* -*- c++ -*- ----------------------------------------------------------
   The _etn_testing module: an ETN fit with the loss of each of its
   evaluations, for the tests. The package never imports it.
------------------------------------------------------------------------- */

#include "bindings_common.h"
#include "etn_fit.h"

void add_option_fields(ETNFitOptions& o, OptionFields& fields);    // etn_fit_bindings.cpp

static py::tuple fit_etn_steps(const std::string& filename, py::iterable structures, const std::map<std::string, std::string>& options, int ranks, py::object comm) {
    std::string log_target;
    const ETNFitOptions o = parse_options<ETNFitOptions>("fit_etn_steps", options, log_target);
    std::vector<std::vector<double>> step_losses;
    py::object potential = train_potential<ETNFit>("fit_etn_steps", filename, structures, o, log_target, ranks, comm, py::none(), [&](const ETNFit& fit) { step_losses = fit.step_losses(); });
    return py::make_tuple(potential, step_losses);
}

PYBIND11_MODULE(_etn_testing, m) {
    m.doc() = "An ETN fit with its losses, for the tests.";
    py::module_::import("otf_engine._etn._etn_ext");    // registers ETN and FitStructure

    m.def("fit_etn_steps", &fit_etn_steps, py::arg("filename"), py::arg("structures"), py::arg("options"), py::arg("ranks") = 1, py::arg("comm") = py::none(), R"doc(
fit_etn's fit, returning the trained potential and, on rank 0, the loss at each evaluation of each step, a list per
step over every phase of the fit; other ranks get no steps.
)doc");
}
