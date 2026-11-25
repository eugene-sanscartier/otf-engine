/* -*- c++ -*- ----------------------------------------------------------
   The _etn_testing module: an ETN fit with the loss at each evaluation,
   and the fit objective. The package never imports it.
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

// Evaluates the objective of an ETNFit on one thread at given coefficients. The ridge stays that of the first call.
class ETNObjective {
  public:
    ETNObjective(const std::string& filename, py::iterable structures, const std::map<std::string, std::string>& options) : potential(filename), process("ETNObjective", 1, py::none()) {
        std::string log_target;
        const ETNFitOptions o = parse_options<ETNFitOptions>("ETNObjective", options, log_target);
        fit = std::make_unique<ETNFit>(potential, std::move(split_structures(structures, 1)[0]), o, process[0]);
    }

    py::tuple operator()(const std::vector<double>& coeffs) {
        std::vector<double> grad;
        const double loss = fit->objective(coeffs, grad);
        return py::make_tuple(loss, py::array_t<double>(grad.size(), grad.data()));
    }

  private:
    ETN potential;
    ProcessRanks process;
    std::unique_ptr<ETNFit> fit;
};

PYBIND11_MODULE(_etn_testing, m) {
    m.doc() = "ETN fit with its losses, and its objective, for the tests.";
    py::module_::import("otf_engine._etn._etn_ext");    // registers ETN and FitStructure

    m.def("fit_etn_steps", &fit_etn_steps, py::arg("filename"), py::arg("structures"), py::arg("options"), py::arg("ranks") = 1, py::arg("comm") = py::none(), R"doc(
Run the fit of fit_etn and return the trained potential with the losses of its steps.

Returns:
    (potential, step_losses). On rank 0, step_losses holds one list per step over all phases
    of the fit, with the loss at each evaluation. On the other ranks it is empty.
)doc");

    py::class_<ETNObjective>(m, "ETNObjective", "Objective of ETNFit on one thread over the structures; a call at coefficients returns (loss, gradient).")
        .def(py::init<const std::string&, py::iterable, const std::map<std::string, std::string>&>(), py::arg("filename"), py::arg("structures"), py::arg("options"))
        .def("__call__", &ETNObjective::operator(), py::arg("coeffs"));
}
