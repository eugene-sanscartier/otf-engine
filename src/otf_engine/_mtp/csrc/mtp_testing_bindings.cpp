/* -*- c++ -*- ----------------------------------------------------------
   The _mtp_testing module: MTPTrainer's stages one at a time, and BFGS on
   the Rosenbrock function, for the tests. The package never imports it.
------------------------------------------------------------------------- */

#include "bindings_common.h"

#include "mtp_trainer.h"

#include <sstream>

void add_option_fields(MTPTrainerOptions& o, OptionFields& fields);    // mtp_trainer_bindings.cpp

// An MTPTrainer on one thread, whose stages the tests call in any order.
class MTPTrainerStages {
  public:
    MTPTrainerStages(const std::string& filename, py::iterable structures, const std::map<std::string, std::string>& options) : potential(filename), process("MTPTrainerStages", 1, py::none()) {
        std::string log_target;
        const MTPTrainerOptions o = parse_options<MTPTrainerOptions>("MTPTrainerStages", options, log_target);
        trainer = std::make_unique<MTPTrainer>(potential, std::move(split_structures(structures, 1)[0]), o, process[0], &log);
    }

    void add_species() { trainer->add_species(); }
    void update_min_dist() { trainer->update_min_dist(); }
    void orthogonalize() { trainer->orthogonalize(); }
    void fit_linear() { trainer->fit_linear(); }
    void rescale() { trainer->rescale(); }
    void fit_nonlinear(int max_iter) { trainer->fit_nonlinear(max_iter); }
    void fit_riemannian(int max_iter) { trainer->fit_riemannian(max_iter); }
    void train() { trainer->train(); }

    py::tuple objective(const std::vector<double>& at) {
        std::vector<double> grad;
        const double loss = trainer->objective(at, grad);
        return py::make_tuple(loss, py::array_t<double>(grad.size(), grad.data()));
    }

    py::array_t<double> get_coeffs() const { return py::array_t<double>(trainer->coeffs.size(), trainer->coeffs.data()); }
    void set_coeffs(const std::vector<double>& c) {
        trainer->coeffs = c;
        potential.set_coeffs(c.data());
    }
    py::array_t<double> get_reg_vector() const { return py::array_t<double>(trainer->reg_vector.size(), trainer->reg_vector.data()); }
    void set_reg_vector(const std::vector<double>& r) { trainer->reg_vector = r; }

    MTP potential;
    std::ostringstream log;

  private:
    ProcessRanks process;
    std::unique_ptr<MTPTrainer> trainer;
};

// The N-dimensional Rosenbrock function and its gradient, with the operations of mlip3_probe.cpp.
static double rosenbrock(const std::vector<double>& x, std::vector<double>& g) {
    const int n = (int) x.size();
    double f = 0.0;
    g.assign(n, 0.0);
    for (int i = 0; i + 1 < n; i++) {
        const double a = x[i + 1] - x[i] * x[i];
        const double b = 1.0 - x[i];
        f += 100.0 * a * a + b * b;
        g[i] += -400.0 * a * x[i] - 2.0 * b;
        g[i + 1] += 200.0 * a;
    }
    return f;
}

// BFGS on the n-dimensional Rosenbrock function from x_i = -1.2 (even i) and 1: x before each of steps calls to
// iterate, and f, in_linesearch and iterate's result after it.
static py::tuple bfgs_rosenbrock(int n, int steps) {
    std::vector<double> x(n), g;
    for (int i = 0; i < n; i++) x[i] = i % 2 == 0 ? -1.2 : 1.0;
    BFGS bfgs;
    bfgs.set_x(x.data(), n);
    bfgs.reset_hessian();
    py::array_t<double> xs({steps, n}), states({steps, 3});
    auto xs_view = xs.mutable_unchecked<2>();
    auto states_view = states.mutable_unchecked<2>();
    for (int step = 0; step < steps; step++) {
        x = bfgs.x();
        const double f = rosenbrock(x, g);
        const bool decreased = bfgs.iterate(f, g);
        for (int i = 0; i < n; i++) xs_view(step, i) = x[i];
        states_view(step, 0) = f;
        states_view(step, 1) = bfgs.in_linesearch();
        states_view(step, 2) = decreased;
    }
    return py::make_tuple(xs, states);
}

// The trainer's objective and its gradient at coeffs, with the structures dealt round-robin over `ranks` threads.
static py::tuple objective_on_threads(const std::string& filename, py::iterable structures, const std::map<std::string, std::string>& options, const std::vector<double>& coeffs, int ranks) {
    std::string log_target;
    const MTPTrainerOptions o = parse_options<MTPTrainerOptions>("objective_on_threads", options, log_target);
    ProcessRanks process("objective_on_threads", ranks, py::none());
    std::vector<std::vector<Sample>> shares = split_structures(structures, ranks);

    double loss = 0.0;
    std::vector<double> grad;
    process.run([&](Comm& comm, int r) {
        MTP potential(filename);
        MTPTrainer trainer(potential, std::move(shares[r]), o, comm);
        std::vector<double> rank_grad;
        const double rank_loss = trainer.objective(coeffs, rank_grad);
        if (r == 0) {
            loss = rank_loss;
            grad = rank_grad;
        }
    });
    return py::make_tuple(loss, py::array_t<double>(grad.size(), grad.data()));
}

PYBIND11_MODULE(_mtp_testing, m) {
    m.doc() = "MTPTrainer's stages and BFGS, for the tests.";
    py::module_::import("otf_engine._mtp._mtp_ext");    // registers MTP and Sample

    py::class_<MTPTrainerStages>(m, "MTPTrainerStages")
        .def(py::init<const std::string&, py::iterable, const std::map<std::string, std::string>&>(), py::arg("filename"), py::arg("structures"), py::arg("options"))
        .def("add_species", &MTPTrainerStages::add_species)
        .def("update_min_dist", &MTPTrainerStages::update_min_dist)
        .def("orthogonalize", &MTPTrainerStages::orthogonalize)
        .def("fit_linear", &MTPTrainerStages::fit_linear)
        .def("rescale", &MTPTrainerStages::rescale)
        .def("fit_nonlinear", &MTPTrainerStages::fit_nonlinear, py::arg("max_iter"))
        .def("fit_riemannian", &MTPTrainerStages::fit_riemannian, py::arg("max_iter"))
        .def("train", &MTPTrainerStages::train)
        .def("objective", &MTPTrainerStages::objective, py::arg("coeffs"))
        .def_property("coeffs", &MTPTrainerStages::get_coeffs, &MTPTrainerStages::set_coeffs)
        .def_property("reg_vector", &MTPTrainerStages::get_reg_vector, &MTPTrainerStages::set_reg_vector)
        .def_property("scaling", [](MTPTrainerStages& s) { return s.potential.get_scaling(); }, [](MTPTrainerStages& s, double v) { s.potential.set_scaling(v); })
        .def_property_readonly("min_dist", [](MTPTrainerStages& s) { return s.potential.get_min_cutoff(); })
        .def_property_readonly("log", [](MTPTrainerStages& s) { return s.log.str(); })
        .def_property_readonly("potential", [](MTPTrainerStages& s) -> MTP& { return s.potential; }, py::return_value_policy::reference_internal);

    m.def("bfgs_rosenbrock", &bfgs_rosenbrock, py::arg("n"), py::arg("steps"));
    m.def("objective_on_threads", &objective_on_threads, py::arg("filename"), py::arg("structures"), py::arg("options"), py::arg("coeffs"), py::arg("ranks"));
}
