/* -*- c++ -*- ----------------------------------------------------------
   Python bindings for MTPTrainer (see mtp_trainer.h), whose ranks are
   threads or those of an mpi4py communicator.
------------------------------------------------------------------------- */

#include "bindings_common.h"

#include "mtp_trainer.h"

// MTPTrainerOptions' fields beside the weights.
void add_option_fields(MTPTrainerOptions& o, OptionFields& fields) {
    fields.doubles.insert({{"penalty_weight", &o.penalty_weight}, {"scale_by_force", &o.scale_by_force}, {"select_factor", &o.select_factor},
                           {"tolerance", &o.tolerance}, {"reg_param", &o.reg_param}, {"init_noise", &o.init_noise}});
    fields.ints = {{"weight_scaling", &o.weight_scaling}, {"weight_scaling_forces", &o.weight_scaling_forces}, {"iteration_limit", &o.iteration_limit}, {"random_seed", &o.random_seed}, {"lbfgs_memory", &o.lbfgs_memory}};
    fields.bools = {{"no_mindist_update", &o.no_mindist_update}, {"init_random", &o.init_random}, {"skip_preinit", &o.skip_preinit}, {"radial_only", &o.radial_only}, {"bfgs_only", &o.bfgs_only}, {"reg_once", &o.reg_once}, {"reg_at_main", &o.reg_at_main}, {"varpro", &o.varpro}, {"no_orthogonalize", &o.no_orthogonalize}, {"riemannian", &o.riemannian}};
}

// The buffer of inverse_hessian, a writable C-contiguous (n, n) float64 array, and its n; nullptr and 0 for None.
static double* inverse_hessian_buffer(py::object inverse_hessian, int& n) {
    n = 0;
    if (inverse_hessian.is_none()) return nullptr;
    auto array = inverse_hessian.cast<py::array_t<double, py::array::c_style>>();
    if (!array.is(inverse_hessian))    // a conversion copied it, and the result would land in the copy
        throw std::runtime_error("train_mtp: inverse_hessian must be a C-contiguous float64 array");
    if (array.ndim() != 2 || array.shape(0) != array.shape(1))
        throw std::runtime_error("train_mtp: inverse_hessian must be a square matrix");
    n = (int) array.shape(0);
    return array.mutable_data();
}

static py::object train(const std::string& filename, py::iterable structures, const std::map<std::string, std::string>& options, int ranks, py::object comm, py::object checkpoint, py::object inverse_hessian, bool balance_species) {
    auto make_options = [&](std::string& log_target) {
        MTPTrainerOptions o = parse_options<MTPTrainerOptions>("train_mtp", options, log_target);
        o.balance_species = balance_species;
        o.inverse_hessian = inverse_hessian_buffer(inverse_hessian, o.inverse_hessian_size);
        return o;
    };
    return train_potential<MTPTrainer>("train_mtp", filename, structures, make_options, ranks, comm, checkpoint);
}

// The trainer's objective and its gradient at coeffs, on one thread.
static py::tuple objective(const std::string& filename, py::iterable structures, const std::map<std::string, std::string>& options, py::array_t<double, py::array::c_style | py::array::forcecast> coeffs, bool balance_species) {
    std::string log_target;
    MTPTrainerOptions trainer_options = parse_options<MTPTrainerOptions>("train_mtp", options, log_target);
    trainer_options.balance_species = balance_species;
    std::vector<Sample> share;
    for (py::handle item : structures)
        share.push_back(item.cast<Sample>());

    MTP potential(filename);
    std::vector<double> at(coeffs.data(), coeffs.data() + coeffs.size()), grad;
    ProcessRanks process("trainer_objective", 1, py::none());
    MTPTrainer trainer(potential, std::move(share), trainer_options, process[0]);
    const double loss = trainer.objective(at, grad);
    return py::make_tuple(loss, py::array_t<double>(grad.size(), grad.data()));
}

void bind_mtp_trainer(py::module_& m) {
    m.def("trainer_objective", &objective, py::arg("filename"), py::arg("structures"), py::arg("options"), py::arg("coeffs"), py::arg("balance_species") = true,
          "The loss the trainer's BFGS minimizes and its gradient at coeffs, for gradient tests; the potential in filename must hold every species of structures.");
    m.def("train_mtp", &train, py::arg("filename"), py::arg("structures"), py::arg("options"), py::arg("ranks") = 1, py::arg("comm") = py::none(), py::arg("checkpoint") = py::none(), py::arg("inverse_hessian") = py::none(), py::arg("balance_species") = true, R"doc(
Train the potential in filename with MTPTrainer on `ranks` threads, or over the ranks of comm, and return it.

structures : iterable of Sample; under comm, this rank's share
options    : the fields of MTPTrainerOptions without the leading "--", as strings, log (stdout, none
             or a file), which rank 0 writes, and species, the species' names for the error tables,
             comma-separated
ranks      : threads
comm       : an mpi4py communicator, in place of threads. Every rank calls with the same other
             arguments and returns the fitted potential; the trainer raises its errors on every rank
checkpoint : called on rank 0 with the potential being trained at each save of the fit; its first
             failure is a warning and ends checkpointing
inverse_hessian : (n, n) float64 array, n the coefficients BFGS trains (the radial ones under
             radial_only, else all); BFGS starts from it and leaves its final inverse Hessian
             in rank 0's. None starts from the identity
balance_species : give every species of a structure an equal share of its force weight
)doc");
}
