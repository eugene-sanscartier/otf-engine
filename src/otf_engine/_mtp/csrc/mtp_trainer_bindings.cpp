/* -*- c++ -*- ----------------------------------------------------------
   Python bindings for MTPTrainer (see mtp_trainer.h). Built into _mtp_ext,
   its ranks are threads (see thread_mpi.h); built with MTP_MPI, this unit is
   the _mtp_mpi module, whose ranks are those of an mpi4py communicator.
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

#ifdef MTP_MPI

static py::object train(const std::string& filename, py::iterable structures, const std::map<std::string, std::string>& options, py::object comm_object, py::object checkpoint, py::object inverse_hessian, bool balance_species) {
    auto make_options = [&](std::string& log_target) {
        MTPTrainerOptions o = parse_options<MTPTrainerOptions>("train_mtp", options, log_target);
        o.balance_species = balance_species;
        o.inverse_hessian = inverse_hessian_buffer(inverse_hessian, o.inverse_hessian_size);
        return o;
    };
    return train_over_mpi<MTPTrainer>("train_mtp", filename, structures, make_options, comm_object, checkpoint);
}

PYBIND11_MODULE(_mtp_mpi, m) {
    m.doc() = "MTPTrainer over MPI.";
    py::module_::import("otf_engine._mtp._mtp_ext");    // registers MTPTraining

    m.def("train_mtp", &train, py::arg("filename"), py::arg("structures"), py::arg("options"), py::arg("comm"), py::arg("checkpoint") = py::none(), py::arg("inverse_hessian") = py::none(), py::arg("balance_species") = true, R"doc(
Train the potential in filename with MTPTrainer over the ranks of comm, and return it.

Every rank calls it with its own share of the structures and the same other arguments, and returns
the fitted potential. An error on any rank is raised on every rank.

structures : this rank's share, an iterable of TrainingStructure
options    : as _mtp_ext.train_mtp's; rank 0 writes the log
comm       : an mpi4py communicator
checkpoint : called on rank 0 with the potential being trained at each save of the fit; a failure
             ends checkpointing and is raised after the fit
inverse_hessian : as _mtp_ext.train_mtp's, left in rank 0's; every rank passes one of the same shape
balance_species : give every species of a structure an equal share of its force weight
)doc");
}

#else

static py::object train(const std::string& filename, py::iterable structures, const std::map<std::string, std::string>& options, int ranks, py::object checkpoint, py::object inverse_hessian, bool balance_species) {
    std::string log_target = "stdout";
    MTPTrainerOptions o = parse_options<MTPTrainerOptions>("train_mtp", options, log_target);
    o.balance_species = balance_species;
    o.inverse_hessian = inverse_hessian_buffer(inverse_hessian, o.inverse_hessian_size);
    return train_on_threads<MTPTrainer>("train_mtp", filename, structures, o, log_target, ranks, checkpoint);
}

// The trainer's objective and its gradient at coeffs, on one thread.
static py::tuple objective(const std::string& filename, py::iterable structures, const std::map<std::string, std::string>& options, py::array_t<double, py::array::c_style | py::array::forcecast> coeffs, bool balance_species) {
    std::string log_target;
    MTPTrainerOptions trainer_options = parse_options<MTPTrainerOptions>("train_mtp", options, log_target);
    trainer_options.balance_species = balance_species;
    std::vector<TrainingStructure> share;
    for (py::handle item : structures)
        share.push_back(item.cast<TrainingStructure>());

    MTPTraining potential(filename);
    std::vector<double> at(coeffs.data(), coeffs.data() + coeffs.size()), grad;
    ThreadGroup group(1);
    ThreadComm comm{&group, 0};
    MTPTrainer trainer(potential, std::move(share), trainer_options, &comm);
    const double loss = trainer.objective(at, grad);
    return py::make_tuple(loss, py::array_t<double>(grad.size(), grad.data()));
}

void bind_mtp_trainer(py::module_& m) {
    m.def("trainer_objective", &objective, py::arg("filename"), py::arg("structures"), py::arg("options"), py::arg("coeffs"), py::arg("balance_species") = true,
          "The loss the trainer's BFGS minimizes and its gradient at coeffs, for gradient tests; the potential in filename must hold every species of structures.");
    m.def("train_mtp", &train, py::arg("filename"), py::arg("structures"), py::arg("options"), py::arg("ranks"), py::arg("checkpoint") = py::none(), py::arg("inverse_hessian") = py::none(), py::arg("balance_species") = true, R"doc(
Train the potential in filename with MTPTrainer on `ranks` threads, and return it.

structures : iterable of TrainingStructure
options    : the fields of MTPTrainerOptions without the leading "--", as strings, log (stdout, none
             or a file), and species, the species' names for the error tables, comma-separated
checkpoint : called with the potential being trained at each save of the fit
inverse_hessian : (n, n) float64 array, n the coefficients BFGS trains (the radial ones under
             radial_only, else all); BFGS starts from it and leaves its final inverse Hessian
             there. None starts from the identity
balance_species : give every species of a structure an equal share of its force weight
)doc");
}

#endif
