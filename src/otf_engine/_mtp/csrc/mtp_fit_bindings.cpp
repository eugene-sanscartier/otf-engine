/* -*- c++ -*- ----------------------------------------------------------
   Python bindings for MTPFit (see mtp_fit.h), whose ranks are
   threads or those of an mpi4py communicator.
------------------------------------------------------------------------- */

#include "bindings_common.h"

#include "mtp_fit.h"

// MTPFitOptions' fields beside the weights.
void add_option_fields(MTPFitOptions& o, OptionFields& fields) {
    fields.doubles.insert({{"penalty_weight", &o.penalty_weight}, {"scale_by_force", &o.scale_by_force}, {"select_factor", &o.select_factor},
                           {"tolerance", &o.tolerance}, {"reg_param", &o.reg_param}, {"init_noise", &o.init_noise}});
    fields.ints = {{"weight_scaling", &o.weight_scaling}, {"weight_scaling_forces", &o.weight_scaling_forces}, {"iteration_limit", &o.iteration_limit}, {"random_seed", &o.random_seed}, {"lbfgs_memory", &o.lbfgs_memory}};
    fields.bools.insert({{"no_mindist_update", &o.no_mindist_update}, {"init_random", &o.init_random}, {"skip_preinit", &o.skip_preinit}, {"radial_only", &o.radial_only}, {"bfgs_only", &o.bfgs_only}, {"reg_once", &o.reg_once}, {"reg_at_main", &o.reg_at_main}, {"varpro", &o.varpro}, {"no_orthogonalize", &o.no_orthogonalize}, {"riemannian", &o.riemannian}, {"ldlt", &o.ldlt}, {"reg_follows_scaling", &o.reg_follows_scaling}});
}

// The buffer of inverse_hessian, a writable C-contiguous (n, n) float64 array, and its n; nullptr and 0 for None.
static double* inverse_hessian_buffer(py::object inverse_hessian, int& n) {
    n = 0;
    if (inverse_hessian.is_none()) return nullptr;
    auto array = inverse_hessian.cast<py::array_t<double, py::array::c_style>>();
    if (!array.is(inverse_hessian))    // a conversion copied it, and the result would land in the copy
        throw std::runtime_error("fit_mtp: inverse_hessian must be a C-contiguous float64 array");
    if (array.ndim() != 2 || array.shape(0) != array.shape(1))
        throw std::runtime_error("fit_mtp: inverse_hessian must be a square matrix");
    n = (int) array.shape(0);
    return array.mutable_data();
}

static py::object train(const std::string& filename, py::iterable structures, const std::map<std::string, std::string>& options, int ranks, py::object comm, py::object checkpoint, py::object inverse_hessian) {
    std::string log_target;
    MTPFitOptions o = parse_options<MTPFitOptions>("fit_mtp", options, log_target);
    o.inverse_hessian = inverse_hessian_buffer(inverse_hessian, o.inverse_hessian_size);
    return train_potential<MTPFit>("fit_mtp", filename, structures, o, log_target, ranks, comm, checkpoint);
}

void bind_mtp_fit(py::module_& m) {
    m.def("fit_mtp", &train, py::arg("filename"), py::arg("structures"), py::arg("options"), py::arg("ranks") = 1, py::arg("comm") = py::none(), py::arg("checkpoint") = py::none(), py::arg("inverse_hessian") = py::none(), R"doc(
Train the potential in filename with MTPFit on `ranks` threads, or over the ranks of comm, and return it.

structures : iterable of FitStructure; under comm, this rank's share
options    : the fields of MTPFitOptions without the leading "--", as strings, log (stdout, none
             or a file), which rank 0 writes, and species, the species' names for the error tables,
             comma-separated
ranks      : threads
comm       : an mpi4py communicator, in place of threads. Every rank calls with the same other
             arguments and returns the fitted potential; the fit raises its errors on every rank
checkpoint : called on rank 0 with the potential being trained at each save of the fit; its first
             failure is a warning and ends checkpointing
inverse_hessian : (n, n) float64 array, n the coefficients BFGS trains (the radial ones under
             radial_only, else all); BFGS starts from it and leaves its final inverse Hessian
             in rank 0's. None starts from the identity
)doc");
}
