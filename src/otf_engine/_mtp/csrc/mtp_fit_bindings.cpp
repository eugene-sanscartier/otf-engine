/* -*- c++ -*- ----------------------------------------------------------
   Python bindings for MTPFit (see mtp_fit.h). Its ranks are threads or
   the ranks of an mpi4py communicator.
------------------------------------------------------------------------- */

#include "bindings_common.h"

#include "mtp_fit.h"

// Registers the fields of MTPFitOptions other than the weights.
void add_option_fields(MTPFitOptions& o, OptionFields& fields) {
    fields.doubles.insert({{"penalty_weight", &o.penalty_weight}, {"scale_by_force", &o.scale_by_force}, {"select_factor", &o.select_factor},
                           {"tolerance", &o.tolerance}, {"reg_param", &o.reg_param}, {"init_noise", &o.init_noise}});
    fields.ints = {{"weight_scaling", &o.weight_scaling}, {"weight_scaling_forces", &o.weight_scaling_forces}, {"iteration_limit", &o.iteration_limit}, {"random_seed", &o.random_seed}, {"lbfgs_memory", &o.lbfgs_memory}};
    fields.bools.insert({{"init_random", &o.init_random}, {"skip_preinit", &o.skip_preinit}, {"radial_only", &o.radial_only}, {"bfgs_only", &o.bfgs_only}, {"reg_once", &o.reg_once}, {"reg_at_main", &o.reg_at_main}, {"varpro", &o.varpro}, {"no_orthogonalize", &o.no_orthogonalize}, {"riemannian", &o.riemannian}, {"ldlt", &o.ldlt}, {"reg_follows_scaling", &o.reg_follows_scaling}});
}

// Returns the buffer of inverse_hessian, a writable C-contiguous (n, n) float64 array, and sets n. None gives nullptr and n = 0.
static double* inverse_hessian_buffer(py::object inverse_hessian, int& n) {
    n = 0;
    if (inverse_hessian.is_none()) return nullptr;
    auto array = inverse_hessian.cast<py::array_t<double, py::array::c_style>>();
    if (!array.is(inverse_hessian))    // a conversion made a copy
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

Args:
    structures: iterable of FitStructure; under comm, the share of this rank.
    options: the fields of MTPFitOptions without the leading "--", as strings. log (stdout, none
        or a file) sets where rank 0 writes the log. species gives the comma-separated species
        names for the error tables.
    ranks: number of threads.
    comm: an mpi4py communicator, in place of threads. All ranks call with the same other
        arguments and return the fitted potential. The fit raises its errors on all ranks.
    checkpoint: called on rank 0 with the potential at each save of the fit. Its first failure
        logs a warning and ends checkpointing.
    inverse_hessian: (n, n) float64 array over the n coefficients BFGS trains, the radial ones
        under radial_only or varpro, else all. BFGS starts from it and writes its final inverse
        Hessian into the array of rank 0. None starts from the identity.
)doc");
}
