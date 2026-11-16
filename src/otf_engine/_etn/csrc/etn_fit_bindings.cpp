/* -*- c++ -*- ----------------------------------------------------------
   Python bindings for ETNFit (see etn_fit.h), whose ranks are
   threads or those of an mpi4py communicator.
------------------------------------------------------------------------- */

#include "bindings_common.h"
#include "etn_fit.h"

// ETNFitOptions' fields beside the weights.
void add_option_fields(ETNFitOptions& o, OptionFields& fields) {
    fields.doubles.insert({{"tolerance", &o.tolerance}, {"wolfe_c1", &o.wolfe_c1}, {"wolfe_c2", &o.wolfe_c2}});
    fields.ints = {{"iteration_limit", &o.iteration_limit}, {"max_func_eval", &o.max_func_eval}, {"bfgs_max_iters_no_improve", &o.bfgs_max_iters_no_improve}};
    fields.bools.insert({{"rescale", &o.rescale}, {"skip_preinit", &o.skip_preinit}});
}

static py::object train(const std::string& filename, py::iterable structures, const std::map<std::string, std::string>& options, int ranks, py::object comm, py::object checkpoint) {
    std::string log_target;
    const ETNFitOptions o = parse_options<ETNFitOptions>("fit_etn", options, log_target);
    return train_potential<ETNFit>("fit_etn", filename, structures, o, log_target, ranks, comm, checkpoint);
}

void bind_etn_fit(py::module_& m) {
    m.def("fit_etn", &train, py::arg("filename"), py::arg("structures"), py::arg("options"), py::arg("ranks") = 1, py::arg("comm") = py::none(), py::arg("checkpoint") = py::none(), R"doc(
Train the potential in filename with ETNFit on `ranks` threads, or over the ranks of comm, and return it.

structures : iterable of Sample; under comm, this rank's share
options    : the fields of ETNFitOptions without the leading "--", as strings, log (stdout, none
             or a file), which rank 0 writes, and species, the species' names for the error tables,
             comma-separated
ranks      : threads
comm       : an mpi4py communicator, in place of threads. Every rank calls with the same other
             arguments and returns the fitted potential; the fit raises its errors on every rank
checkpoint : called on rank 0 with the potential being trained at every step outside a line
             search; its first failure is a warning and ends checkpointing
)doc");
}
