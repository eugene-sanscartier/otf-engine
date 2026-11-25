/* -*- c++ -*- ----------------------------------------------------------
   Python bindings for ETNFit (see etn_fit.h). The ranks of the fit are
   threads or the ranks of an mpi4py communicator.
------------------------------------------------------------------------- */

#include "bindings_common.h"
#include "etn_fit.h"

// Adds the fields of ETNFitOptions other than the weights.
void add_option_fields(ETNFitOptions& o, OptionFields& fields) {
    fields.doubles.insert({{"tolerance", &o.tolerance}, {"wolfe_c1", &o.wolfe_c1}, {"wolfe_c2", &o.wolfe_c2}, {"init_scale", &o.init_scale}, {"reg_param", &o.reg_param}});
    fields.ints = {{"iteration_limit", &o.iteration_limit}, {"max_func_eval", &o.max_func_eval}, {"bfgs_max_iters_no_improve", &o.bfgs_max_iters_no_improve}, {"random_seed", &o.random_seed}, {"varpro_redraw_every", &o.varpro_redraw_every}};
    fields.bools.insert({{"rescale", &o.rescale}, {"skip_preinit", &o.skip_preinit}, {"init_random", &o.init_random}, {"init_species", &o.init_species}, {"varpro", &o.varpro}, {"reg_at_main", &o.reg_at_main}});
}

static py::object train(const std::string& filename, py::iterable structures, const std::map<std::string, std::string>& options, int ranks, py::object comm, py::object checkpoint) {
    std::string log_target;
    const ETNFitOptions o = parse_options<ETNFitOptions>("fit_etn", options, log_target);
    return train_potential<ETNFit>("fit_etn", filename, structures, o, log_target, ranks, comm, checkpoint);
}

void bind_etn_fit(py::module_& m) {
    m.def("fit_etn", &train, py::arg("filename"), py::arg("structures"), py::arg("options"), py::arg("ranks") = 1, py::arg("comm") = py::none(), py::arg("checkpoint") = py::none(), R"doc(
Train the potential in filename with ETNFit on `ranks` threads or over the ranks of comm, and return it.

Args:
    structures: iterable of FitStructure; under comm, the share of this rank.
    options: the fields of ETNFitOptions without the leading "--", as strings. Also log (stdout,
        none or a file), which rank 0 writes, and species, the comma-separated species names
        for the error tables.
    ranks: number of threads.
    comm: an mpi4py communicator, in place of threads. Every rank calls with the same other
        arguments and returns the fitted potential. The fit raises its errors on every rank.
    checkpoint: called on rank 0 with the potential in training at every step outside a line
        search. Its first failure logs a warning and ends checkpointing.
)doc");
}
