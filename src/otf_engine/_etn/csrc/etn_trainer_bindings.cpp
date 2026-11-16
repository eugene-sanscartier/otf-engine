/* -*- c++ -*- ----------------------------------------------------------
   Python bindings for ETNTrainer (see etn_trainer.h), whose ranks are
   threads or those of an mpi4py communicator.
------------------------------------------------------------------------- */

#include "bindings_common.h"
#include "etn_trainer.h"

// ETNTrainerOptions' fields beside the weights.
void add_option_fields(ETNTrainerOptions& o, OptionFields& fields) {
    fields.doubles.insert({{"tolerance", &o.tolerance}, {"wolfe_c1", &o.wolfe_c1}, {"wolfe_c2", &o.wolfe_c2}});
    fields.ints = {{"iteration_limit", &o.iteration_limit}, {"max_func_eval", &o.max_func_eval}, {"bfgs_max_iters_no_improve", &o.bfgs_max_iters_no_improve}};
    fields.bools = {{"rescale", &o.rescale}, {"skip_preinit", &o.skip_preinit}};
}

static py::object train(const std::string& filename, py::iterable structures, const std::map<std::string, std::string>& options, int ranks, py::object comm, py::object checkpoint, bool balance_species) {
    auto make_options = [&](std::string& log_target) {
        ETNTrainerOptions o = parse_options<ETNTrainerOptions>("train_etn", options, log_target);
        o.balance_species = balance_species;
        return o;
    };
    return train_potential<ETNTrainer>("train_etn", filename, structures, make_options, ranks, comm, checkpoint);
}

void bind_etn_trainer(py::module_& m) {
    m.def("train_etn", &train, py::arg("filename"), py::arg("structures"), py::arg("options"), py::arg("ranks") = 1, py::arg("comm") = py::none(), py::arg("checkpoint") = py::none(), py::arg("balance_species") = true, R"doc(
Train the potential in filename with ETNTrainer on `ranks` threads, or over the ranks of comm, and return it.

structures : iterable of Sample; under comm, this rank's share
options    : the fields of ETNTrainerOptions without the leading "--", as strings, log (stdout, none
             or a file), which rank 0 writes, and species, the species' names for the error tables,
             comma-separated
ranks      : threads
comm       : an mpi4py communicator, in place of threads. Every rank calls with the same other
             arguments and returns the fitted potential; an error on any rank is raised on every rank
checkpoint : called on rank 0 with the potential being trained at every step outside a line
             search; a failure ends checkpointing and is raised after the fit
balance_species : give every species of a structure an equal share of its force weight
)doc");
}
