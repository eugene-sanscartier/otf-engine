/* -*- c++ -*- ----------------------------------------------------------
   Python bindings for ETNTrainer (see etn_trainer.h). Built into _etn_ext,
   its ranks are threads (see thread_mpi.h); built with MTP_MPI, this unit is
   the _etn_mpi module, whose ranks are those of an mpi4py communicator.
------------------------------------------------------------------------- */

#include "bindings_common.h"
#include "etn_trainer.h"

// mlip-4's training settings beside the weights, under mlp's names where mlp has the option.
void add_option_fields(ETNTrainerOptions& o, OptionFields& fields) {
    fields.doubles.insert({{"tolerance", &o.tolerance}, {"wolfe_c1", &o.wolfe_c1}, {"wolfe_c2", &o.wolfe_c2}});
    fields.ints = {{"iteration_limit", &o.iteration_limit}, {"max_func_eval", &o.max_func_eval}, {"bfgs_max_iters_no_improve", &o.bfgs_max_iters_no_improve}};
    fields.bools = {{"rescale", &o.rescale}, {"skip_preinit", &o.skip_preinit}};
}

#ifdef MTP_MPI

static py::object train(const std::string& filename, py::iterable structures, const std::map<std::string, std::string>& options, py::object comm_object, py::object checkpoint, bool balance_species) {
    auto make_options = [&](std::string& log_target) {
        ETNTrainerOptions o = parse_options<ETNTrainerOptions>("train_etn", options, log_target);
        o.balance_species = balance_species;
        return o;
    };
    return train_over_mpi<ETNTrainer>("train_etn", filename, structures, make_options, comm_object, checkpoint);
}

PYBIND11_MODULE(_etn_mpi, m) {
    m.doc() = "ETNTrainer over MPI.";
    py::module_::import("otf_engine._etn._etn_ext");    // registers ETNTraining

    m.def("train_etn", &train, py::arg("filename"), py::arg("structures"), py::arg("options"), py::arg("comm"), py::arg("checkpoint") = py::none(), py::arg("balance_species") = true, R"doc(
Train the potential in filename as mlip-4's Train does, over the ranks of comm, and return it.

Every rank calls it with its own share of the structures and the same other arguments, and returns
the fitted potential. An error on any rank is raised on every rank.

structures : this rank's share, as _etn_ext.train_etn's
options    : as _etn_ext.train_etn's; rank 0 writes the log
comm       : an mpi4py communicator
checkpoint : called on rank 0 with the potential being trained at every step outside a line
             search; a failure ends checkpointing and is raised after the fit
balance_species : give every species of a structure an equal share of its force weight; False
             weighs every force alike, as mlip-4 does
)doc");
}

#else

static py::object train(const std::string& filename, py::iterable structures, const std::map<std::string, std::string>& options, int ranks, py::object checkpoint, bool balance_species) {
    std::string log_target = "stdout";
    ETNTrainerOptions o = parse_options<ETNTrainerOptions>("train_etn", options, log_target);
    o.balance_species = balance_species;
    return train_on_threads<ETNTrainer>("train_etn", filename, structures, o, log_target, ranks, checkpoint);
}

void bind_etn_trainer(py::module_& m) {
    m.def("train_etn", &train, py::arg("filename"), py::arg("structures"), py::arg("options"), py::arg("ranks") = 1, py::arg("checkpoint") = py::none(), py::arg("balance_species") = true, R"doc(
Train the potential in filename as mlip-4's Train does, on `ranks` threads, and return it.
structures : iterable of (NeighList, energy|None, forces(n_atoms,3)|None, virial(6)|None, number),
             neighbor lists at the potential's cutoff, virial in eV, xx,yy,zz,xy,xz,yz, and the
             number the error tables name it by
options    : mlip-4's training settings as strings, under mlp's names where mlp has the option
             (energy_weight, force_weight, stress_weight, iteration_limit, max_func_eval, tolerance,
             rescale, skip_preinit, wolfe_c1, wolfe_c2, bfgs_max_iters_no_improve), log (stdout,
             none or a file), and species, the species' names for the error tables, comma-separated
checkpoint : called with the potential being trained at every step outside a line search
balance_species : give every species of a structure an equal share of its force weight; False
             weighs every force alike, as mlip-4 does
)doc");
}

#endif
