/* -*- c++ -*- ----------------------------------------------------------
   Python bindings for ETNTrainer (see etn_trainer.h). Built into _etn_ext,
   its ranks are threads (see thread_mpi.h); built with MTP_MPI, this unit is
   the _etn_mpi module, whose ranks are those of an mpi4py communicator.
------------------------------------------------------------------------- */

#include "bindings_common.h"
#include "etn_trainer.h"

#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>

// mlip-4's training settings; log is where the fit log goes.
static ETNTrainerOptions parse_options(const std::map<std::string, std::string>& options, std::string& log) {
    ETNTrainerOptions o;
    const std::map<std::string, double*> doubles = {
        {"energy_weight", &o.energy_weight}, {"force_weight", &o.force_weight}, {"stress_weight", &o.stress_weight},
        {"linstop", &o.linstop}, {"wolfe_c1", &o.wolfe_c1}, {"wolfe_c2", &o.wolfe_c2}};
    const std::map<std::string, int*> ints = {
        {"max_iter_count", &o.max_iter_count}, {"max_func_eval", &o.max_func_eval}, {"bfgs_max_iters_no_improve", &o.bfgs_max_iters_no_improve}};
    const std::map<std::string, bool*> bools = {{"rescale", &o.rescale}, {"pretrain", &o.pretrain}};

    for (const auto& [key, value] : options) {
        if (key == "log")
            log = value;
        else if (key == "species") {
            std::stringstream names(value);
            for (std::string name; std::getline(names, name, ',');)
                o.species.push_back(name);
        } else if (doubles.count(key))
            *doubles.at(key) = std::stod(value);
        else if (ints.count(key))
            *ints.at(key) = std::stoi(value);
        else if (bools.count(key)) {
            if (value == "true" || value == "True" || value == "TRUE")
                *bools.at(key) = true;
            else if (value == "false" || value == "False" || value == "FALSE")
                *bools.at(key) = false;
            else
                throw std::runtime_error("train_etn: " + key + "=" + value + " is not true or false");
        } else
            throw std::runtime_error("train_etn: unknown option '" + key + "'");
    }
    return o;
}

// Raises unless every atom of share has one of the potential's species.
static void check_share_species(const ETNTraining& potential, const std::vector<TrainingStructure>& share) {
    for (const TrainingStructure& s : share)
        for (int t : s.types)
            if (t < 0 || t >= potential.get_species_count())
                throw std::runtime_error("train_etn: a training structure has species " + std::to_string(t) + ", outside the potential's " + std::to_string(potential.get_species_count()));
}

#ifdef MTP_MPI

static py::object train(const std::string& filename, py::iterable structures, const std::map<std::string, std::string>& options, py::object comm_object, py::object checkpoint, bool balance_species) {
    const MPI_Comm comm = MPI_Comm_f2c(comm_object.attr("py2f")().cast<MPI_Fint>());
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    // A rank failing to set up fails every rank, before the trainer's first collective.
    std::string error;
    ETNTrainerOptions trainer_options;
    std::vector<TrainingStructure> share;
    std::ofstream log_file;
    std::ostream* log = nullptr;
    py::object potential;
    try {
        std::string log_target = "stdout";
        trainer_options = parse_options(options, log_target);
        trainer_options.balance_species = balance_species;
        for (py::handle item : structures)
            share.push_back(make_structure(item));
        if (rank == 0) log = open_log(log_target, log_file);
        potential = py::cast(std::make_unique<ETNTraining>(filename));
        check_share_species(potential.cast<ETNTraining&>(), share);
    } catch (const std::exception& e) {
        error = e.what();
    }
    raise_together(comm, error);

    ETNTraining& potential_ref = potential.cast<ETNTraining&>();
    std::string checkpoint_error;    // the first failed checkpoint, which ends checkpointing
    std::function<void()> save;
    if (rank == 0 && !checkpoint.is_none())
        save = [&]() {
            if (!checkpoint_error.empty()) return;
            py::gil_scoped_acquire gil;
            try {
                checkpoint(potential);
            } catch (const std::exception& e) {
                checkpoint_error = std::string("train_etn: checkpoint failed: ") + e.what();
            }
        };

    {
        py::gil_scoped_release unlocked;
        try {
            ETNTrainer trainer(potential_ref, std::move(share), trainer_options, comm, log, save);
            trainer.train();
        } catch (const std::runtime_error& e) {
            error = e.what();    // ETNTrainer raises its errors on every rank together
        } catch (...) {
            MPI_Abort(comm, 1);    // the other ranks are left waiting in a collective
        }
    }
    raise_together(comm, error);
    raise_together(comm, checkpoint_error);
    return potential;
}

PYBIND11_MODULE(_etn_mpi, m) {
    m.doc() = "ETNTrainer over MPI.";
    py::module_::import("otf_engine._etn._etn_ext");    // registers ETNTraining, and through _mtp_ext NeighList

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
    ETNTrainerOptions trainer_options = parse_options(options, log_target);
    trainer_options.balance_species = balance_species;
    if (ranks < 1)
        throw std::runtime_error("train_etn: ranks must be at least 1");

    // Dealt round-robin, as MTPTrainer's are
    std::vector<std::vector<TrainingStructure>> shares(ranks);
    int count = 0;
    for (py::handle item : structures)
        shares[count++ % ranks].push_back(make_structure(item));
    if (count == 0)
        throw std::runtime_error("train_etn: no training structures");

    std::ofstream log_file;
    std::ostream* log = open_log(log_target, log_file);

    // Rank 0 trains the potential that is returned, and hands it to checkpoint.
    py::object potential = py::cast(std::make_unique<ETNTraining>(filename));
    ETNTraining& potential_ref = potential.cast<ETNTraining&>();
    for (const auto& share : shares)
        check_share_species(potential_ref, share);
    std::function<void()> save;
    if (!checkpoint.is_none())
        save = [&]() {
            py::gil_scoped_acquire gil;
            checkpoint(potential);
        };

    ThreadGroup group(ranks);
    std::vector<ThreadComm> comms(ranks);
    for (int r = 0; r < ranks; r++)
        comms[r] = ThreadComm{&group, r};
    std::exception_ptr failure;
    std::mutex failure_mutex;
    auto run_rank = [&](int r) {
        try {
            std::unique_ptr<ETNTraining> own;
            if (r != 0) own = std::make_unique<ETNTraining>(filename);
            ETNTraining& pot = r == 0 ? potential_ref : *own;
            ETNTrainer trainer(pot, std::move(shares[r]), trainer_options, &comms[r], r == 0 ? log : nullptr, r == 0 ? save : nullptr);
            trainer.train();
        } catch (const CommAborted&) {
        } catch (...) {
            std::lock_guard<std::mutex> lock(failure_mutex);
            if (!failure) failure = std::current_exception();
            group.abort();
        }
    };

    {
        py::gil_scoped_release unlocked;
        std::vector<std::thread> threads;
        for (int r = 1; r < ranks; r++)
            threads.emplace_back(run_rank, r);
        run_rank(0);
        for (std::thread& t : threads)
            t.join();
    }
    if (failure) std::rethrow_exception(failure);
    return potential;
}

void bind_etn_trainer(py::module_& m) {
    m.def("train_etn", &train, py::arg("filename"), py::arg("structures"), py::arg("options"), py::arg("ranks") = 1, py::arg("checkpoint") = py::none(), py::arg("balance_species") = true, R"doc(
Train the potential in filename as mlip-4's Train does, on `ranks` threads, and return it.
structures : iterable of (NeighList, energy|None, forces(n_atoms,3)|None, virial(6)|None, number),
             neighbor lists at the potential's cutoff, virial in eV, xx,yy,zz,xy,xz,yz, and the
             number the error tables name it by
options    : mlip-4's training settings as strings (energy_weight, force_weight, stress_weight,
             max_iter_count, max_func_eval, linstop, rescale, pretrain, wolfe_c1, wolfe_c2,
             bfgs_max_iters_no_improve), log (stdout, none or a file), and species, the species'
             names for the error tables, comma-separated
checkpoint : called with the potential being trained at every step outside a line search
balance_species : give every species of a structure an equal share of its force weight; False
             weighs every force alike, as mlip-4 does
)doc");
    m.def("error_table", [](PairETN& potential, py::iterable structures, const std::vector<std::string>& species, const std::string& title) {
        std::vector<TrainingStructure> share;
        for (py::handle item : structures)
            share.push_back(make_structure(item));
        ThreadGroup group(1);
        ThreadComm comm{&group, 0};
        py::gil_scoped_release unlocked;
        return error_table(potential, share, species, &comm, title);
    }, py::arg("potential"), py::arg("structures"), py::arg("species"), py::arg("title"), R"doc(
The errors of potential on the structures, as the table train_etn logs, under title.
structures : as train_etn's
species    : the species' names
)doc");
}

#endif
