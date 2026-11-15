/* -*- c++ -*- ----------------------------------------------------------
   Python bindings for ETNTrainer (see etn_trainer.h). Its ranks are threads
   (see thread_mpi.h).
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

static py::object train(const std::string& filename, py::iterable structures, const std::map<std::string, std::string>& options, int ranks) {
    std::string log_target = "stdout";
    ETNTrainerOptions trainer_options = parse_options(options, log_target);
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

    // Rank 0 trains the potential that is returned.
    py::object potential = py::cast(std::make_unique<ETNTraining>(filename));
    ETNTraining& potential_ref = potential.cast<ETNTraining&>();
    for (const auto& share : shares)
        for (const TrainingStructure& s : share)
            for (int t : s.types)
                if (t < 0 || t >= potential_ref.get_species_count())
                    throw std::runtime_error("train_etn: a training structure has species " + std::to_string(t) + ", outside the potential's " + std::to_string(potential_ref.get_species_count()));

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
            ETNTrainer trainer(pot, std::move(shares[r]), trainer_options, &comms[r], r == 0 ? log : nullptr);
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
    m.def("train_etn", &train, py::arg("filename"), py::arg("structures"), py::arg("options"), py::arg("ranks") = 1, R"doc(
Train the potential in filename as mlip-4's Train does, on `ranks` threads, and return it.
structures : iterable of (NeighList, energy|None, forces(n_atoms,3)|None, virial(6)|None, number),
             neighbor lists at the potential's cutoff, virial in eV, xx,yy,zz,xy,xz,yz, and the
             number the error tables name it by
options    : mlip-4's training settings as strings (energy_weight, force_weight, stress_weight,
             max_iter_count, max_func_eval, linstop, rescale, pretrain, wolfe_c1, wolfe_c2,
             bfgs_max_iters_no_improve), log (stdout, none or a file), and species, the species'
             names for the error tables, comma-separated
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
