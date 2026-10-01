/* -*- c++ -*- ----------------------------------------------------------
   Python bindings for MTPTrainer (see mtp_trainer.h). Built into _mtp_ext,
   its ranks are threads (see thread_mpi.h); built with MTP_MPI, this unit is
   the _mtp_mpi module, whose ranks are those of an mpi4py communicator.
------------------------------------------------------------------------- */

#include "bindings_common.h"

#include "mtp_trainer.h"

#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <thread>

static TrainingStructure make_structure(const PyNeighbors& nb, py::object energy, py::object forces, py::object virial) {
    const NeighList& list = nb.view();
    const int n_pairs = std::accumulate(list.numneigh, list.numneigh + list.inum, 0);

    TrainingStructure s;
    s.types.assign(list.types, list.types + list.n_atoms);
    s.ilist.assign(list.ilist, list.ilist + list.inum);
    s.numneigh.assign(list.numneigh, list.numneigh + list.inum);
    s.firstneigh.assign(list.firstneigh, list.firstneigh + n_pairs);
    s.displacements.assign(list.displacements, list.displacements + (size_t) n_pairs * 3);

    if (!energy.is_none()) {
        s.energy = energy.cast<double>();
        s.has_energy = true;
    }
    if (!forces.is_none()) {
        DoubleArray f = forces.cast<DoubleArray>();
        if (f.ndim() != 2 || f.shape(0) != list.n_atoms || f.shape(1) != 3)
            throw std::runtime_error("train_mtp: forces must have shape (n_atoms, 3)");
        s.forces.assign(f.data(), f.data() + f.size());
        s.has_forces = true;
    }
    if (!virial.is_none()) {
        DoubleArray v = virial.cast<DoubleArray>();
        if (v.size() != 6)
            throw std::runtime_error("train_mtp: virial must have 6 components");
        std::copy(v.data(), v.data() + 6, s.virial);
        s.has_virial = true;
    }
    return s;
}

// mlp train's options; log is where the fit log goes.
static TrainerOptions parse_options(const std::map<std::string, std::string>& options, std::string& log) {
    TrainerOptions o;
    const std::map<std::string, double*> doubles = {
        {"energy_weight", &o.energy_weight}, {"force_weight", &o.force_weight}, {"stress_weight", &o.stress_weight},
        {"penalty_weight", &o.penalty_weight}, {"scale_by_force", &o.scale_by_force}, {"select_factor", &o.select_factor},
        {"tolerance", &o.tolerance}};
    const std::map<std::string, int*> ints = {
        {"weight_scaling", &o.weight_scaling}, {"weight_scaling_forces", &o.weight_scaling_forces}, {"iteration_limit", &o.iteration_limit}};
    const std::map<std::string, bool*> bools = {{"no_mindist_update", &o.no_mindist_update}, {"init_random", &o.init_random}, {"skip_preinit", &o.skip_preinit}};

    for (const auto& [key, value] : options) {
        if (key == "log")
            log = value;
        else if (doubles.count(key))
            *doubles.at(key) = std::stod(value);
        else if (ints.count(key))
            *ints.at(key) = std::stoi(value);
        else if (bools.count(key)) {
            if (value == "true" || value == "True" || value == "TRUE")
                *bools.at(key) = true;
            else if (value == "false" || value == "False" || value == "FALSE")
                *bools.at(key) = false;
            else
                throw std::runtime_error("train_mtp: " + key + "=" + value + " is not true or false");
        } else
            throw std::runtime_error("train_mtp: unknown option '" + key + "'");
    }
    return o;
}

// The stream the log option names: stdout (or empty), none, or a file, opened into file.
static std::ostream* open_log(const std::string& target, std::ofstream& file) {
    if (target == "stdout" || target.empty()) return &std::cout;
    if (target == "none") return nullptr;
    file.open(target);
    if (!file)
        throw std::runtime_error("train_mtp: cannot open log '" + target + "'");
    return &file;
}

#ifdef MTP_MPI

// Raises error on every rank of comm, or else the error of the lowest rank that has one.
static void raise_together(MPI_Comm comm, const std::string& error) {
    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    const int mine = error.empty() ? size : rank;
    int failed = size;
    MPI_Allreduce(&mine, &failed, 1, MPI_INT, MPI_MIN, comm);
    if (failed == size) return;

    std::string message = rank == failed ? error : std::string();
    int length = (int) message.size();
    MPI_Bcast(&length, 1, MPI_INT, failed, comm);
    message.resize(length);
    MPI_Bcast(message.data(), length, MPI_CHAR, failed, comm);
    throw std::runtime_error(message);
}

static py::object train(const std::string& filename, py::iterable structures, const std::map<std::string, std::string>& options, py::object comm_object, py::object checkpoint) {
    const MPI_Comm comm = MPI_Comm_f2c(comm_object.attr("py2f")().cast<MPI_Fint>());
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    // A rank failing to set up fails every rank, before the trainer's first collective.
    std::string error;
    TrainerOptions trainer_options;
    std::vector<TrainingStructure> share;
    std::ofstream log_file;
    std::ostream* log = nullptr;
    py::object potential;
    try {
        std::string log_target = "stdout";
        trainer_options = parse_options(options, log_target);
        for (py::handle item : structures) {
            py::tuple t = py::reinterpret_borrow<py::tuple>(item);
            share.push_back(make_structure(t[0].cast<const PyNeighbors&>(), t[1], t[2], t[3]));
        }
        if (rank == 0) log = open_log(log_target, log_file);
        potential = py::cast(std::make_unique<MTPTraining>(filename));
    } catch (const std::exception& e) {
        error = e.what();
    }
    raise_together(comm, error);

    MTPTraining& potential_ref = potential.cast<MTPTraining&>();
    std::string checkpoint_error;    // the first failed checkpoint, which ends checkpointing
    std::function<void()> save;
    if (rank == 0 && !checkpoint.is_none())
        save = [&]() {
            if (!checkpoint_error.empty()) return;
            py::gil_scoped_acquire gil;
            try {
                checkpoint(potential);
            } catch (const std::exception& e) {
                checkpoint_error = std::string("train_mtp: checkpoint failed: ") + e.what();
            }
        };

    {
        py::gil_scoped_release unlocked;
        try {
            MTPTrainer trainer(potential_ref, std::move(share), trainer_options, comm, log, save);
            trainer.train();
        } catch (const std::runtime_error& e) {
            error = e.what();    // MTPTrainer raises its errors on every rank together
        } catch (...) {
            MPI_Abort(comm, 1);    // the other ranks are left waiting in a collective
        }
    }
    raise_together(comm, error);
    raise_together(comm, checkpoint_error);
    return potential;
}

PYBIND11_MODULE(_mtp_mpi, m) {
    m.doc() = "MTPTrainer over MPI.";
    py::module_::import("otf_engine._mtp._mtp_ext");    // registers MTPTraining, NeighList, Equations and MaxVol
    bind_maxvol(m);

    m.def("train_mtp", &train, py::arg("filename"), py::arg("structures"), py::arg("options"), py::arg("comm"), py::arg("checkpoint") = py::none(), R"doc(
Train the potential in filename as mlip-3's `mlp train` does, over the ranks of comm, and return it.

Every rank calls it with its own share of the structures and the same other arguments, and returns
the fitted potential. An error on any rank is raised on every rank.

structures : this rank's share, an iterable of (NeighList, energy|None, forces(n_atoms,3)|None,
             virial(6)|None), neighbor lists at the potential's cutoff, virial as mlip-3's
             PlusStress in eV, xx,yy,zz,xy,xz,yz
options    : mlp train options without the leading "--", as strings; rank 0 writes the log
comm       : an mpi4py communicator
checkpoint : called on rank 0 with the potential being trained wherever mlp saves it during the
             fit; a failure ends checkpointing and is raised after the fit
)doc");
}

#else

static py::object train(const std::string& filename, py::iterable structures, const std::map<std::string, std::string>& options, int ranks, py::object checkpoint) {
    std::string log_target = "stdout";
    const TrainerOptions trainer_options = parse_options(options, log_target);
    if (ranks < 1)
        throw std::runtime_error("train_mtp: ranks must be at least 1");

    // Dealt round-robin, as mlp's MPI_LoadCfgs deals a file
    std::vector<std::vector<TrainingStructure>> shares(ranks);
    int count = 0;
    for (py::handle item : structures) {
        py::tuple t = py::reinterpret_borrow<py::tuple>(item);
        shares[count++ % ranks].push_back(make_structure(t[0].cast<const PyNeighbors&>(), t[1], t[2], t[3]));
    }
    if (count == 0)
        throw std::runtime_error("train_mtp: no training structures");

    std::ofstream log_file;
    std::ostream* log = open_log(log_target, log_file);

    // Rank 0 trains the potential that is returned, and hands it to checkpoint.
    py::object potential = py::cast(std::make_unique<MTPTraining>(filename));
    MTPTraining& potential_ref = potential.cast<MTPTraining&>();
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
            std::unique_ptr<MTPTraining> own;
            if (r != 0) own = std::make_unique<MTPTraining>(filename);
            MTPTraining& pot = r == 0 ? potential_ref : *own;
            MTPTrainer trainer(pot, std::move(shares[r]), trainer_options, &comms[r], r == 0 ? log : nullptr, r == 0 ? save : nullptr);
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

void bind_trainer(py::module_& m) {
    m.def("train_mtp", &train, py::arg("filename"), py::arg("structures"), py::arg("options"), py::arg("ranks"), py::arg("checkpoint") = py::none(), R"doc(
Train the potential in filename as mlip-3's `mlp train` does, on `ranks` threads, and return it.

structures : iterable of (NeighList, energy|None, forces(n_atoms,3)|None, virial(6)|None),
             neighbor lists at the potential's cutoff, virial as mlip-3's PlusStress in eV,
             xx,yy,zz,xy,xz,yz
options    : mlp train options without the leading "--", as strings
checkpoint : called with the potential being trained wherever mlp saves it during the fit
)doc");
}

#endif
