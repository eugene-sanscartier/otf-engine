/* -*- c++ -*- ----------------------------------------------------------
   Shared pieces of the pybind11 layer.

   One binding unit per source pair, bound in the same order. In the
   _core_ext module:
     core_bindings.cpp                    <- neigh_list.h, pair.{h,cpp}, error_table.h
     maxvol_bindings.cpp                  <- maxvol.{h,cpp}
   in the _mtp_ext module under ../../_mtp/csrc/:
     pair_mtp_bindings.cpp                <- pair_mtp.{h,cpp}
     pair_mtp_extrapolation_bindings.cpp  <- pair_mtp_extrapolation.{h,cpp}
     mtp_training_bindings.cpp            <- mtp_training.{h,cpp}
     mtp_trainer_bindings.cpp             <- mtp_trainer.{h,cpp}
   and in the _etn_ext module under ../../_etn/csrc/:
     etn_harmonics_bindings.cpp           <- etn_harmonics.{h,cpp}
     pair_etn_bindings.cpp                <- pair_etn.{h,cpp}
     etn_training_bindings.cpp            <- etn_training.{h,cpp}
     etn_trainer_bindings.cpp             <- etn_trainer.{h,cpp}
   Built with MTP_MPI, core_bindings.cpp and maxvol_bindings.cpp are the
   _core_mpi module, mtp_trainer_bindings.cpp the _mtp_mpi module, and
   etn_trainer_bindings.cpp the _etn_mpi module.
------------------------------------------------------------------------- */

#pragma once

#include "pair.h"
#include "training_structure.h"

#ifdef MTP_MPI
#include <mpi.h>
#else
#include "thread_mpi.h"
#endif

#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

using IntArray = py::array_t<int, py::array::c_style | py::array::forcecast>;
using DoubleArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

// Owns the numpy arrays a NeighList points into.
class PyNeighbors {
  public:
    PyNeighbors(IntArray types, IntArray ilist, IntArray numneigh, IntArray firstneigh, DoubleArray displacements)
        : types_(types), ilist_(ilist), numneigh_(numneigh), firstneigh_(firstneigh), displacements_(displacements) {
        if (ilist_.shape(0) != numneigh_.shape(0))
            throw std::runtime_error("NeighList: ilist and numneigh must have the same length");
        if (displacements_.ndim() != 2 || displacements_.shape(1) != 3)
            throw std::runtime_error("NeighList: displacements must have shape (n_pairs, 3)");
        if (firstneigh_.shape(0) != displacements_.shape(0))
            throw std::runtime_error("NeighList: firstneigh and displacements must have the same length");

        list_.n_atoms = (int) types_.shape(0);
        list_.types = types_.data();
        list_.inum = (int) ilist_.shape(0);
        list_.ilist = ilist_.data();
        list_.numneigh = numneigh_.data();
        list_.firstneigh = firstneigh_.data();
        list_.displacements = displacements_.data();
    }

    const NeighList& view() const { return list_; }
    int n_atoms() const { return list_.n_atoms; }
    int inum() const { return list_.inum; }
    IntArray types() const { return types_; }
    DoubleArray displacements() const { return displacements_; }

  private:
    IntArray types_, ilist_, numneigh_, firstneigh_;
    DoubleArray displacements_;
    NeighList list_;
};

// A copy of values as a numpy array of the given shape, by default flat.
template <class T>
py::array_t<T> to_array(const std::vector<T>& values, std::vector<py::ssize_t> shape = {}) {
    if (shape.empty()) shape = {(py::ssize_t) values.size()};
    return py::array_t<T>(shape, values.data());
}

inline DoubleArray zeros(std::vector<py::ssize_t> shape) {
    DoubleArray a(shape);
    std::fill(a.mutable_data(), a.mutable_data() + a.size(), 0.0);
    return a;
}

// The results Pair::compute leaves in the potential, copied out:
// "energy", "forces" (n_atoms, 3), "virials" (6) and "eatom" (n_atoms).
inline py::dict potential_results(const Pair& self, int n_atoms) {
    py::dict result;
    result["energy"] = self.get_energy();
    result["forces"] = to_array(self.get_forces(), {n_atoms, 3});
    result["virials"] = DoubleArray({6}, self.get_virial().data());
    result["eatom"] = to_array(self.get_eatom());
    return result;
}

// The stream a trainer's log option names: stdout (or empty), none, or a file, opened into file.
inline std::ostream* open_log(const std::string& target, std::ofstream& file) {
    if (target == "stdout" || target.empty()) return &std::cout;
    if (target == "none") return nullptr;
    file.open(target);
    if (!file)
        throw std::runtime_error("cannot open log '" + target + "'");
    return &file;
}

#ifdef MTP_MPI
// Raises error on every rank of comm, or else the error of the lowest rank that has one.
inline void raise_together(MPI_Comm comm, const std::string& error) {
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
#endif

// Raises unless every atom of nb has one of the potential's species, which the
// kernels index their coefficients with.
inline void check_species(const Pair& self, const PyNeighbors& nb) {
    const NeighList& list = nb.view();
    for (int i = 0; i < list.n_atoms; i++)
        if (list.types[i] < 0 || list.types[i] >= self.get_species_count())
            throw std::runtime_error("NeighList: atom " + std::to_string(i) + " has species " + std::to_string(list.types[i]) + ", outside the potential's " + std::to_string(self.get_species_count()) + " species");
}

// Binds fn, which takes a NeighList, checking the species first.
// `extra` carries the py::arg entries for any trailing parameters.
template <class Cls, class Self, class Ret, class... Args, class... Extra>
void def_neighbors(Cls& cls, const char* name, Ret (*fn)(Self&, const PyNeighbors&, Args...), const char* doc, Extra... extra) {
    cls.def(
        name,
        [fn](Self& self, const PyNeighbors& nb, Args... args) {
            check_species(self, nb);
            return fn(self, nb, args...);
        },
        py::arg("neighbors"), extra..., doc);
}

// d(E_i, F, virial)/dc of the structure nb, as eval(site_energy_grad, force_grad, virial_grad) fills them for
// a potential of width coefficients, the force and virial arrays null unless asked for: site_energy_grad (inum, width)
// alone, or with forces the tuple (site_energy_grad, force_grad (n_atoms, 3, width), virial_grad (6, width)|None).
inline py::object grad_arrays(const PyNeighbors& nb, int width, bool forces, bool virial, const std::function<void(double*, double*, double*)>& eval) {
    if (virial && !forces)
        throw std::runtime_error("eval_grad: the virial gradient comes with the force gradient");
    auto eg = zeros({nb.inum(), width});
    DoubleArray fg, vg;
    if (forces) fg = zeros({nb.n_atoms(), 3, width});
    if (virial) vg = zeros({6, width});
    double* eg_ptr = eg.mutable_data();
    double* fg_ptr = forces ? fg.mutable_data() : nullptr;
    double* vg_ptr = virial ? vg.mutable_data() : nullptr;
    {
        py::gil_scoped_release unlocked;
        eval(eg_ptr, fg_ptr, vg_ptr);
    }
    if (!forces) return eg;
    return py::make_tuple(eg, fg, virial ? py::object(vg) : py::none());
}

// Pointers into a trainer's options, by the names they take as numbers or flags.
struct OptionFields {
    std::map<std::string, double*> doubles;
    std::map<std::string, int*> ints;
    std::map<std::string, bool*> bools;
};

// A trainer's options from strings, as name (train_mtp, train_etn) takes them: log, the log target, into log;
// species, the species' names, comma-separated; the weights; and the fields add_option_fields(o, fields) adds.
template <class Options>
Options parse_options(const char* name, const std::map<std::string, std::string>& options, std::string& log) {
    Options o;
    OptionFields fields{{{"energy_weight", &o.energy_weight}, {"force_weight", &o.force_weight}, {"stress_weight", &o.stress_weight}}, {}, {}};
    add_option_fields(o, fields);

    for (const auto& [key, value] : options) {
        if (key == "log")
            log = value;
        else if (key == "species") {
            std::stringstream names(value);
            for (std::string species_name; std::getline(names, species_name, ',');)
                o.species.push_back(species_name);
        } else if (fields.doubles.count(key))
            *fields.doubles.at(key) = std::stod(value);
        else if (fields.ints.count(key))
            *fields.ints.at(key) = std::stoi(value);
        else if (fields.bools.count(key)) {
            if (value == "true" || value == "True" || value == "TRUE")
                *fields.bools.at(key) = true;
            else if (value == "false" || value == "False" || value == "FALSE")
                *fields.bools.at(key) = false;
            else
                throw std::runtime_error(std::string(name) + ": " + key + "=" + value + " is not true or false");
        } else
            throw std::runtime_error(std::string(name) + ": unknown option '" + key + "'");
    }
    return o;
}

#ifdef MTP_MPI

// Trains the potential in filename with TrainerT over the ranks of comm_object, each passing its share of the
// structures, and returns it on every rank. make_options gives the options and sets the log target, which rank 0
// writes. An error on any rank is raised on every rank, and an exception other than the trainer's errors ends the
// job with MPI_Abort. name (train_mtp, train_etn) prefixes the errors.
template <class TrainerT>
py::object train_over_mpi(const char* name, const std::string& filename, py::iterable structures, const std::function<typename TrainerT::Options(std::string&)>& make_options, py::object comm_object, py::object checkpoint) {
    using Training = typename TrainerT::Training;
    const MPI_Comm comm = MPI_Comm_f2c(comm_object.attr("py2f")().cast<MPI_Fint>());
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    // A rank failing to set up fails every rank, before the trainer's first collective.
    std::string error;
    typename TrainerT::Options trainer_options;
    std::vector<TrainingStructure> share;
    std::ofstream log_file;
    std::ostream* log = nullptr;
    py::object potential;
    try {
        std::string log_target = "stdout";
        trainer_options = make_options(log_target);
        for (py::handle item : structures)
            share.push_back(item.cast<TrainingStructure>());
        if (rank == 0) log = open_log(log_target, log_file);
        potential = py::cast(std::make_unique<Training>(filename));
    } catch (const std::exception& e) {
        error = e.what();
    }
    raise_together(comm, error);

    Training& potential_ref = potential.cast<Training&>();
    std::string checkpoint_error;    // the first failed checkpoint, which ends checkpointing
    std::function<void()> save;
    if (rank == 0 && !checkpoint.is_none())
        save = [&]() {
            if (!checkpoint_error.empty()) return;
            py::gil_scoped_acquire gil;
            try {
                checkpoint(potential);
            } catch (const std::exception& e) {
                checkpoint_error = std::string(name) + ": checkpoint failed: " + e.what();
            }
        };

    {
        py::gil_scoped_release unlocked;
        try {
            TrainerT trainer(potential_ref, std::move(share), trainer_options, comm, log, save);
            trainer.train();
        } catch (const std::runtime_error& e) {
            error = e.what();    // a trainer raises its errors on every rank together
        } catch (...) {
            MPI_Abort(comm, 1);    // the other ranks are left waiting in a collective
        }
    }
    raise_together(comm, error);
    raise_together(comm, checkpoint_error);
    return potential;
}

#else

// Trains the potential in filename with TrainerT on ranks threads, the structures dealt round-robin as mlp's
// MPI_LoadCfgs deals a file, and returns it. Rank 0 trains the potential returned and hands it to checkpoint.
// name (train_mtp, train_etn) prefixes the errors.
template <class TrainerT>
py::object train_on_threads(const char* name, const std::string& filename, py::iterable structures, const typename TrainerT::Options& options, const std::string& log_target, int ranks, py::object checkpoint) {
    using Training = typename TrainerT::Training;
    if (ranks < 1)
        throw std::runtime_error(std::string(name) + ": ranks must be at least 1");

    std::vector<std::vector<TrainingStructure>> shares(ranks);
    int count = 0;
    for (py::handle item : structures)
        shares[count++ % ranks].push_back(item.cast<TrainingStructure>());

    std::ofstream log_file;
    std::ostream* log = open_log(log_target, log_file);

    py::object potential = py::cast(std::make_unique<Training>(filename));
    Training& potential_ref = potential.cast<Training&>();
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
            std::unique_ptr<Training> own;
            if (r != 0) own = std::make_unique<Training>(filename);
            Training& pot = r == 0 ? potential_ref : *own;
            TrainerT trainer(pot, std::move(shares[r]), options, &comms[r], r == 0 ? log : nullptr, r == 0 ? save : nullptr);
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

#endif

void bind_core(py::module_& m);
void bind_maxvol(py::module_& m);
void bind_pair_mtp(py::module_& m);
void bind_pair_mtp_extrapolation(py::module_& m);
void bind_mtp_training(py::module_& m);
void bind_mtp_trainer(py::module_& m);
void bind_etn_harmonics(py::module_& m);
void bind_pair_etn(py::module_& m);
void bind_etn_training(py::module_& m);
void bind_etn_trainer(py::module_& m);
