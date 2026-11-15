/* -*- c++ -*- ----------------------------------------------------------
   What MTPTrainer and ETNTrainer share.

   A trainer is SPMD: every rank of a communicator holds a copy of the
   potential and a share of the structures, and the MPI collectives combine
   them. Built with MTP_MPI it uses <mpi.h>; otherwise thread_mpi.h, whose
   ranks are threads.
------------------------------------------------------------------------- */

#pragma once

#include "error_table.h"
#include "training_structure.h"

#ifdef MTP_MPI
#include <mpi.h>
#else
#include "thread_mpi.h"
#endif

#include <functional>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

// The options of both trainers, under mlp's and mlip-4's names, with their defaults.
struct TrainerOptions {
    double energy_weight = 1.0;
    double force_weight = 0.01;
    double stress_weight = 0.001;
    bool balance_species = true;         // weigh each force by N / (k n_s), so every species of a structure has an equal share of its force weight
    std::vector<std::string> species;    // the species' names in the error tables
};

// Training is the potential's training class, Options its trainer's options.
template <class TrainingT, class OptionsT>
class Trainer {
  public:
    using Training = TrainingT;
    using Options = OptionsT;

    // log receives the fit log on the rank given one; checkpoint saves the potential, which then
    // holds the coefficients. Throws on every rank together when no rank has a structure.
    Trainer(Training& potential, std::vector<TrainingStructure> structures, const Options& options, MPI_Comm comm, std::ostream* log, std::function<void()> checkpoint)
        : potential(potential), structures(std::move(structures)), options(options), comm(comm), log(log), checkpoint(std::move(checkpoint)) {
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);

        if (options.balance_species)
            for (TrainingStructure& s : this->structures) s.balance_species();

        int local_count = (int) this->structures.size();
        MPI_Allreduce(&local_count, &structure_count, 1, MPI_INT, MPI_SUM, comm);
        if (structure_count == 0)
            throw std::runtime_error("no training structures");

        coeffs.resize(potential.coeff_count());
        potential.get_coeffs(coeffs.data());
    }

  protected:
    // Logs the errors of the potential at coeffs on the training set under title, unless it is untrained.
    void log_errors(const std::string& title) {
        if (!potential.is_trained()) return;
        potential.set_coeffs(coeffs.data());
        const std::string table = error_table(potential, structures, options.species, comm, title);
        if (log) *log << table;
    }

    Training& potential;
    std::vector<TrainingStructure> structures;
    Options options;
    MPI_Comm comm;
    int rank = 0, size = 1;
    int structure_count = 0;    // over all ranks
    std::ostream* log;
    std::function<void()> checkpoint;

    std::vector<double> coeffs;           // what potential holds, in its coefficient order
    std::vector<double> dloss_dforces;    // per-structure scratch
};
