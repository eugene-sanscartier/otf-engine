/* -*- c++ -*- ----------------------------------------------------------
   Comm: a group of ranks and the collectives they meet in, which the
   trainers, error_table and a MaxVol search run over, SPMD. ThreadComm's
   ranks are threads of one process, MPIComm's the processes of an MPI
   communicator.
------------------------------------------------------------------------- */

#pragma once

#include <mpi.h>

#include <condition_variable>
#include <cstddef>
#include <exception>
#include <mutex>
#include <string>
#include <vector>

enum class ReduceOp { sum, min, max };

class Comm {
  public:
    virtual ~Comm() = default;

    virtual int rank() const = 0;
    virtual int size() const = 0;
    virtual void barrier() = 0;

    // Copies data [count] from root to every rank.
    template <class T>
    void bcast(T* data, int count, int root) { bcast_bytes(data, count * sizeof(T), root); }

    // Combines in [count] of every rank into out [count], on root or on every rank.
    virtual void reduce(const double* in, double* out, int count, ReduceOp op, int root) = 0;
    virtual void allreduce(const double* in, double* out, int count, ReduceOp op) = 0;
    virtual void allreduce(const int* in, int* out, int count, ReduceOp op) = 0;

    // Ends every rank after an exception on this one alone: MPI ends the job, and the other threads raise CommAborted.
    virtual void abort() = 0;

  protected:
    virtual void bcast_bytes(void* data, size_t bytes, int root) = 0;
};

// Returns if no rank of comm has an error, else raises on every rank the error of the lowest rank that has one.
void raise_together(Comm& comm, const std::string& error);

// Raised in every rank still waiting on a group another rank has aborted.
class CommAborted : public std::exception {
  public:
    const char* what() const noexcept override { return "thread group aborted"; }
};

// The threads of one process that are a group's ranks, and the buffers they publish to meet.
class ThreadGroup {
  public:
    explicit ThreadGroup(int size) : size(size), slots(size) {}

    const int size;
    std::vector<const void*> slots;    // one buffer per rank, published before a barrier

    void barrier();
    // Releases every waiting rank with CommAborted.
    void abort();

  private:
    std::mutex mutex;
    std::condition_variable cv;
    int arrived = 0;
    long long generation = 0;
    bool aborted = false;
};

// A rank of a ThreadGroup. Reductions combine the ranks in rank order, so a result depends on the group size and
// not on scheduling.
class ThreadComm : public Comm {
  public:
    ThreadComm(ThreadGroup& group, int rank) : group(group), me(rank) {}

    int rank() const override { return me; }
    int size() const override { return group.size; }
    void barrier() override { group.barrier(); }
    void reduce(const double* in, double* out, int count, ReduceOp op, int root) override;
    void allreduce(const double* in, double* out, int count, ReduceOp op) override;
    void allreduce(const int* in, int* out, int count, ReduceOp op) override;
    void abort() override { group.abort(); }

  protected:
    void bcast_bytes(void* data, size_t bytes, int root) override;

  private:
    ThreadGroup& group;
    const int me;
};

// This process's rank of an MPI communicator.
class MPIComm : public Comm {
  public:
    explicit MPIComm(MPI_Comm comm);

    int rank() const override { return me; }
    int size() const override { return count; }
    void barrier() override;
    void reduce(const double* in, double* out, int count, ReduceOp op, int root) override;
    void allreduce(const double* in, double* out, int count, ReduceOp op) override;
    void allreduce(const int* in, int* out, int count, ReduceOp op) override;
    void abort() override;

  protected:
    void bcast_bytes(void* data, size_t bytes, int root) override;

  private:
    MPI_Comm comm;
    int me = 0, count = 1;
};
