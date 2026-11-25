/* -*- c++ -*- ----------------------------------------------------------
   Comm is a group of ranks and their collectives. The fits, error_table
   and MaxVol::maximize_volume run SPMD over a Comm. The ranks of a
   ThreadComm are threads of one process, and those of an MPIComm are the
   processes of an MPI communicator.
------------------------------------------------------------------------- */

#pragma once

#include <mpi.h>

#include <condition_variable>
#include <cstddef>
#include <exception>
#include <mutex>
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

    // Reduces in [count] over the ranks into out [count]: on root for reduce, on every rank for allreduce.
    virtual void reduce(const double* in, double* out, int count, ReduceOp op, int root) = 0;
    virtual void allreduce(const double* in, double* out, int count, ReduceOp op) = 0;
    virtual void allreduce(const int* in, int* out, int count, ReduceOp op) = 0;

    // Stops the other ranks after an exception on this rank alone. MPIComm ends the job, and the other threads of a
    // ThreadComm raise CommAborted.
    virtual void abort() = 0;

  protected:
    virtual void bcast_bytes(void* data, size_t bytes, int root) = 0;
};

// The exception a ThreadGroup barrier raises in every rank once a rank has aborted the group.
class CommAborted : public std::exception {
  public:
    const char* what() const noexcept override { return "thread group aborted"; }
};

// The ranks of a ThreadComm, which are threads of one process, and the buffers they publish in a collective.
class ThreadGroup {
  public:
    explicit ThreadGroup(int size) : size(size), slots(size) {}

    const int size;
    std::vector<const void*> slots;    // [size] the buffer each rank publishes before a barrier

    void barrier();
    // Makes every waiting barrier and every later one raise CommAborted.
    void abort();

  private:
    std::mutex mutex;
    std::condition_variable cv;
    int arrived = 0;
    long long generation = 0;
    bool aborted = false;
};

// A rank of a ThreadGroup. Reductions combine the ranks in rank order.
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

// The rank of this process in an MPI communicator.
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
