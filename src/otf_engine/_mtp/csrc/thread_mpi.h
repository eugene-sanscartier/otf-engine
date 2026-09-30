/* -*- c++ -*- ----------------------------------------------------------
   The MPI calls MTPTrainer makes, over a group of threads.

   Stands in for <mpi.h>: each thread of a ThreadGroup is one rank with its
   own communicator handle, and meets the others in these collectives. The
   signatures are MPI's. Reductions combine ranks in rank order, so a result
   depends on the group size and not on scheduling.
------------------------------------------------------------------------- */

#pragma once

#include <condition_variable>
#include <exception>
#include <mutex>
#include <vector>

// Raised in every rank still waiting on a group another rank has aborted.
class CommAborted : public std::exception {
  public:
    const char* what() const noexcept override { return "thread group aborted"; }
};

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

struct ThreadComm {
    ThreadGroup* group;
    int rank;
};

typedef ThreadComm* MPI_Comm;
typedef int MPI_Datatype;
typedef int MPI_Op;

constexpr int MPI_SUCCESS = 0;

constexpr MPI_Datatype MPI_INT = 0;
constexpr MPI_Datatype MPI_DOUBLE = 1;
constexpr MPI_Datatype MPI_C_BOOL = 2;

constexpr MPI_Op MPI_MAX = 0;
constexpr MPI_Op MPI_MIN = 1;
constexpr MPI_Op MPI_SUM = 2;

int MPI_Comm_rank(MPI_Comm comm, int* rank);
int MPI_Comm_size(MPI_Comm comm, int* size);
int MPI_Barrier(MPI_Comm comm);
int MPI_Bcast(void* buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm);
int MPI_Reduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm);
int MPI_Allreduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
