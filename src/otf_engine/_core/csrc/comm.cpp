/* -*- c++ -*- ----------------------------------------------------------
   Comm's collectives over a group of threads and over MPI.
------------------------------------------------------------------------- */

#include "comm.h"

#include <algorithm>
#include <cstring>

void ThreadGroup::barrier() {
    std::unique_lock<std::mutex> lock(mutex);
    if (aborted) throw CommAborted();
    const long long gen = generation;
    if (++arrived == size) {
        arrived = 0;
        generation++;
        cv.notify_all();
        return;
    }
    cv.wait(lock, [&] { return generation != gen || aborted; });
    if (generation == gen) throw CommAborted();
}

void ThreadGroup::abort() {
    std::lock_guard<std::mutex> lock(mutex);
    aborted = true;
    cv.notify_all();
}

// Combines every rank's published buffer into out, in rank order.
template <class T>
static void reduce_slots(const std::vector<const void*>& slots, T* out, int count, ReduceOp op) {
    for (int i = 0; i < count; i++) {
        T acc = static_cast<const T*>(slots[0])[i];
        for (size_t r = 1; r < slots.size(); r++) {
            const T v = static_cast<const T*>(slots[r])[i];
            if (op == ReduceOp::sum) acc = acc + v;
            else if (op == ReduceOp::min) acc = std::min(acc, v);
            else acc = std::max(acc, v);
        }
        out[i] = acc;
    }
}

// Combines in of every rank of group into out on every rank; out may be another rank's in until all have read them.
template <class T>
static void allreduce_slots(ThreadGroup& group, int me, const T* in, T* out, int count, ReduceOp op) {
    group.slots[me] = in;
    group.barrier();
    std::vector<T> result(count);
    reduce_slots(group.slots, result.data(), count, op);
    group.barrier();
    std::copy(result.begin(), result.end(), out);
}

void ThreadComm::bcast_bytes(void* data, size_t bytes, int root) {
    group.slots[me] = data;
    group.barrier();
    if (me != root) std::memcpy(data, group.slots[root], bytes);
    group.barrier();
}

void ThreadComm::reduce(const double* in, double* out, int count, ReduceOp op, int root) {
    group.slots[me] = in;
    group.barrier();
    if (me == root) reduce_slots(group.slots, out, count, op);
    group.barrier();
}

void ThreadComm::allreduce(const double* in, double* out, int count, ReduceOp op) { allreduce_slots(group, me, in, out, count, op); }

void ThreadComm::allreduce(const int* in, int* out, int count, ReduceOp op) { allreduce_slots(group, me, in, out, count, op); }

static MPI_Op mpi_op(ReduceOp op) {
    if (op == ReduceOp::sum) return MPI_SUM;
    if (op == ReduceOp::min) return MPI_MIN;
    return MPI_MAX;
}

MPIComm::MPIComm(MPI_Comm comm) : comm(comm) {
    MPI_Comm_rank(comm, &me);
    MPI_Comm_size(comm, &count);
}

void MPIComm::barrier() { MPI_Barrier(comm); }

void MPIComm::bcast_bytes(void* data, size_t bytes, int root) { MPI_Bcast(data, (int) bytes, MPI_BYTE, root, comm); }

void MPIComm::reduce(const double* in, double* out, int count, ReduceOp op, int root) { MPI_Reduce(in, out, count, MPI_DOUBLE, mpi_op(op), root, comm); }

void MPIComm::allreduce(const double* in, double* out, int count, ReduceOp op) { MPI_Allreduce(in, out, count, MPI_DOUBLE, mpi_op(op), comm); }

void MPIComm::allreduce(const int* in, int* out, int count, ReduceOp op) { MPI_Allreduce(in, out, count, MPI_INT, mpi_op(op), comm); }

void MPIComm::abort() { MPI_Abort(comm, 1); }
