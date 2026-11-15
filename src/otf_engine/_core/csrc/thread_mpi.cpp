/* -*- c++ -*- ----------------------------------------------------------
   The MPI calls MTPTrainer makes, over a group of threads.
------------------------------------------------------------------------- */

#include "thread_mpi.h"

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

static size_t type_size(MPI_Datatype datatype) {
    if (datatype == MPI_INT) return sizeof(int);
    if (datatype == MPI_DOUBLE) return sizeof(double);
    return sizeof(bool);
}

template <class T>
static void reduce_typed(const std::vector<const void*>& slots, T* out, int count, MPI_Op op) {
    for (int i = 0; i < count; i++) {
        T acc = static_cast<const T*>(slots[0])[i];
        for (size_t r = 1; r < slots.size(); r++) {
            const T v = static_cast<const T*>(slots[r])[i];
            if (op == MPI_SUM) acc = acc + v;
            else if (op == MPI_MIN) acc = std::min(acc, v);
            else acc = std::max(acc, v);
        }
        out[i] = acc;
    }
}

// Combines every rank's published buffer into out, in rank order.
static void reduce_slots(const std::vector<const void*>& slots, void* out, int count, MPI_Datatype datatype, MPI_Op op) {
    if (datatype == MPI_INT) reduce_typed(slots, static_cast<int*>(out), count, op);
    else if (datatype == MPI_DOUBLE) reduce_typed(slots, static_cast<double*>(out), count, op);
    else reduce_typed(slots, static_cast<bool*>(out), count, op);
}

int MPI_Comm_rank(MPI_Comm comm, int* rank) {
    *rank = comm->rank;
    return MPI_SUCCESS;
}

int MPI_Comm_size(MPI_Comm comm, int* size) {
    *size = comm->group->size;
    return MPI_SUCCESS;
}

int MPI_Barrier(MPI_Comm comm) {
    comm->group->barrier();
    return MPI_SUCCESS;
}

int MPI_Bcast(void* buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm) {
    ThreadGroup& group = *comm->group;
    group.slots[comm->rank] = buffer;
    group.barrier();
    if (comm->rank != root) std::memcpy(buffer, group.slots[root], count * type_size(datatype));
    group.barrier();
    return MPI_SUCCESS;
}

int MPI_Reduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm) {
    ThreadGroup& group = *comm->group;
    group.slots[comm->rank] = sendbuf;
    group.barrier();
    if (comm->rank == root) reduce_slots(group.slots, recvbuf, count, datatype, op);
    group.barrier();
    return MPI_SUCCESS;
}

int MPI_Allreduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
    ThreadGroup& group = *comm->group;
    group.slots[comm->rank] = sendbuf;
    group.barrier();
    std::vector<char> result(count * type_size(datatype));
    reduce_slots(group.slots, result.data(), count, datatype, op);
    group.barrier();
    std::memcpy(recvbuf, result.data(), result.size());
    return MPI_SUCCESS;
}
