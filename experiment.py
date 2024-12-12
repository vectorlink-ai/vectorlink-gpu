from typing import List, Optional
from torch import Tensor
import torch
from torch import profiler
import sys
from datetime import datetime
import time
import sys

import numba
from numba import cuda, gdb_init, void, float32, int64, int32

# INT32_MAX = int32(2147483647)
# FLOAT32_MAX = float32(3.4028235e38)

MAXINT = 99_999_999
MAXFLOAT = 99_999_999.0
MAXINT_CUDA = int32(MAXINT)
MAXFLOAT_CUDA = float32(MAXFLOAT)
DEVICE = "cuda"


class FakeStream:
    def __init__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exception_type, exception_value, traceback):
        pass


def allocate_stream():
    if False and torch.cuda.is_available():
        return torch.cuda.Stream()
    else:
        return FakeStream()


def current_stream(s):
    if False and torch.cuda.is_available():
        return torch.cuda.stream(s)
    else:
        return FakeStream()


def wait_stream(s1, s2):
    if False and torch.cuda.is_available():
        s1.stream_wait(s2)


def record_stream(A, s):
    if False and torch.cuda.is_available():
        A.record_stream(s)


def timed(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000


# print(f"[{datetime.now().strftime('%H:%M:%S.%f')}] {msg}")
def log_time(func):
    def wrapper(*args, **kwargs):
        def closure():
            return func(*args, **kwargs)

        (result, time) = timed(closure)
        print(f"time spent: {time}")
        return result

    return wrapper


class Queue:
    def __init__(self, num_queues: int, queue_length: int, capacity: int):
        assert queue_length < capacity
        self.length = queue_length
        self.indices = torch.full((num_queues, capacity), MAXINT, dtype=torch.int32)
        self.distances = torch.full(
            (num_queues, capacity), MAXFLOAT, dtype=torch.float32
        )
        self.buffers = torch.empty((num_queues, self.length), dtype=torch.int32)

    def initialize_from_queue(self, queue):
        head_length = min(queue.length, self.length)
        indices_head = self.indices.narrow(1, 0, head_length)
        distances_head = self.distances.narrow(1, 0, head_length)
        indices_head.copy_(queue.indices.narrow(1, 0, head_length))
        distances_head.copy_(queue.distances.narrow(1, 0, head_length))

    def insert(
        self,
        vector_id_batch: Tensor,
        distances_batch: Tensor,
        exclude: Optional[Tensor] = None,  # formatted like [[0,MAXINT...],[1],[2]]
    ):
        self.buffers = torch.narrow_copy(self.indices, 1, 0, self.length)
        (batches, size_per_batch) = vector_id_batch.size()
        indices_tail = self.indices.narrow(1, self.length, size_per_batch)
        distances_tail = self.distances.narrow(1, self.length, size_per_batch)
        indices_tail.copy_(vector_id_batch)
        distances_tail.copy_(distances_batch)

        if exclude is not None:
            punchout_excluded(indices_tail, distances_tail, exclude)

        (self.indices, self.distances) = queue_sort(self.indices, self.distances)
        did_something_mask = self.indices.narrow(1, 0, self.length) != self.buffers
        return did_something_mask

    def insert_random_padded(
        self,
        vector_id_batch: Tensor,
        distances_batch: Tensor,
        exclude: Optional[Tensor] = None,  # formatted like [[0],[1],[2]]
    ):
        (batch_size, total_length) = vector_id_batch.size()
        (total_batch_size, _) = self.indices.size()
        difference = total_batch_size - batch_size
        random_padding = generate_circulant_neighborhoods(
            difference, primes(total_length)
        )
        random_distances = cuda_distances(vectors, random_padding)
        ids = torch.vstack([vector_id_batch, random_padding])
        dist = torch.vstack([distances_batch, random_distances])
        self.insert(ids, dist, exclude)

    def print(self, tail: Optional[bool] = False):
        (_bs, dim) = self.size()
        print(self.indices.narrow(1, 0, self.length))
        print(self.distances.narrow(1, 0, self.length))
        if tail:
            print("----------------")
            print(self.indices.narrow(1, self.length, dim - self.length))
            print(self.distances.narrow(1, self.length, dim - self.length))

    def pop_n_ids(self, n: int):
        length = self.length
        take = min(length, n)
        indices = self.indices.narrow(1, 0, take)
        distances = self.distances.narrow(1, 0, take)
        copied_indices = indices.clone()
        indices.fill_(MAXINT)
        distances.fill_(MAXFLOAT)
        return copied_indices

    def size(self):
        return self.indices.size()


@cuda.jit(void(float32[::1], int64, int64, int64), device=True)
def sum_part(vec, dim, scale, idx):
    if idx + scale < dim and idx < scale:
        vec[idx] += vec[idx + scale]


@cuda.jit(float32(float32[::1], int64, int64), device=True)
def sum(vec, dim, idx):
    scale = dim
    while scale > 32:
        scale /= 2
        sum_part(vec, dim, scale, idx)
    numba.cuda.syncthreads()
    result = 0.0
    if idx == 0:
        for i in range(0, scale):
            result += vec[i]
    return result


@cuda.jit(void(float32[::1], float32[::1]))
def sum_kernel(vec, out):
    dim = vec.shape[0]
    vector_idx_group = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
    result = sum(vec, dim, vector_idx_group)
    numba.cuda.syncthreads()
    if vector_idx_group == 0:
        out[0] = result


@cuda.jit(float32(float32[::1], float32[::1], float32[::1], int64, int64), device=True)
def cosine_distance(vec1, vec2, buf, vector_dimension, idx):
    groups = int((vector_dimension + 1023) / 1024)
    for group_index in range(0, groups):
        inner_idx = idx + group_index * 1024
        if inner_idx >= vector_dimension:
            break

        buf[inner_idx] = vec1[inner_idx] * vec2[inner_idx]

    numba.cuda.syncthreads()
    # TODO sum will only work for vectors up to dim 2048 the way things are written now
    cos_theta = sum(buf, vector_dimension, idx)
    numba.cuda.syncthreads()
    result = 1234.0
    if idx == 0:
        result = (1 - cos_theta) / 2
        """
        print(
            "vec1: ",
            vec1[0],
            ",",
            vec1[1],
            "| vec2: ",
            vec2[0],
            ",",
            vec2[1],
            "cos_theta: ",
            cos_theta,
            "result: ",
            result,
        )
        """
        if result < 0.0:
            result = 0.0
        elif result > 1.0:
            result = 1.0
    return result


@cuda.jit(void(float32[::1], float32[::1], float32[::1]))
def cosine_distance_kernel(vec1, vec2, out):
    scratch_buffer = numba.cuda.shared.array(1536 * 4, numba.float32)
    dimension = vec1.shape[0]
    vector_idx_group = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
    if vector_idx_group < dimension:
        result = cosine_distance(
            vec1, vec2, scratch_buffer, dimension, vector_idx_group
        )
        if vector_idx_group == 0:
            out[0] = result


@cuda.jit(
    void(
        float32[:, ::1],
        int32[:, ::1],
        int32[:, ::1],
        float32[:, ::1],
        int32[:, ::1],
        float32[:, ::1],
    )
)
def cuda_search_from_seeds_kernel(
    query_vecs, neighbors_to_visit, neighborhoods, vectors, index_out, distance_out
):
    distance_buffer = numba.cuda.shared.array(4 * 1536, float32)

    batch_idx = cuda.blockIdx.x
    queue_idx = cuda.blockIdx.y
    neighborhood_idx = cuda.blockIdx.z

    vector_idx = cuda.threadIdx.x

    batch_size = query_vecs.shape[0]
    vector_count = vectors.shape[0]
    vector_dim = query_vecs.shape[1]
    neighborhood_size = neighborhoods.shape[1]
    queue_size = neighbors_to_visit.shape[1]

    if vector_idx >= vector_dim or queue_idx >= queue_size or batch_idx >= batch_size:
        return

    assert batch_idx < batch_size
    assert queue_idx < queue_size
    node_id = neighbors_to_visit[batch_idx, queue_idx]
    if node_id == MAXINT:
        if vector_idx == 0:
            output_idx = queue_idx * neighborhood_size + neighborhood_idx
            index_out[batch_idx, output_idx] = MAXINT
            distance_out[batch_idx, output_idx] = MAXFLOAT
    elif node_id < vector_count:
        neighbor_vector_id = neighborhoods[node_id, neighborhood_idx]
        assert neighbor_vector_id < vector_count
        vector = vectors[neighbor_vector_id]
        query_vector = query_vecs[batch_idx]
        result = cosine_distance(
            vector, query_vector, distance_buffer, vector_dim, vector_idx
        )
        if vector_idx == 0:
            output_idx = queue_idx * neighborhood_size + neighborhood_idx
            index_out[batch_idx, output_idx] = neighbor_vector_id
            distance_out[batch_idx, output_idx] = result
    else:
        assert False


@cuda.jit(
    void(
        float32[:, ::1],
        int32[:, ::1],
        float32[:, ::1],
    )
)
def cuda_distances_kernel(
    vectors: Tensor, neighborhoods: Tensor, distances_out: Tensor
):
    distance_buffer = numba.cuda.shared.array(4 * 1536, float32)

    vector_id = cuda.blockIdx.x
    neighborhood_idx = cuda.blockIdx.y
    vector_group_idx = cuda.threadIdx.x

    vector_dimension = vectors.shape[1]

    neighbor_id = neighborhoods[vector_id, neighborhood_idx]
    if neighbor_id == MAXINT:
        distances_out[vector_id, neighborhood_idx] = MAXFLOAT
        return

    vector = vectors[vector_id]
    neighbor_vector = vectors[neighbor_id]

    result = cosine_distance(
        vector, neighbor_vector, distance_buffer, vector_dimension, vector_group_idx
    )

    if vector_group_idx == 0:
        """
        print(
            "vector_id: ",
            vector_id,
            "neighborhood_idx: ",
            neighborhood_idx,
            "neighbor_id: ",
            neighbor_id,
            "result: ",
            result,
        )
        """
        distances_out[vector_id, neighborhood_idx] = result


@log_time
def cuda_distances(
    vectors: Tensor, neighborhoods: Tensor, stream=None, distances_out=None
):
    assert vectors.dtype == torch.float32
    assert vectors.is_contiguous()
    assert neighborhoods.dtype == torch.int32
    assert neighborhoods.is_contiguous()
    if distances_out:
        assert distances_out.is_contiguous()

    (vector_count, vector_dimension) = vectors.size()
    (batches, neighborhood_size) = neighborhoods.size()

    assert batches == vector_count

    vector_idx_group_size = min(vector_dimension, 1024)
    float_size = 4

    grid = (vector_count, neighborhood_size, 1)
    block = (vector_idx_group_size, 1, 1)

    if not distances_out:
        distances_out = torch.empty(
            (vector_count, neighborhood_size), dtype=torch.float32
        )

    cuda_distances_kernel[grid, block, stream, vector_dimension * float_size](
        vectors, neighborhoods, distances_out
    )

    return distances_out


COUNT = 0


def cuda_search_from_seeds(
    query_vecs: Tensor,
    neighbors_to_visit: Tensor,
    neighborhoods: Tensor,
    vectors: Tensor,
) -> (Tensor, Tensor):
    global COUNT

    assert query_vecs.dtype == torch.float32
    assert query_vecs.is_contiguous()
    assert neighbors_to_visit.dtype == torch.int32
    assert neighbors_to_visit.is_contiguous()
    assert neighborhoods.dtype == torch.int32
    assert neighborhoods.is_contiguous()
    assert vectors.is_contiguous()
    assert vectors.dtype == torch.float32

    (batch_size, visit_length) = neighbors_to_visit.size()
    (_, neighborhood_size) = neighborhoods.size()
    (_, vector_dimension) = vectors.size()
    # TODO 1024 should probably be queried instead
    vector_idx_group_size = min(vector_dimension, 1024)
    float_size = 4
    # print(f"COUNT {COUNT}")

    grid = (batch_size, visit_length, neighborhood_size)
    block = (vector_idx_group_size, 1, 1)

    # index_out = torch.empty(
    #     (batch_size, visit_length * neighborhood_size), dtype=torch.int32, device=DEVICE
    # )

    index_out = torch.full(
        (batch_size, visit_length * neighborhood_size),
        MAXINT,
        dtype=torch.int32,
        device=DEVICE,
    )

    # distance_out = torch.empty(
    #     (batch_size, visit_length * neighborhood_size),
    #     dtype=torch.float32,
    #     device=DEVICE,
    # )
    distance_out = torch.full(
        (batch_size, visit_length * neighborhood_size),
        MAXFLOAT,
        dtype=torch.float32,
        device=DEVICE,
    )

    cuda_search_from_seeds_kernel[grid, block, None, vector_dimension * float_size](
        query_vecs, neighbors_to_visit, neighborhoods, vectors, index_out, distance_out
    )
    # print("index out")
    # print(index_out)
    # COUNT += 1
    # if COUNT > 10:
    #     COUNT = 0
    #     return
    return (index_out, distance_out)


@cuda.jit(void(int32[:, ::1]))
def mark_kernel(indices: Tensor):
    queue_len = indices.shape[1]

    batch_idx = cuda.blockIdx.x
    queue_idx = cuda.threadIdx.x
    if queue_len > 1 and queue_idx < queue_len - 1:
        flag = int32(1) << 30
        bitmask = ~flag
        left = indices[batch_idx, queue_idx] & bitmask
        right = indices[batch_idx, queue_idx + 1] & bitmask
        if left == right:
            indices[batch_idx, queue_idx + 1] |= flag


@cuda.jit(void(int32[:, ::1], float32[:, ::1]))
def punchout_pair_kernel(indices: Tensor, distances: Tensor):
    queue_len = indices.shape[1]

    batch_idx = cuda.blockIdx.x
    queue_idx = cuda.threadIdx.x

    if queue_idx < queue_len:
        flag = int32(1) << 30
        bitmask = ~flag
        value = indices[batch_idx, queue_idx]
        if value & flag != 0:
            indices[batch_idx, queue_idx] = MAXINT
            distances[batch_idx, queue_idx] = MAXFLOAT


@cuda.jit(void(int32[:, ::1]))
def punchout_kernel(indices: Tensor):
    queue_len = indices.shape[1]

    batch_idx = cuda.blockIdx.x
    queue_idx = cuda.threadIdx.x

    if queue_idx < queue_len:
        flag = int32(1) << 30
        bitmask = ~flag
        value = indices[batch_idx, queue_idx]
        if value & flag != 0:
            indices[batch_idx, queue_idx] = MAXINT


def dedup_tensor_(ids: Tensor, stream=None):
    (batch_size, queue_size) = ids.size()
    assert ids.is_contiguous()
    assert ids.dtype == torch.int32
    grid = (batch_size, 1, 1)
    block = (queue_size, 1, 1)
    mark_kernel[grid, block, stream, 0](ids)
    punchout_kernel[grid, block, stream, 0](ids, distances)


def dedup_tensor_pair_(ids: Tensor, distances: Tensor, stream=None):
    """Assumes sorted tensors!"""
    (batch_size, queue_size) = ids.size()
    (batch_size2, queue_size2) = distances.size()
    assert batch_size == batch_size2
    assert queue_size == queue_size2
    assert ids.is_contiguous()
    assert distances.is_contiguous()

    grid = (batch_size, 1, 1)
    block = (queue_size, 1, 1)
    mark_kernel[grid, block, stream, 0](ids)
    punchout_pair_kernel[grid, block, stream, 0](ids, distances)


def comparison(qvs, nvs):
    with profiler.record_function("comparison"):
        batch_size, queue_size, vector_dim = nvs.size()
        results = (
            1 - nvs.bmm(qvs.reshape(batch_size, 1, vector_dim).transpose(1, 2))
        ) / 2
        return results.reshape(batch_size, queue_size)


def cpu_search_from_seeds(
    query_vecs: Tensor,
    neighbors_to_visit: Tensor,
    neighborhoods: Tensor,
    vectors: Tensor,
):
    # print_timestamp("start of search_from_seeds")
    (batch_size, visit_length) = neighbors_to_visit.size()
    (_, neighborhood_size) = neighborhoods.size()
    (_, vector_dimension) = vectors.size()

    # print_timestamp("making filter mask")
    filter_mask = neighbors_to_visit == MAXINT
    # print_timestamp("punchout")
    neighbors_to_visit[filter_mask] = 0  # set to 0 to get a valid element
    # print_timestamp("index select")
    index_list = neighborhoods.index_select(0, neighbors_to_visit.flatten()).flatten()
    # print_timestamp("index view")
    indexes_of_comparisons = index_list.view(
        batch_size, visit_length * neighborhood_size
    )
    print_timestamp("vectors")
    # preallocated

    vectors_for_comparison = torch.index_select(vectors, 0, index_list).view(
        batch_size, visit_length * neighborhood_size, vector_dimension
    )
    print_timestamp("did index select")
    # return (query_vecs, vectors_for_comparison)
    # print_timestamp(" before compare")
    distances_from_comparison = comparison(query_vecs, vectors_for_comparison)
    # print_timestamp(" after compare")

    expanded_filter_mask = (
        filter_mask.reshape(batch_size, visit_length, 1)
        .expand(batch_size, visit_length, neighborhood_size)
        .reshape(batch_size, visit_length * neighborhood_size)
    )
    indexes_of_comparisons[expanded_filter_mask] = MAXINT
    distances_from_comparison[expanded_filter_mask] = MAXFLOAT
    # print_timestamp("end of search_from_seeds")
    return (indexes_of_comparisons, distances_from_comparison)


def search_from_seeds(
    query_vecs: Tensor,
    neighbors_to_visit: Tensor,
    neighborhoods: Tensor,
    vectors: Tensor,
):
    with profiler.record_function("search_from_seeds"):
        if torch.cuda.is_available():
            return cuda_search_from_seeds(
                query_vecs, neighbors_to_visit, neighborhoods, vectors
            )
        else:
            return cpu_search_from_seeds(
                query_vecs, neighbors_to_visit, neighborhoods, vectors
            )


PARALLEL_VISIT_COUNT = 3
VISIT_QUEUE_LEN = 24 * 3
EXCLUDE_FACTOR = 5


def closest_vectors(
    query_vecs: Tensor,
    search_queue: Queue,
    vectors: Tensor,
    neighborhoods: Tensor,
    exclude: Optional[Tensor] = None,
):
    with profiler.record_function("closest_vectors"):
        # print_timestamp("start of closest vectors")
        (_, vector_dimension) = vectors.size()
        (neighborhood_count, neighborhood_size) = neighborhoods.size()
        extra_capacity = neighborhood_size * PARALLEL_VISIT_COUNT
        (batch_size, queue_capacity) = search_queue.size()
        capacity = VISIT_QUEUE_LEN + extra_capacity
        visit_queue = Queue(batch_size, VISIT_QUEUE_LEN, capacity)
        visit_queue.initialize_from_queue(search_queue)

        did_something = torch.full([batch_size], True)
        seen = torch.full(
            (batch_size, VISIT_QUEUE_LEN * EXCLUDE_FACTOR),
            MAXINT,
            dtype=torch.int32,
        )
        """
        Stream Diagram

        s1                       s2               s3
        search_from_seeds--wait->|                |
        |                        |                |
        search_queue_insert   rowwise -- wait --->|
        |                        |                |
        |               indexes of comp    distances of comp
        |                        |                |
        |                        |                |
        |                        |<-wait----------|
        |                 visit queue insert      |
        |                        |                |
        |<-wait------------------o-----------------
        |<-wait------------------|   U

        Stream DAG
                                s1
                                 |
                           search_from_seeds
                        s1/       \\s2
        search_queue_insert        rowwise______________
                     s1|             |s2                \\ s3
                       |    indexes of comparison   ____distances of comparison
                       |             |             /                      |
                       |             |s2        s3/                       |s3
                       |       visit_queue_insert                         |
                      \\             |s2                                 /
                        ------------------------------------------------
                                               |
                                          back to search from seeds
        """

        while torch.any(did_something):
            # print_timestamp("start of loop")
            neighbors_to_visit = visit_queue.pop_n_ids(PARALLEL_VISIT_COUNT)
            if seen is not None:
                seen = add_new_to_seen_(seen, neighbors_to_visit)
            (_, visit_length) = neighbors_to_visit.size()
            narrow_to = batch_size * visit_length * neighborhood_size
            # print_timestamp("popped")
            s1 = allocate_stream()
            s2 = allocate_stream()
            s3 = allocate_stream()

            with current_stream(s1):
                (indexes_of_comparisons, distances_of_comparisons) = search_from_seeds(
                    query_vecs,
                    neighbors_to_visit,
                    neighborhoods,
                    vectors,
                )
            wait_stream(s2, s1)
            # Search queue
            with current_stream(s1):
                did_something = search_queue.insert(
                    indexes_of_comparisons, distances_of_comparisons, exclude=exclude
                )
            wait_stream(s2, s3)
            with current_stream(s2):
                if seen is not None:
                    visit_queue.insert(
                        indexes_of_comparisons, distances_of_comparisons, exclude=seen
                    )

            wait_stream(s1, s2)
            wait_stream(s1, s3)
    return True


def search_layers(
    layers: List[Tensor], query_vecs: Tensor, search_queue: Queue, vectors: Tensor
):
    (number_of_batches, _) = query_vecs.size()
    # we don't exclude everything, since we're starting from actual query vecs, not indices
    for layer in layers:
        closest_vectors(query_vecs, search_queue, vectors, layer, None)


def dedup_sort(tensor: Tensor):
    (tensor, _) = tensor.sort()
    dedup_tensor_(tensor)
    (tensor, _) = tensor.sort()
    return tensor


def add_new_to_seen_(seen, indices):
    (dim1, dim2) = seen.size()
    mask = seen == MAXINT
    punched_mask = torch.all(mask, dim=0)
    ascending = torch.arange(dim2, dtype=torch.int32)
    match_indices = ascending[punched_mask]
    (size,) = match_indices.size()
    if size == 0:
        return None
    else:
        """
        [ [ 0, 3,   999],
          [ 1, 999, 999]
        ]

        [2] = [ 0 , 1,  2] [punch_mask]
        """

        first = match_indices[0]
        (new_dim1, new_dim2) = indices.size()
        remaining = dim2 - first
        num_to_copy = min(remaining, new_dim2)
        if num_to_copy == 0:
            return seen
        seen_tail = seen.narrow(1, first, num_to_copy)
        indices_tail = indices.narrow(1, 0, num_to_copy)
        seen_tail.copy_(indices_tail)
        seen.copy_(dedup_sort(seen))
        return seen


#    return shrink_to_fit(seen)


def shrink_to_fit(seen):
    with profiler.record_function("shrink_to_fit"):
        (values, _) = seen.sort()
        (dim1, dim2) = seen.size()
        shifted = torch.hstack(
            [values[:, 1:], torch.full([dim1, 1], MAXINT, dtype=torch.int32)]
        )
        mask = values == shifted
        values[mask] = MAXINT
        (values, _) = values.sort()
        max_val_mask = values == MAXINT
        punched_mask = torch.all(max_val_mask, dim=0)
        index_upto_dim = torch.arange(dim2)
        match_indices = index_upto_dim[punched_mask]
        (size,) = match_indices.size()
        if size > 0:
            max_col = match_indices[0]
        else:
            max_col = 0
        return values.narrow(1, 0, max_col)


def shrink_to_fit_old(seen):
    with profiler.record_function("shrink_to_fit"):
        (values, _) = seen.sort()
        (dim1, dim2) = seen.size()
        shifted = torch.hstack([values[:, 1:], torch.full([dim1, 1], MAXINT)])
        mask = values == shifted
        values[mask] = MAXINT
        (values, _) = values.sort()
        max_val_mask = values == MAXINT
        nonzeroes = torch.nonzero(torch.all(max_val_mask, dim=0))
        # print(nonzeroes.size())
        # print(nonzeroes.size()[0])
        if nonzeroes.size()[0] == 0:
            return seen
        max_col = (nonzeroes[0].sum() - 1).clamp(min=0)
        return seen.narrow(1, 0, max_col)


def punch_out_duplicates(ids: Tensor, distances: Tensor):
    with profiler.record_function("punch_out_duplicates"):
        dedup_tensor_pair_(ids, distances)
        (distances, perm) = distances.sort()
        ids = index_by_tensor(ids, perm)
        assert ids.dtype == torch.int32
        assert distances.dtype == torch.float32
        return (ids, distances)


# def punch_out_duplicates(ids: Tensor, distances: Tensor):
#     with profiler.record_function("punch_out_duplicates"):
#         dim1, dim2 = ids.size()
#         shifted_ids = torch.hstack([ids[:, 1:], torch.full([dim1, 1], MAXINT)])
#         mask = ids == shifted_ids
#         ids[mask] = MAXINT
#         distances[mask] = MAXFLOAT
#         return index_sort(ids, distances)


def index_by_tensor(a: Tensor, b: Tensor):
    with profiler.record_function("index_by_tensor"):
        dim1, dim2 = a.size()
        a = a[
            torch.arange(dim1).unsqueeze(1).expand((dim1, dim2)).flatten(), b.flatten()
        ].view(dim1, dim2)
        return a


def index_sort(neighborhoods: Tensor, neighborhood_distances: Tensor):
    with profiler.record_function("index_sort"):
        (ns, indices) = neighborhoods.sort(1)

        nds = index_by_tensor(neighborhood_distances, indices)

        (nds, indices) = nds.sort(dim=1, stable=True)

        ns = index_by_tensor(ns, indices)

        return (ns, nds)


def queue_sort(neighborhoods: Tensor, neighborhood_distances: Tensor):
    with profiler.record_function("queue_sort"):
        (ns, nds) = index_sort(neighborhoods, neighborhood_distances)
        return punch_out_duplicates(ns, nds)


"""

We can prune an edge as "detourable" if it is strictly less than the elements of a path, i.e.

max(a,b) < c

  a     b
x -  y  - z
 '_______'
    c

"""


def example_db():
    vs = torch.tensor(
        [
            [0.0, 1.0],  # 0
            [1.0, 0.0],  # 1
            [-1.0, 0.0],  # 2
            [0.0, -1.0],  # 3
            [0.707, 0.707],  # 4
            [0.707, -0.707],  # 5
            [-0.707, 0.707],  # 6
            [-0.707, -0.707],  # 7
        ],
        dtype=torch.float32,
        device="cuda",
    )

    neighborhoods = torch.tensor(
        [[4, 6], [4, 5], [6, 7], [5, 7], [0, 1], [1, 3], [0, 2], [2, 3]],
        dtype=torch.int32,
        device="cuda",
    )

    return (vs, neighborhoods)


def two_hop(ns: Tensor):
    dim1, dim2 = ns.size()
    return ns.index_select(0, ns.flatten()).flatten().view(dim1, dim2 * dim2)


def distances(vs: Tensor, neighborhoods: Tensor):
    """
    NOTE: Need a distances kernel!

    """
    _, vec_dim = vs.size()
    number_of_vectors, neighborhood_size = neighborhoods.size()
    query_vector_indices = torch.arange(number_of_vectors)
    qvs = vs.index_select(0, query_vector_indices).reshape(
        number_of_vectors, vec_dim, 1
    )
    nvs = vs.index_select(0, neighborhoods.flatten()).view(
        number_of_vectors, neighborhood_size, vec_dim
    )
    return comparison(qvs, nvs)


def prune(neighborhood: Tensor, neighborhood_distances: Tensor, vectors: Tensor):
    two_hop_neighbors = two_hop(neighborhood)

    pass


PRIMES = torch.tensor(
    [
        1,
        2,
        59063,
        79193,
        3,
        5,
        7,
        11,
        13,
        17,
        19,
        23,
        29,
        31,
        37,
        41,
        43,
        47,
        53,
        59,
        61,
        67,
        71,
        73,
        79,
        83,
        89,
        97,
        101,
        103,
        107,
        109,
        113,
        127,
        131,
        137,
        139,
        149,
        151,
        157,
        163,
        167,
        173,
        179,
        181,
        191,
        193,
        197,
        199,
        211,
        223,
        227,
        229,
        233,
        239,
        241,
        251,
        257,
        263,
        269,
        271,
        277,
        281,
        283,
        293,
        307,
        311,
        313,
        317,
        331,
        337,
        347,
        349,
        353,
        359,
        367,
        373,
        379,
        383,
        389,
        397,
        401,
        409,
        419,
        421,
        431,
        433,
        439,
        443,
        449,
        457,
        461,
        463,
        467,
        479,
        487,
        491,
        499,
        503,
        509,
        521,
        523,
        541,
        547,
        557,
        563,
        569,
        571,
        577,
        587,
        593,
        599,
        601,
        607,
        613,
        617,
        619,
        631,
        641,
        643,
        647,
        653,
        659,
        661,
        673,
        677,
        683,
        691,
        701,
        709,
        719,
        727,
        733,
        739,
        743,
        751,
        757,
        761,
        769,
        773,
        787,
        797,
        809,
        811,
        821,
        823,
        827,
        829,
        839,
        853,
        857,
        859,
        863,
        877,
        881,
        883,
        887,
        907,
        911,
        919,
        929,
        937,
        941,
        947,
        953,
        967,
        971,
        977,
        983,
        991,
        997,
    ],
    dtype=torch.int32,
    device=DEVICE,
)


def primes(size: int):
    return PRIMES.narrow(0, 0, size)


def generate_circulant_neighborhoods(num_vecs: int, primes: Tensor) -> Tensor:
    indices = torch.arange(num_vecs, device=DEVICE, dtype=torch.int32)
    (nhs,) = primes.size()
    repeated_indices = indices.expand(nhs, num_vecs).transpose(0, 1)
    repeated_primes = primes.expand(num_vecs, nhs)
    circulant_neighbors = repeated_indices + repeated_primes
    return circulant_neighbors.sort().values % num_vecs


def generate_layered_ann(neighborhood_size: int, vectors: Tensor):
    """ """
    layers = []
    (count, _) = vectors.size()
    order = neighborhood_size * 3
    size = order
    while True:
        bound = min(size, count)
        (ann, _) = generate_ann(neighborhood_size, vectors[0:bound])
        layers.append(ann)
        if size > count:
            break
        else:
            size = order * size

    return layers


def generate_hnsw(neighborhood_size: int, vectors: Tensor):
    """ """
    (vector_count, _) = vectors.size()
    order = neighborhood_size * 3
    layer_size = order
    bound = min(layer_size, vector_count)
    (ann, distances) = generate_ann(neighborhood_size, vectors[0:bound])

    layers = [ann]
    queue_length = neighborhood_size * NEIGHBORHOOD_QUEUE_FACTOR
    remaining_capacity = queue_length * PARALLEL_VISIT_COUNT

    queue = Queue(
        bound,
        queue_length,
        queue_length + remaining_capacity,
    )  # make que from neighborhoods + neigbhorhood_distances
    # print("neigbhorhoods")
    # print(neighborhoods)
    # print("neighborhood distances")
    # print(neighborhood_distances)

    c = 0
    print(f"layer_size: {layer_size}, vector_count: {vector_count}")
    while layer_size < vector_count:
        layer_size = order * layer_size
        bound = min(layer_size, vector_count)

        print(f"round: {c}, size: {bound}")
        queue = Queue(
            bound,
            queue_length,
            queue_length + remaining_capacity,
        )
        queue.insert_random_padded(ann, distances)

        qvs = vectors[:bound]
        search_layers(layers, qvs, queue, vectors)

        ann = queue.indices.narrow_copy(1, 0, neighborhood_size)
        print(ann)
        print(f"Isolated layer recall")
        ann_calculate_recall(vectors[:bound], ann)
        distances = queue.distances.narrow_copy(1, 0, neighborhood_size)

        layers.append(ann)
        c += 1

    return layers


NEIGHBORHOOD_QUEUE_FACTOR = 3
CAGRA_LOOPS = 3


def generate_ann(neighborhood_size: int, vectors: Tensor) -> Tensor:
    # print_timestamp("generating ann")
    (num_vecs, vec_dim) = vectors.size()
    queue_length = neighborhood_size * NEIGHBORHOOD_QUEUE_FACTOR
    neighbor_primes = primes(queue_length)
    neighborhoods = generate_circulant_neighborhoods(num_vecs, neighbor_primes)
    assert neighborhoods.dtype == torch.int32
    # print_timestamp("circulant neighborhoods generated")
    neighborhood_distances = cuda_distances(vectors, neighborhoods)

    # print_timestamp("distances calculated")
    # we want to be able to add a 'big' neighborhood at the end, which happens to be queue_length
    remaining_capacity = queue_length * PARALLEL_VISIT_COUNT
    queue = Queue(
        num_vecs,
        queue_length,
        queue_length + remaining_capacity,
    )  # make que from neighborhoods + neigbhorhood_distances
    queue.insert(neighborhoods, neighborhood_distances)
    # print("neigbhorhoods")
    # print(neighborhoods)
    # print("neighborhood distances")
    # print(neighborhood_distances)
    # print_timestamp("initial queue constructed from neighborhoods")
    exclude = excluding_self(num_vecs, queue_length)

    for i in range(0, CAGRA_LOOPS):
        print_timestamp(f"start of cagra loop {i}")
        closest_vectors(vectors, queue, vectors, neighborhoods, exclude)
        # print_timestamp(f" closest vectors calculated")
        # print(f"queue at cagra loop {i}")
        # queue.print()
        neighborhoods = queue.indices.narrow_copy(1, 0, queue_length)
        ann_calculate_recall(
            vectors, neighborhoods.narrow_copy(1, 0, neighborhood_size)
        )
        print_timestamp(f"end of cagra loop {i}")
    return (
        neighborhoods.narrow_copy(1, 0, neighborhood_size),
        queue.distances.narrow_copy(1, 0, neighborhood_size),
    )


@cuda.jit(void(int32[:, :], float32[:, :], int32[:, :]))
def punchout_excluded_kernel(indices: Tensor, distances: Tensor, exclusions: Tensor):
    exclusions_length = exclusions.shape[1]
    batch_idx = cuda.blockIdx.x
    queue_idx = cuda.threadIdx.x

    value = indices[batch_idx, queue_idx]
    for i in range(0, exclusions_length):
        exclude = exclusions[batch_idx, i]
        if exclude == MAXINT:
            break
        if value == exclude:
            indices[batch_idx, queue_idx] = MAXINT
            distances[batch_idx, queue_idx] = MAXFLOAT


def punchout_excluded(indices, distances, exclusions, stream=None):
    (batch_size, queue_size) = indices.size()
    (batch_size2, exclusion_size) = exclusions.size()
    assert batch_size == batch_size2
    grid = (batch_size, 1, 1)
    block = (queue_size, 1, 1)
    punchout_excluded_kernel[grid, block, stream, 0](indices, distances, exclusions)


def rowwise_isin(indices, exclusions):
    matches = indices.unsqueeze(2) == exclusions.unsqueeze(1)

    # result: boolean tensor of shape (N, K) where result[n, k] is torch.isin(tensor_1[n, k], target_tensor[n])
    result = torch.sum(matches, dim=2, dtype=torch.bool)

    return result


def initial_queue(vectors: Tensor, neighborhood_size: int, queue_size: int):
    (batch_size, _) = vectors.size()
    extra_capacity = max(queue_size, PARALLEL_VISIT_COUNT * neighborhood_size)
    queue = Queue(batch_size, queue_size, queue_size + extra_capacity)
    p = primes(queue_size)
    initial_queue_indices = generate_circulant_neighborhoods(batch_size, p)
    d = cuda_distances(vectors, initial_queue_indices)
    queue.insert(initial_queue_indices, d)

    return queue


RECALL_SEARCH_QUEUE_LENGTH = 6


def ann_calculate_recall(vectors, neighborhoods):
    (number_of_vectors, neighborhood_size) = neighborhoods.size()

    queue = initial_queue(
        vectors, neighborhood_size, RECALL_SEARCH_QUEUE_LENGTH * neighborhood_size
    )
    # print_timestamp("queues allocated")

    closest_vectors(vectors, queue, vectors, neighborhoods)
    # print_timestamp("closest vectors calculated")
    expected = torch.arange(number_of_vectors)
    actual = queue.indices.t()[0]
    found = (expected == actual).sum().item()

    # print_timestamp("calculated recall")

    print(found)
    print(found / number_of_vectors)


def hnsw_calculate_recall(vectors, hnsw):
    (number_of_vectors, neighborhood_size) = hnsw[-1].size()

    queue = initial_queue(
        vectors, neighborhood_size, RECALL_SEARCH_QUEUE_LENGTH * neighborhood_size
    )
    # print_timestamp("queues allocated")

    search_layers(hnsw, vectors, queue, vectors)
    # print_timestamp("closest vectors calculated")
    expected = torch.arange(number_of_vectors)
    actual = queue.indices.t()[0]
    found = (expected == actual).sum().item()

    # print_timestamp("calculated recall")

    print(found)
    print(found / number_of_vectors)


def ann_recall_test(vectors: Tensor, neighborhood_size: int = 24):
    # print_timestamp("vectors allocated")
    (neighborhoods, _) = generate_ann(neighborhood_size, vectors)
    print_timestamp("=> ANN generated")

    ann_calculate_recall(vectors, neighborhoods)


def layered_ann_recall_test(vectors: Tensor, neighborhood_size: int = 24):
    # print_timestamp("vectors allocated")
    hnsw = generate_layered_ann(neighborhood_size, vectors)
    print_timestamp("=> Layered ANN generated")
    hnsw_calculate_recall(vectors, hnsw)


def hnsw_recall_test(vectors: Tensor, neighborhood_size: int = 24):
    # print_timestamp("vectors allocated")
    hnsw = generate_hnsw(neighborhood_size, vectors)
    for layer in hnsw:
        (a, b) = layer.size()
        print(f"shape: ({a}, {b})")

    print_timestamp("=> HNSW generated")

    hnsw_calculate_recall(vectors, hnsw)


def generate_random_vectors(number_of_vectors: int, dimensions: int = 1536) -> Tensor:
    vectors = torch.nn.functional.normalize(
        torch.randn(number_of_vectors, dimensions, dtype=torch.float32), dim=1
    )
    return vectors


# Function to print timestamps
def print_timestamp(msg):
    print(msg)


@log_time
def test_sort(queue_len=128, vector_count=100):
    indices = torch.randint(
        0, vector_count, (vector_count, queue_len), dtype=torch.int32, device="cuda"
    )
    distances = torch.rand(
        (vector_count, queue_len), dtype=torch.float32, device="cuda"
    )
    (distances, reorder) = torch.sort(distances)
    indices = index_by_tensor(indices, reorder)
    (indices, reorder) = torch.sort(indices, stable=True)
    distances = index_by_tensor(distances, reorder)


def excluding_self(num_vecs, queue_length):
    exclude = torch.full(
        (num_vecs, VISIT_QUEUE_LEN * EXCLUDE_FACTOR), MAXINT, dtype=torch.int32
    )
    exclude_front = exclude.narrow(1, 0, 1)
    excluded_ids = torch.arange(num_vecs).unsqueeze(1)
    exclude_front.copy_(excluded_ids)
    return exclude


@log_time
def generate_test_ann(
    neighborhood_size=24, vector_dimension=1536, vector_count=1_000_000
):
    torch.set_default_device(DEVICE)
    torch.set_float32_matmul_precision("high")

    vectors = torch.nn.functional.normalize(
        torch.randn(vector_count, vector_dimension, dtype=torch.float32, device="cuda"),
        dim=1,
    )

    queue_length = neighborhood_size * NEIGHBORHOOD_QUEUE_FACTOR
    neighborhood_primes = primes(queue_length)
    neighborhoods = generate_circulant_neighborhoods(vector_count, neighborhood_primes)
    neighborhood_distances = cuda_distances(vectors, neighborhoods)
    remaining_capacity = queue_length * PARALLEL_VISIT_COUNT
    queue = Queue(
        vector_count,
        queue_length,
        queue_length + remaining_capacity,
    )
    queue.insert(neighborhoods, neighborhood_distances)

    exclude = excluding_self(vector_count, queue_length)

    for i in range(0, CAGRA_LOOPS):
        print_timestamp(f"start of cagra loop {i}")
        closest_vectors(vectors, queue, vectors, neighborhoods, exclude)
        # print_timestamp(f" closest vectors calculated")
        # print(f"queue at cagra loop {i}")
        # queue.print()
        neighborhoods = queue.indices.narrow_copy(1, 0, queue_length)
        ann_calculate_recall(
            vectors, neighborhoods.narrow_copy(1, 0, neighborhood_size)
        )
        print_timestamp(f"end of cagra loop {i}")


if __name__ == "__main__":
    torch.set_default_device(DEVICE)
    torch.set_float32_matmul_precision("high")

    # grid = (1, 1, 1)
    # block = (1024, 1, 1)
    # vec_in = torch.arange(1536, dtype=torch.float32, device="cuda")
    # out = torch.tensor([0.0], dtype=torch.float32, device="cuda")
    # sum_kernel[grid, block](vec_in, out)
    # print(out)
    # sys.exit(0)

    # v1 = torch.nn.functional.normalize(torch.full((1536,), 1.0), dim=0)
    # v2 = torch.nn.functional.normalize(torch.full((1536,), 1.0), dim=0)
    # grid = (1, 1, 1)
    # block = (1024, 1, 1)
    # result = torch.tensor([4.0], dtype=torch.float32, device="cuda")
    # cosine_distance_kernel[grid, block](v1, v2, numba.cuda.as_cuda_array(result))
    # print("cosine kernel result")
    # print(result)
    # sys.exit(0)

    # with torch.profiler.profile(
    #     activities=[
    #         torch.profiler.ProfilerActivity.CUDA,
    #         torch.profiler.ProfilerActivity.CPU,
    #     ],
    #     record_shapes=True,
    #     profile_memory=True,
    #     with_stack=True,
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler("./log", use_gzip=True),
    # ) as prof:
    #     prof.step()
    #     # with torch.profiler.record_function("recall_test"):
    vectors = generate_random_vectors(1000)
    print("ANN >>>>>")
    ann_recall_test(vectors)
    # print("HNSW >>>>>")
    # hnsw_recall_test(vectors)
    # print("LAYERED ANN >>>>>")
    # layered_ann_recall_test(vectors)
