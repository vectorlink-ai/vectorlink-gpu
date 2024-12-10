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

MAXINT = 99
MAXFLOAT = 99.0
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
        # print(f"time spent: {time}")
        return result

    return wrapper


class Queue:
    def __init__(self, num_queues: int, queue_length: int, capacity: int):
        assert queue_length < capacity
        self.length = queue_length
        self.indices = torch.full((num_queues, capacity), MAXINT)
        self.distances = torch.full((num_queues, capacity), MAXFLOAT)
        self.buffers = torch.empty((num_queues, self.length))

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
        exclude: Optional[Tensor] = None,  # formatted like [[0],[1],[2]]
    ):
        self.buffers = torch.narrow_copy(self.indices, 1, 0, self.length)
        (batches, size_per_batch) = vector_id_batch.size()
        indices_tail = self.indices.narrow(1, self.length, size_per_batch)
        distances_tail = self.distances.narrow(1, self.length, size_per_batch)
        indices_tail.copy_(vector_id_batch)
        distances_tail.copy_(distances_batch)

        if exclude is not None:
            exclude_mask = indices_tail == exclude.expand(batches, size_per_batch)
            indices_tail[exclude_mask] = MAXINT
            distances_tail[exclude_mask] = MAXFLOAT

        (self.indices, self.distances) = queue_sort(self.indices, self.distances)
        did_something_mask = self.indices.narrow(1, 0, self.length) != self.buffers
        return did_something_mask

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


@cuda.jit(void(float32[::1], int64, int64), device=True)
def sum_part(vec, scale, idx):
    if idx < scale:
        vec[idx] += vec[idx + scale]


@cuda.jit(float32(float32[::1], int64, int64), device=True)
def sum(vec, dim, idx):
    while dim > 32:
        dim /= 2
        sum_part(vec, dim, idx)
    numba.cuda.syncthreads()
    result = 0.0
    if idx == 0:
        for i in range(0, dim):
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
        groups -= 1

    numba.cuda.syncthreads()
    # TODO sum will only work for vectors up to dim 2048 the way things are written now
    cos_theta = sum(buf, vector_dimension, idx)
    numba.cuda.syncthreads()
    result = 1234.0
    if idx == 0:
        result = (1 - cos_theta) / 2
        if result < 0.0:
            result = 0.0
        elif result > 1.0:
            result = 1.0
    return result


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

    node_id = neighbors_to_visit[batch_idx, queue_idx]
    if node_id > vector_count:
        if vector_idx == 0:
            index_out[batch_idx, queue_idx] = MAXINT
            distance_out[batch_idx, queue_idx] = MAXFLOAT
    else:
        neighbor_vector_id = neighborhoods[node_id, neighborhood_idx]
        vector = vectors[neighbor_vector_id]
        query_vector = query_vecs[batch_idx]
        result = cosine_distance(
            vector, query_vector, distance_buffer, vector_dim, vector_idx
        )
        if vector_idx == 0:
            output_idx = queue_idx * neighborhood_size + neighborhood_idx
            index_out[batch_idx, output_idx] = neighbor_vector_id
            distance_out[batch_idx, output_idx] = result


def cuda_search_from_seeds(
    query_vecs: Tensor,
    neighbors_to_visit: Tensor,
    neighborhoods: Tensor,
    vectors: Tensor,
) -> (Tensor, Tensor):
    (batch_size, visit_length) = neighbors_to_visit.size()
    (_, neighborhood_size) = neighborhoods.size()
    (_, vector_dimension) = vectors.size()
    # TODO 1024 should probably be queried instead
    vector_idx_group_size = max(int(vector_dimension / 2), 1024)
    float_size = 4

    grid = (batch_size, visit_length, neighborhood_size)
    block = (vector_idx_group_size, 1, 1)

    index_out = torch.tensor(
        (batch_size, visit_length * neighborhood_size), dtype=torch.int32, device=DEVICE
    )
    distance_out = torch.tensor(
        (batch_size, visit_length * neighborhood_size),
        dtype=torch.float32,
        device=DEVICE,
    )
    print(grid)
    print(block)

    cuda_search_from_seeds_kernel[grid, block, None, vector_dimension * float_size](
        query_vecs, neighbors_to_visit, neighborhoods, vectors, index_out, distance_out
    )
    return (index_out, distance_out)


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
        seen = exclude if exclude is not None else torch.empty(batch_size, 0)
        """
        Stream Diagram

        s1                       s2               s3              s4
        search_from_seeds--wait->|                |                |
        |                        |                |                |
        search_queue_inesrt   rowwise -- wait --->|                |
        |                        |                |                |
        |               indexes of comp    distances of comp       |
        |                        |----------------o----------wait->|
        |                        |                |                |
        |                        |<-wait----------|                |
        |                 visit queue insert      |             add_new_to_seen
        |                        |                |                |
        |<-wait------------------o----------------o-----------------
        |<-wait------------------|   U

        Stream DAG
                                s1
                                 |
                           search_from_seeds
                        s1/       \\s2
        search_queue_insert        rowwise______________
                     s1|             |s2                \\ s3
                       |    indexes of comparison_  ____distances of comparison
                       |             |           \\/                      |
                       |             |s2        s3/\\s4                   |s3
                       |       visit_queue_insert   add_new_to_seen       |
                      \\             |s2             |s4                 /
                        ------------------------------------------------
                                               |
                                          back to search from seeds
        """

        while torch.any(did_something):
            # print_timestamp("start of loop")
            # visit_queue.print()
            neighbors_to_visit = visit_queue.pop_n_ids(PARALLEL_VISIT_COUNT)
            (_, visit_length) = neighbors_to_visit.size()
            narrow_to = batch_size * visit_length * neighborhood_size
            # print_timestamp("popped")
            s1 = allocate_stream()
            s2 = allocate_stream()
            s3 = allocate_stream()
            s4 = allocate_stream()

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

            with current_stream(s2):
                mask = rowwise_isin(indexes_of_comparisons, seen)

            wait_stream(s3, s2)
            with current_stream(s2):
                indexes_of_comparisons[mask] = MAXINT
                record_stream(indexes_of_comparisons, s2)
            wait_stream(s4, s2)
            with current_stream(s3):
                distances_of_comparisons[mask] = MAXFLOAT
                record_stream(indexes_of_comparisons, s3)

            wait_stream(s2, s3)
            with current_stream(s2):
                visit_queue.insert(indexes_of_comparisons, distances_of_comparisons)

            with current_stream(s4):
                seen = add_new_to_seen(seen, indexes_of_comparisons)
                record_stream(seen, s4)

            wait_stream(s1, s2)
            wait_stream(s1, s3)
            wait_stream(s1, s4)


def search_layers(
    layers: List[Tensor], query_vecs: Tensor, search_queue: Queue, vectors: Tensor
):
    (number_of_batches, _) = query_vecs.size()
    # we don't exclude everything, since we're starting from actual query vecs, not indices
    exclude = torch.empty(number_of_batches, 0)
    for layer in layers:
        closest_vectors(query_vecs, search_queue, vectors, layer, exclude)


def search_from_initial():
    pass


def add_new_to_seen(seen, indices):
    seen = torch.concat([seen, indices], dim=1)
    return shrink_to_fit(seen)


def shrink_to_fit(seen):
    with profiler.record_function("shrink_to_fit"):
        (values, _) = seen.sort()
        (dim1, dim2) = seen.size()
        shifted = torch.hstack([values[:, 1:], torch.full([dim1, 1], MAXINT)])
        mask = values == shifted
        values[mask] = MAXINT
        (values, _) = values.sort()
        max_val_mask = values == MAXINT
        punched_mask = torch.all(max_val_mask, dim=0)
        match_indices = torch.arange(dim2)[punched_mask]
        (size,) = match_indices.size()
        if size > 0:
            max_col = torch.arange(dim2)[punched_mask][0]
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
        dim1, dim2 = ids.size()
        shifted_ids = torch.hstack([ids[:, 1:], torch.full([dim1, 1], MAXINT)])
        mask = ids == shifted_ids
        ids[mask] = MAXINT
        distances[mask] = MAXFLOAT
        return index_sort(ids, distances)


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
    ],
    dtype=torch.int32,
    device=DEVICE,
)


def primes(size: int):
    return PRIMES.narrow(0, 0, size)


def generate_circulant_neighborhoods(num_vecs: Tensor, primes: Tensor):
    indices = torch.arange(num_vecs, device=DEVICE)
    (nhs,) = primes.size()
    repeated_indices = indices.expand(nhs, num_vecs).transpose(0, 1)
    repeated_primes = primes.expand(num_vecs, nhs)
    circulant_neighbors = repeated_indices + repeated_primes
    return circulant_neighbors.sort().values % num_vecs


def generate_hnsw():
    """ """
    pass


NEIGHBORHOOD_QUEUE_FACTOR = 3
CAGRA_LOOPS = 3


def generate_ann(primes: Tensor, vectors: Tensor) -> Tensor:
    # print_timestamp("generating ann")
    (num_vecs, vec_dim) = vectors.size()
    neighborhoods = generate_circulant_neighborhoods(num_vecs, primes)
    # print_timestamp("circulant neighborhoods generated")
    (_, neighborhood_size) = neighborhoods.size()
    neighborhood_distances = distances(vectors, neighborhoods)
    # print_timestamp("distances calculated")
    queue_length = neighborhood_size * NEIGHBORHOOD_QUEUE_FACTOR
    # we want to be able to add a 'big' neighborhood at the end, which happens to be queue_length
    remaining_capacity = queue_length * PARALLEL_VISIT_COUNT
    queue = Queue(
        num_vecs,
        queue_length,
        queue_length + remaining_capacity,
    )  # make que from neighborhoods + neigbhorhood_distances
    # print_timestamp("queue allocated")
    # queue.print()
    # print("neigbhorhoods")
    # print(neighborhoods)
    # print("neighborhood distances")
    # print(neighborhood_distances)
    queue.insert(neighborhoods, neighborhood_distances)
    # print_timestamp("initial queue constructed from neighborhoods")
    exclude = torch.arange(num_vecs).unsqueeze(1)
    for i in range(0, CAGRA_LOOPS):
        # print_timestamp(f"start of cagra loop {i}")
        closest_vectors(vectors, queue, vectors, neighborhoods, exclude)
        # print_timestamp(f" closest vectors calculated")
        # print(f"cagra loop {i}")
        # queue.print()
        neighborhoods = queue.indices.narrow_copy(1, 0, queue_length)
        # print_timestamp(f"end of cagra loop {i}")
    return neighborhoods.narrow(1, 0, neighborhood_size)


"""
1.
Create circulants
2. Search for candidate neighborhoods
3. 
"""


"""
Example:

thv = 8 x 4 x 2
qvs = 8 x 2

qvst = qvs.t().reshape(8, 2, 1)

thv.bmm(qvst)

"""


def rowwise_isin(tensor_1, target_tensor):
    matches = tensor_1.unsqueeze(2) == target_tensor.unsqueeze(1)

    # result: boolean tensor of shape (N, K) where result[n, k] is torch.isin(tensor_1[n, k], target_tensor[n])
    result = torch.sum(matches, dim=2, dtype=torch.bool)

    return result


def initial_queue(vectors: Tensor, neighborhood_size: int, queue_size: int):
    (batch_size, _) = vectors.size()
    extra_capacity = max(queue_size, PARALLEL_VISIT_COUNT * neighborhood_size)
    queue = Queue(batch_size, queue_size, queue_size + extra_capacity)
    p = primes(queue_size)
    initial_queue_indices = generate_circulant_neighborhoods(batch_size, p)
    d = distances(vectors, initial_queue_indices)
    queue.insert(initial_queue_indices, d)

    return queue


def recall_test(
    number_of_vectors: int, dimensions: int = 1535, neighborhood_size: int = 24
):
    # print_timestamp("recall test starts")
    vectors = torch.nn.functional.normalize(
        torch.randn(number_of_vectors, dimensions), dim=1
    )
    # print_timestamp("vectors allocated")
    neighborhoods = generate_ann(primes(24), vectors)
    # print_timestamp("ann generated")
    queue = initial_queue(vectors, neighborhood_size, 3 * neighborhood_size)
    # print_timestamp("queues allocated")

    closest_vectors(vectors, queue, vectors, neighborhoods)
    # print_timestamp("closest vectors calculated")
    expected = torch.arange(number_of_vectors)
    actual = queue.indices.t()[0]
    found = (expected == actual).sum().item()

    # print_timestamp("calculated recall")

    print(found)
    print(found / number_of_vectors)


# Function to print timestamps
def print_timestamp(msg):
    pass


if __name__ == "__main__":
    torch.set_default_device(DEVICE)
    torch.set_float32_matmul_precision("high")
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CUDA,
            torch.profiler.ProfilerActivity.CPU,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./log", use_gzip=True),
    ) as prof:
        prof.step()
        # with torch.profiler.record_function("recall_test"):
        recall_test(2000)
