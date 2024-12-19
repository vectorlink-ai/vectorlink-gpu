from typing import List, Optional
from torch import Tensor
import torch
from torch import profiler
import sys
from datetime import datetime
import time
import sys
import os
import subprocess
import argparse
from typing import Dict
import json

import numba
from numba import cuda, gdb_init, void, float32, int64, int32

from torch.cuda import Stream

from primes import PRIMES

# This gives more headroom, but is harder to read
MAXINT = 2147483647  # 2**31-1
MAXFLOAT = 3.4028e37
DEVICE = "cuda"


def timed(fn):
    wall_start = time.time()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    torch.cuda.synchronize()
    end.record()
    torch.cuda.synchronize()
    wall_end = time.time()
    return result, start.elapsed_time(end) / 1_000, (wall_end - wall_start)


# print(f"[{datetime.now().strftime('%H:%M:%S.%f')}] {msg}")
CLOSEST_VECTORS_BATCH_TIME = 0.0


def log_time(func):
    global CLOSEST_VECTORS_BATCH_TIME

    def wrapper(*args, **kwargs):
        global CLOSEST_VECTORS_BATCH_TIME

        def closure():
            return func(*args, **kwargs)

        (result, cuda_time, wall_time) = timed(closure)
        print(f"[{func.__name__}]\n\tCUDA time: {cuda_time}\n\tWALL time {wall_time}")
        if func.__name__ == "closest_vectors":
            CLOSEST_VECTORS_BATCH_TIME = max(wall_time, CLOSEST_VECTORS_BATCH_TIME)
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
        output_update: bool = False,
    ):
        if output_update:
            buffers = torch.narrow_copy(self.indices, 1, 0, self.length)
        (batches, size_per_batch) = vector_id_batch.size()
        indices_tail = self.indices.narrow(1, self.length, size_per_batch)
        distances_tail = self.distances.narrow(1, self.length, size_per_batch)
        indices_tail.copy_(vector_id_batch)
        distances_tail.copy_(distances_batch)

        if exclude is not None:
            punchout_excluded_(indices_tail, distances_tail, exclude)

        (self.indices, self.distances) = queue_sort(self.indices, self.distances)

        if output_update:
            did_something_mask = self.indices.narrow(1, 0, self.length) != buffers
            return did_something_mask

        return None

    def insert_random_padded(
        self,
        vector_id_batch: Tensor,
        distances_batch: Tensor,
        exclude: Optional[Tensor] = None,  # formatted like [[0],[1],[2]]
    ):
        (batch_size, total_length) = vector_id_batch.size()
        (total_batch_size, _) = self.indices.size()
        difference = total_batch_size - batch_size
        random_padding = generate_circulant_beams(difference, primes(total_length))
        random_distances = calculate_distances(vectors, random_padding)
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


@cuda.jit(void(float32[::1], int64), device=True, inline=True, fastmath=True)
def warp_reduce(vec, thread_idx):
    if thread_idx < 32:
        vec[thread_idx] += vec[thread_idx + 32]
        vec[thread_idx] += vec[thread_idx + 16]
        vec[thread_idx] += vec[thread_idx + 8]
        vec[thread_idx] += vec[thread_idx + 4]
        vec[thread_idx] += vec[thread_idx + 2]
        vec[thread_idx] += vec[thread_idx + 1]


@cuda.jit(float32(float32[::1], int64, int64), device=True, inline=True, fastmath=True)
def sum(vec, dim, idx):
    # assert dim > 1024

    groups = int((dim + 1023) / 1024)
    for group in range(1, groups):
        inner_idx = 1024 * group + idx
        if inner_idx >= dim:
            break
        vec[idx] += vec[inner_idx]
    numba.cuda.syncthreads()

    if idx < 512:
        vec[idx] += vec[idx + 512]
    else:
        return 0.0
    numba.cuda.syncthreads()
    if idx < 256:
        vec[idx] += vec[idx + 256]
    else:
        return 0.0
    numba.cuda.syncthreads()
    if idx < 128:
        vec[idx] += vec[idx + 128]
    else:
        return 0.0
    numba.cuda.syncthreads()
    if idx < 64:
        vec[idx] += vec[idx + 64]
    else:
        return 0.0
    numba.cuda.syncthreads()

    warp_reduce(vec, idx)
    return vec[0]


@cuda.jit(void(float32[::1], float32[::1]))
def sum_kernel(vec, out):
    dim = vec.shape[0]
    vector_idx_group = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
    result = sum(vec, dim, vector_idx_group)
    numba.cuda.syncthreads()
    if vector_idx_group == 0:
        out[0] = result


@cuda.jit(
    float32(float32[::1], float32[::1], float32[::1], int64, int64),
    device=True,
    inline=True,
    fastmath=True,
)
def cosine_distance(vec1, vec2, buf, vector_dimension, idx):
    groups = int((vector_dimension + 1023) / 1024)
    for group_offset in range(0, groups):
        inner_idx = idx * groups + group_offset
        if inner_idx >= vector_dimension:
            break

        buf[inner_idx] = vec1[inner_idx] * vec2[inner_idx]

    numba.cuda.syncthreads()
    if groups > 1:
        buf[idx] = buf[idx]
    numba.cuda.syncthreads()
    # TODO sum will only work on vectors with dim >= 1024
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
        float32[:, ::1],
    )
)
def cuda_distances_kernel(vectors: Tensor, beams: Tensor, distances_out: Tensor):
    distance_buffer = numba.cuda.shared.array(4 * 1536, float32)

    vector_id = cuda.blockIdx.x
    beam_idx = cuda.blockIdx.y
    vector_group_idx = cuda.threadIdx.x

    vector_dimension = vectors.shape[1]

    neighbor_id = beams[vector_id, beam_idx]
    if neighbor_id == MAXINT:
        distances_out[vector_id, beam_idx] = MAXFLOAT
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
            "beam_idx: ",
            beam_idx,
            "neighbor_id: ",
            neighbor_id,
            "result: ",
            result,
        )
        """
        distances_out[vector_id, beam_idx] = result


@log_time
def calculate_distances(vectors: Tensor, beams: Tensor, distances_out=None):
    assert vectors.dtype == torch.float32
    assert vectors.is_contiguous()
    assert beams.dtype == torch.int32
    assert beams.is_contiguous()
    if distances_out:
        assert distances_out.is_contiguous()

    (vector_count, vector_dimension) = vectors.size()
    (batches, beam_size) = beams.size()

    assert batches == vector_count

    vector_idx_group_size = min(vector_dimension, 1024)
    float_size = 4

    grid = (vector_count, beam_size, 1)
    block = (vector_idx_group_size, 1, 1)

    if not distances_out:
        distances_out = torch.empty((vector_count, beam_size), dtype=torch.float32)

    cuda_distances_kernel[
        grid, block, numba_current_stream(), vector_dimension * float_size
    ](vectors, beams, distances_out)

    return distances_out


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
    query_vecs, neighbors_to_visit, beams, vectors, index_out, distance_out
):
    shared = numba.cuda.shared.array(0, float32)

    batch_idx = cuda.blockIdx.x
    queue_idx = cuda.blockIdx.y
    beam_idx = cuda.blockIdx.z

    vector_idx = cuda.threadIdx.x

    batch_size = query_vecs.shape[0]
    vector_count = vectors.shape[0]
    vector_dim = query_vecs.shape[1]
    beam_size = beams.shape[1]
    queue_size = neighbors_to_visit.shape[1]

    distance_buffer = shared[0:vector_dim]

    if vector_idx >= vector_dim or queue_idx >= queue_size or batch_idx >= batch_size:
        return

    assert batch_idx < batch_size
    assert queue_idx < queue_size
    node_id = neighbors_to_visit[batch_idx, queue_idx]
    if node_id == MAXINT:
        if vector_idx == 0:
            output_idx = queue_idx * beam_size + beam_idx
            index_out[batch_idx, output_idx] = MAXINT
            distance_out[batch_idx, output_idx] = MAXFLOAT
    elif node_id < vector_count:
        neighbor_vector_id = beams[node_id, beam_idx]
        assert neighbor_vector_id < vector_count
        vector = vectors[neighbor_vector_id]
        query_vector = query_vecs[batch_idx]
        result = cosine_distance(
            vector, query_vector, distance_buffer, vector_dim, vector_idx
        )
        if vector_idx == 0:
            output_idx = queue_idx * beam_size + beam_idx
            index_out[batch_idx, output_idx] = neighbor_vector_id
            distance_out[batch_idx, output_idx] = result
    else:
        assert False


def cuda_search_from_seeds(
    query_vecs: Tensor,
    neighbors_to_visit: Tensor,
    beams: Tensor,
    vectors: Tensor,
) -> (Tensor, Tensor):

    assert query_vecs.dtype == torch.float32
    assert query_vecs.is_contiguous()
    assert neighbors_to_visit.dtype == torch.int32
    assert neighbors_to_visit.is_contiguous()
    assert beams.dtype == torch.int32
    assert beams.is_contiguous()
    assert vectors.is_contiguous()
    assert vectors.dtype == torch.float32

    (batch_size, visit_length) = neighbors_to_visit.size()
    (_, beam_size) = beams.size()
    (_, vector_dimension) = vectors.size()
    # TODO 1024 should probably be queried instead
    number_of_groups = int((vector_dimension + 1023) / 1024)
    vector_idx_group_size = int(
        (vector_dimension + number_of_groups - 1) / number_of_groups
    )
    float_size = 4
    # print(f"COUNT {COUNT}")

    grid = (batch_size, visit_length, beam_size)
    block = (vector_idx_group_size, 1, 1)

    # index_out = torch.empty(
    #     (batch_size, visit_length * beam_size), dtype=torch.int32, device=DEVICE
    # )

    index_out = torch.full(
        (batch_size, visit_length * beam_size),
        MAXINT,
        dtype=torch.int32,
        device=DEVICE,
    )

    # distance_out = torch.empty(
    #     (batch_size, visit_length * beam_size),
    #     dtype=torch.float32,
    #     device=DEVICE,
    # )
    distance_out = torch.full(
        (batch_size, visit_length * beam_size),
        MAXFLOAT,
        dtype=torch.float32,
        device=DEVICE,
    )

    cuda_search_from_seeds_kernel[
        grid,
        block,
        numba_current_stream(),
        vector_dimension * float_size,
    ](query_vecs, neighbors_to_visit, beams, vectors, index_out, distance_out)
    return (index_out, distance_out)


@cuda.jit(void(int32[:, ::1]))
def mark_kernel(indices: Tensor):
    queue_len = indices.shape[1]

    batch_idx = cuda.blockIdx.x
    queue_group_idx = cuda.threadIdx.x

    groups = int((queue_len + 1023) / 1024)
    for group in range(0, groups):
        queue_idx = 1024 * group + queue_group_idx
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
    queue_group_idx = cuda.threadIdx.x

    groups = int((queue_len + 1023) / 1024)
    for group in range(0, groups):
        queue_idx = 1024 * group + queue_group_idx
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
    queue_group_idx = cuda.threadIdx.x

    groups = int((queue_len + 1023) / 1024)
    for group in range(0, groups):
        queue_idx = 1024 * group + queue_group_idx
        if queue_idx < queue_len:
            flag = int32(1) << 30
            bitmask = ~flag
            value = indices[batch_idx, queue_idx]
            if value & flag != 0:
                indices[batch_idx, queue_idx] = MAXINT


def dedup_tensor_(ids: Tensor):
    (batch_size, queue_size) = ids.size()
    assert ids.is_contiguous()
    assert ids.dtype == torch.int32

    queue_size = min(queue_size, 1024)

    grid = (batch_size, 1, 1)
    block = (queue_size, 1, 1)
    mark_kernel[grid, block, numba_current_stream(), 0](ids)
    punchout_kernel[grid, block, numba_current_stream(), 0](ids)


def dedup_tensor_pair_(ids: Tensor, distances: Tensor):
    """Assumes sorted tensors!"""
    (batch_size, queue_size) = ids.size()
    (batch_size2, queue_size2) = distances.size()
    assert batch_size == batch_size2
    assert queue_size == queue_size2
    assert ids.is_contiguous()
    assert distances.is_contiguous()

    queue_size = min(queue_size, 1024)

    grid = (batch_size, 1, 1)
    block = (queue_size, 1, 1)
    mark_kernel[grid, block, numba_current_stream(), 0](ids)
    punchout_pair_kernel[grid, block, numba_current_stream(), 0](ids, distances)


def comparison(qvs, nvs):
    with profiler.record_function("comparison"):
        batch_size, queue_size, vector_dim = nvs.size()
        results = (
            1 - nvs.bmm(qvs.reshape(batch_size, 1, vector_dim).transpose(1, 2))
        ) / 2
        return results.reshape(batch_size, queue_size)


def search_from_seeds(
    query_vecs: Tensor,
    neighbors_to_visit: Tensor,
    beams: Tensor,
    vectors: Tensor,
):
    with profiler.record_function("search_from_seeds"):
        return cuda_search_from_seeds(query_vecs, neighbors_to_visit, beams, vectors)


def test_semantics():
    torch.set_default_device(DEVICE)
    s1 = Stream()
    s2 = Stream()
    number_of_vectors = 1000
    dimensions = 1536
    with torch.cuda.stream(s1):
        A = torch.nn.functional.normalize(
            torch.torch.randn(number_of_vectors, dimensions, dtype=torch.float32), dim=1
        )
    s2.wait_stream(s1)
    with torch.cuda.stream(s2):
        A.square_()
        torch.torch.randn(number_of_vectors * 1_000, dimensions, dtype=torch.float32)
        B = A.sum(dim=1)
    torch.cuda.default_stream().wait_stream(s2)
    print(B)


@log_time
def closest_vectors(
    query_vecs: Tensor,
    search_queue: Queue,
    vectors: Tensor,
    beams: Tensor,
    config: Dict,
    exclude: Optional[Tensor] = None,
):
    with profiler.record_function("closest_vectors"):
        (_, vector_dimension) = vectors.size()
        (beam_count, beam_size) = beams.size()
        extra_capacity = beam_size * config["parallel_visit_count"]
        (batch_size, queue_capacity) = search_queue.size()
        visit_queue_len = config["visit_queue_factor"] * config["beam_size"]
        capacity = visit_queue_len + extra_capacity
        visit_queue = Queue(batch_size, visit_queue_len, capacity)
        visit_queue.initialize_from_queue(search_queue)
        exclude_len = config["exclude_factor"] * visit_queue_len

        did_something = torch.full([batch_size], True)
        seen = torch.full(
            (batch_size, exclude_len),
            MAXINT,
            dtype=torch.int32,
        )

        progressing = True
        count = 0
        while progressing:
            count += 1
            checking_output = (count % config["speculative_readahead"]) == 0
            neighbors_to_visit = visit_queue.pop_n_ids(config["parallel_visit_count"])

            (indexes_of_comparisons, distances_of_comparisons) = search_from_seeds(
                query_vecs, neighbors_to_visit, beams, vectors
            )

            did_something = search_queue.insert(
                indexes_of_comparisons,
                distances_of_comparisons,
                exclude=exclude,
                output_update=checking_output,
            )

            if seen is not None:
                seen = add_new_to_seen_(seen, neighbors_to_visit)

            if seen is not None:
                visit_queue.insert(
                    indexes_of_comparisons, distances_of_comparisons, exclude=seen
                )

            if checking_output:
                progressing = bool(torch.any(did_something))

    # torch.cuda.default_stream().wait_stream(search_queue_stream)

    return True


def search_layers(
    layers: List[Tensor],
    query_vecs: Tensor,
    search_queue: Queue,
    vectors: Tensor,
    config: Dict,
):
    (number_of_batches, _) = query_vecs.size()
    # we don't exclude everything, since we're starting from actual query vecs, not indices
    for layer in layers:
        closest_vectors(query_vecs, search_queue, vectors, layer, config, None)


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


def punch_out_duplicates_(ids: Tensor, distances: Tensor):
    with profiler.record_function("punch_out_duplicates"):
        dedup_tensor_pair_(ids, distances)
        (distances, perm) = distances.sort()
        ids = index_by_tensor(ids, perm)
        assert ids.dtype == torch.int32
        assert distances.dtype == torch.float32
        return (ids, distances)


# NOTE: Potentially can be made a kernel
def index_by_tensor(a: Tensor, b: Tensor):
    with profiler.record_function("index_by_tensor"):
        dim1, dim2 = a.size()
        a = a[
            torch.arange(dim1).unsqueeze(1).expand((dim1, dim2)).flatten(), b.flatten()
        ].view(dim1, dim2)
        return a


def index_sort(beams: Tensor, beam_distances: Tensor):
    with profiler.record_function("index_sort"):
        (ns, indices) = beams.sort(1)

        nds = index_by_tensor(beam_distances, indices)

        (nds, indices) = nds.sort(dim=1, stable=True)

        ns = index_by_tensor(ns, indices)

        return (ns, nds)


def queue_sort(beams: Tensor, beam_distances: Tensor):
    with profiler.record_function("queue_sort"):
        (ns, nds) = index_sort(beams, beam_distances)
        return punch_out_duplicates_(ns, nds)


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

    beams = torch.tensor(
        [[4, 6], [4, 5], [6, 7], [5, 7], [0, 1], [1, 3], [0, 2], [2, 3]],
        dtype=torch.int32,
        device="cuda",
    )

    return (vs, beams)


def two_hop(ns: Tensor):
    dim1, dim2 = ns.size()
    return ns.index_select(0, ns.flatten()).flatten().view(dim1, dim2 * dim2)


def primes(size: int):
    return PRIMES.narrow(0, 0, size)


def generate_circulant_beams(num_vecs: int, primes: Tensor) -> Tensor:
    indices = torch.arange(num_vecs, device=DEVICE, dtype=torch.int32)
    (nhs,) = primes.size()
    repeated_indices = indices.expand(nhs, num_vecs).transpose(0, 1)
    repeated_primes = primes.expand(num_vecs, nhs)
    circulant_neighbors = repeated_indices + repeated_primes
    return circulant_neighbors.sort().values % num_vecs


def generate_layered_ann(vectors: Tensor, config: Dict):
    """ """
    beam_size = config["beam_size"]
    layers = []
    (count, _) = vectors.size()
    order = beam_size * 3
    size = 10_000
    while True:
        bound = min(size, count)
        (ann, _) = generate_ann(vectors[0:bound], config)
        layers.append(ann)
        if size >= count:
            break
        else:
            size = order * size

    return layers


def generate_hnsw(vectors: Tensor, config: Dict):
    """ """
    (vector_count, _) = vectors.size()
    beam_size = config["beam_size"]
    order = beam_size * 3
    layer_size = order
    bound = min(layer_size, vector_count)
    (ann, distances) = generate_ann(beam_size, vectors[0:bound])

    layers = [ann]
    queue_length = beam_size * config["beam_queue_factor"]
    remaining_capacity = queue_length * config["parallel_visit_count"]

    queue = Queue(
        bound,
        queue_length,
        queue_length + remaining_capacity,
    )  # make que from beams + neigbhorhood_distances
    # print("neigbhorhoods")
    # print(beams)
    # print("beam distances")
    # print(beam_distances)

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

        ann = queue.indices.narrow_copy(1, 0, beam_size)
        print(ann)
        print(f"Isolated layer recall")
        ann_calculate_recall(vectors[:bound], ann)
        distances = queue.distances.narrow_copy(1, 0, beam_size)

        layers.append(ann)
        c += 1

    return layers


def generate_ann(vectors: Tensor, config: Dict) -> Tensor:
    # print_timestamp("generating ann")
    (num_vecs, vec_dim) = vectors.size()
    beam_size = config["beam_size"]
    queue_length = beam_size * config["beam_queue_factor"]
    neighbor_primes = primes(queue_length)
    beams = generate_circulant_beams(num_vecs, neighbor_primes)
    assert beams.dtype == torch.int32
    # print_timestamp("circulant beams generated")
    beam_distances = calculate_distances(vectors, beams)

    # print_timestamp("distances calculated")
    # we want to be able to add a 'big' beam at the end, which happens to be queue_length
    remaining_capacity = queue_length * config["parallel_visit_count"]

    batch_size = config["batch_size"]
    number_of_batches = int((num_vecs + batch_size - 1) / batch_size)
    remaining = num_vecs

    for i in range(0, config["cagra_loops"]):
        print_timestamp(f"start of cagra loop {i}")
        next_beams = torch.empty_like(beams)
        next_beam_distances = torch.empty_like(beam_distances)
        for batch_count in range(0, number_of_batches):
            print_timestamp(f"  start of batch loop {batch_count}")
            batch_start_idx = batch_count * batch_size
            batch = min(batch_size, remaining)
            batch_end_idx = batch_count * batch_size + batch

            exclude = excluding_self(batch_start_idx, batch_end_idx, queue_length)

            queue = Queue(
                batch,
                queue_length,
                queue_length + remaining_capacity,
            )
            beams_slice = beams.narrow(0, batch_start_idx, batch)
            beam_distances_slice = beam_distances.narrow(0, batch_start_idx, batch)
            queue.insert(beams_slice, beam_distances_slice)

            closest_vectors(
                vectors[batch_start_idx:batch_end_idx],
                queue,
                vectors,
                beams,
                config,
                exclude,
            )

            next_beams_slice = next_beams.narrow(0, batch_start_idx, batch)
            next_beam_distance_slice = next_beam_distances.narrow(
                0, batch_start_idx, batch
            )
            queue_indices = queue.indices.narrow(1, 0, queue_length)
            queue_distances = queue.distances.narrow(1, 0, queue_length)
            next_beams_slice.copy_(queue_indices)
            next_beam_distance_slice.copy_(queue_distances)
            remaining -= batch

        remaining = num_vecs

        ## We prune here!
        if config["prune"]:
            (beams, distances) = prune_(next_beams, next_beam_distances)
        else:
            (beams, distances) = (
                next_beams,
                next_beam_distances,
            )

        prefix = min(vectors.size()[0], 1000)
        (found, recall) = ann_calculate_recall(
            vectors,
            beams.narrow_copy(1, 0, beam_size),
            config,
            sample=vectors[:prefix],
        )

        if recall == 1.0 or remaining <= 0:
            break
        print_timestamp(f"end of cagra loop {i}")
    return (
        beams.narrow_copy(1, 0, beam_size),
        beam_distances.narrow_copy(1, 0, beam_size),
    )


@cuda.jit(void(int32[:, :], float32[:, :], int32[:, :]))
def punchout_excluded_kernel(indices: Tensor, distances: Tensor, exclusions: Tensor):
    queue_size = indices.shape[1]
    exclusions_length = exclusions.shape[1]
    batch_idx = cuda.blockIdx.x
    queue_groups = int((queue_size + 1023) / 1024)
    queue_group_idx = cuda.threadIdx.x
    for group in range(0, queue_groups):
        queue_idx = queue_group_idx * queue_groups + group

        value = indices[batch_idx, queue_idx]
        for i in range(0, exclusions_length):
            exclude = exclusions[batch_idx, i]
            if exclude == MAXINT:
                break
            if value == exclude:
                indices[batch_idx, queue_idx] = MAXINT
                distances[batch_idx, queue_idx] = MAXFLOAT


def punchout_excluded_(indices, distances, exclusions):
    (batch_size, queue_size) = indices.size()
    (batch_size2, exclusion_size) = exclusions.size()
    assert batch_size == batch_size2
    queue_groups = int((queue_size + 1023) / 1024)
    queue_idx_size = int((queue_size + queue_groups - 1) / queue_groups)
    grid = (batch_size, 1, 1)
    block = (queue_idx_size, 1, 1)
    punchout_excluded_kernel[grid, block, numba_current_stream(), 0](
        indices, distances, exclusions
    )


def numba_current_stream():
    torch_current_stream = torch.cuda.current_stream()
    return numba.cuda.external_stream(torch_current_stream.cuda_stream)


def stream_to_numba(stream: Optional[Stream]):
    numba_stream = None
    if stream is not None:
        numba_stream = numba.cuda.external_stream(stream.cuda_stream)

    return numba_stream


def initial_queue(vectors: Tensor, config: Dict):
    queue_size = config["recall_search_queue_factor"] * config["beam_size"]
    (batch_size, _) = vectors.size()
    extra_capacity = max(
        queue_size, config["parallel_visit_count"] * config["beam_size"]
    )
    queue = Queue(batch_size, queue_size, queue_size + extra_capacity)
    p = primes(queue_size)
    initial_queue_indices = generate_circulant_beams(batch_size, p)
    d = calculate_distances(vectors, initial_queue_indices)
    queue.insert(initial_queue_indices, d)

    return queue


def ann_calculate_recall(
    vectors: Tensor,
    beams: Tensor,
    config: Dict,
    sample: Optional[Tensor] = None,
):
    if sample is None:
        sample = vectors
    (sample_size, _) = sample.size()
    (_, beam_size) = beams.size()

    queue = initial_queue(sample, config)
    # print_timestamp("queues allocated")

    closest_vectors(sample, queue, vectors, beams, config)
    # print_timestamp("closest vectors calculated")
    expected = torch.arange(sample_size)
    actual = queue.indices.t()[0]
    found = (expected == actual).sum().item()

    # print_timestamp("calculated recall")

    print(f"found: {found} / {sample_size}")
    print(f"recall: {found / sample_size}")
    return (found, found / sample_size)


def hnsw_calculate_recall(
    vectors: Tensor, hnsw: List[Tensor], config: Dict, sample: Optional[Tensor] = None
):
    if sample is None:
        sample = vectors
    (sample_size, _) = sample.size()

    (_, beam_size) = hnsw[-1].size()

    queue = initial_queue(sample, config)
    # print_timestamp("queues allocated")

    search_layers(hnsw, sample, queue, vectors, config)
    # print_timestamp("closest vectors calculated")
    expected = torch.arange(sample_size)
    actual = queue.indices.t()[0]
    found = (expected == actual).sum().item()

    # print_timestamp("calculated recall")

    print(f"found: {found} / {sample_size}")
    print(f"recall: {found / sample_size}")
    return (found, found / sample_size)


def ann_recall_test(vectors: Tensor, config: Dict):
    # print_timestamp("vectors allocated")
    (beams, _) = generate_ann(vectors, config)
    print_timestamp("=> ANN generated")

    prefix = min(vectors.size()[0], 10_000)
    return ann_calculate_recall(vectors, beams, config, sample=vectors[:prefix])


def layered_ann_recall_test(vectors: Tensor, config):
    # print_timestamp("vectors allocated")
    hnsw = generate_layered_ann(vectors, config)
    print_timestamp("=> Layered ANN generated")
    return hnsw_calculate_recall(vectors, hnsw, config)


def hnsw_recall_test(vectors: Tensor, config):
    # print_timestamp("vectors allocated")
    hnsw = generate_hnsw(vectors, config)
    for layer in hnsw:
        (a, b) = layer.size()
        print(f"shape: ({a}, {b})")

    print_timestamp("=> HNSW generated")

    hnsw_calculate_recall(vectors, hnsw, config)


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


def excluding_self(start, end, queue_length):
    exclude = torch.full((end - start, 1), MAXINT, dtype=torch.int32)
    exclude_front = exclude.narrow(1, 0, 1)
    excluded_ids = torch.arange(start, end, dtype=torch.int32).unsqueeze(1)
    exclude_front.copy_(excluded_ids)
    return exclude


@cuda.jit(void(int32[:, ::1], float32[:, ::1]))
def prune_kernel(beams, distances):
    (_, beam_size) = beams.shape
    node_x_id = cuda.blockIdx.x
    x_link_y = cuda.blockIdx.y
    y_link_z = cuda.threadIdx.y

    node_y_id = beams[node_x_id, x_link_y]
    if node_y_id == MAXINT:
        return

    node_z_id = beams[node_y_id, y_link_z]
    if node_z_id == MAXINT:
        return

    x_link_z = MAXINT
    for i in range(0, beam_size):
        if node_z_id == beams[node_x_id, i]:
            x_link_z = i

    if x_link_z == MAXINT:
        return

    distance_xz = distances[node_x_id, x_link_z]
    distance_xy = distances[node_x_id, x_link_y]
    distance_yz = distances[node_y_id, y_link_z]

    if (distance_xz > distance_xy) and (distance_xz > distance_yz):
        beams[node_x_id, x_link_z] = MAXINT
        distances[node_x_id, x_link_z] = MAXFLOAT


def prune_(beams, distances):
    """
    Remove detourable links. Mark out with out-of-band value
    if we can reach x->z by another (better) path

    i.e. if d_xz > d_xy and d_xz > d_yz

       prunable!
       |
    x  v   z
     . -> .
     ↓  ↗
      .
     y

    ..since we'll be able to find z via y anyhow.
    """
    assert beams.dtype == torch.int32
    assert distances.dtype == torch.float32

    (batch_size, beam_size) = beams.size()
    grid = (batch_size, beam_size, 1)
    block = (beam_size, 1, 1)
    prune_kernel[grid, block, numba_current_stream(), 0](beams, distances)

    (distances, permutation) = distances.sort()
    beams = index_by_tensor(beams, permutation)
    return (beams, distances)


def grid_search():
    torch.set_default_device(DEVICE)
    torch.set_float32_matmul_precision("high")
    config = {
        "type": "cagra",
        "vector_count": 1_000_000,
        "vector_dimension": 1536,
        "exclude_factor": 5,
        "batch_size": 10_000,
        "beam_queue_factor": 3,
        "recall_search_queue_factor": 6,
    }
    vectors = generate_random_vectors(
        number_of_vectors=config["vector_count"], dimensions=config["vector_dimension"]
    )

    os.makedirs("./grid-log", exist_ok=True)
    for prune in [True, False]:
        for beam_size in [24, 32, 46]:
            for parallel_visit_count in [1, 4]:
                for cagra_loops in [1, 2]:
                    for visit_queue_factor in [3, 4]:
                        config["prune"] = prune
                        config["beam_size"] = beam_size
                        config["parallel_visit_count"] = parallel_visit_count
                        config["visit_queue_factor"] = visit_queue_factor
                        config["cagra_loops"] = cagra_loops
                        result = main(vectors, config)
                        log_name = f"./grid-log/experiment-{time.time()}.log"
                        with open(log_name, "w") as w:
                            json.dump(result, w)


def main(vectors, configuration):
    start = time.time()
    recall = 0.0
    if configuration["type"] == "layered":
        print("LAYERED ANN >>>>>")
        (_, recall) = layered_ann_recall_test(vectors, configuration)
        print("<<<<< FINISHED LAYERED ANN")
    else:
        print("CAGRA ANN >>>>>")
        (_, recall) = ann_recall_test(vectors, configuration)
        print("<<<<< FINISHED CAGRA")
    end = time.time()

    configuration["construction_time"] = end - start
    configuration["recall"] = recall
    commit = subprocess.check_output(["git", "rev-parse", "--verify", "HEAD"])
    configuration["commit"] = commit.decode("utf-8").strip()
    gpu_arch = subprocess.check_output(["nvidia-smi", "-L"])
    configuration["gpu_arch"] = gpu_arch.decode("utf-8").strip()
    configuration["closest_vectors_batch_time"] = CLOSEST_VECTORS_BATCH_TIME

    return configuration


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--vector-count", help="number of vectors", type=int, default=10_000
    )
    parser.add_argument(
        "-d", "--vector-dimension", help="dimension of vectors", type=int, default=1536
    )
    parser.add_argument(
        "-L", "--layered", help="use layered cagra", default=False, action="store_true"
    )
    parser.add_argument(
        "-C", "--cagra", help="use cagra", default=False, action="store_true"
    )
    parser.add_argument(
        "-P", "--prune", help="prune beams", default=False, action="store_true"
    )
    parser.add_argument(
        "-n",
        "--beam_size",
        help="beam size (max number of neighbors)",
        type=int,
        default=24,
    )
    parser.add_argument("-l", "--cagra_loops", help="cagra loops", type=int, default=3)
    parser.add_argument(
        "-x",
        "--exclude_factor",
        help="factor larger than visit queue for exclusion buffer",
        type=int,
        default=5,
    )
    parser.add_argument(
        "-q",
        "--visit_queue_factor",
        help="factor larger than beam for visit queue",
        type=int,
        default=3,
    )
    parser.add_argument(
        "-f",
        "--beam_queue_factor",
        help="factor larger than beam for initial beams",
        type=int,
        default=3,
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        help="how many vectors/beams to run per batch",
        type=int,
        default=10_000,
    )
    parser.add_argument(
        "-r",
        "--recall_search_queue_factor",
        help="recall search queue factor",
        type=int,
        default=6,
    )
    parser.add_argument(
        "-p",
        "--parallel_visit_count",
        help="parallel visit count",
        type=int,
        default=3,
    )
    parser.add_argument(
        "-s",
        "--speculative_readahead",
        help="how far to try to read-ahead before checking for progress",
        type=int,
        default=5,
    )
    args = parser.parse_args()

    torch.set_default_device(DEVICE)
    torch.set_float32_matmul_precision("high")

    build_params = {
        "type": "layered" if args.layered else "cagra",
        "vector_count": args.vector_count,
        "vector_dimension": args.vector_dimension,
        "prune": args.prune,
        "beam_size": args.beam_size,
        "parallel_visit_count": args.parallel_visit_count,
        "beam_queue_factor": args.beam_queue_factor,
        "visit_queue_factor": args.visit_queue_factor,
        "exclude_factor": args.exclude_factor,
        "cagra_loops": args.cagra_loops,
        "batch_size": args.batch_size,
        "recall_search_queue_factor": args.recall_search_queue_factor,
        "speculative_readahead": args.speculative_readahead,
    }

    vectors = generate_random_vectors(
        number_of_vectors=args.vector_count, dimensions=args.vector_dimension
    )

    configuration = main(vectors, build_params)

    os.makedirs("./log", exist_ok=True)
    log_name = f"./log/experiment-{time.time()}.log"
    with open(log_name, "w") as w:
        print(configuration)
        json.dump(configuration, w)
