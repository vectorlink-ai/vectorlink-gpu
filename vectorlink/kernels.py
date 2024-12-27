from torch import Tensor
from torch.cuda import Stream
import torch

from numba import cuda, gdb_init, void, float32, int64, int32
import numba

from typing import Optional, Tuple

from .constants import PRIMES, MAXINT, MAXFLOAT, DEVICE
from .utils import index_by_tensor
from .logging import log_time


def numba_current_stream():
    torch_current_stream = torch.cuda.current_stream()
    return numba.cuda.external_stream(torch_current_stream.cuda_stream)


def stream_to_numba(stream: Optional[Stream]):
    numba_stream = None
    if stream is not None:
        numba_stream = numba.cuda.external_stream(stream.cuda_stream)

    return numba_stream


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
    result = 1234.0  # intentionally a strange value so we can see results slipping
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


def search_from_seeds(
    query_vecs: Tensor, neighbors_to_visit: Tensor, beams: Tensor, vectors: Tensor
) -> Tuple[Tensor, Tensor]:

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

    grid = (batch_size, visit_length, beam_size)
    block = (vector_idx_group_size, 1, 1)

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


def punch_out_duplicates_(ids: Tensor, distances: Tensor):
    dedup_tensor_pair_(ids, distances)
    (distances, perm) = distances.sort()
    ids = index_by_tensor(ids, perm)
    assert ids.dtype == torch.int32
    assert distances.dtype == torch.float32
    return (ids, distances)


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


def symmetrize_kernel(beams, distances, output_beams, output_distances):
    pass


def symmetrize(beams: Tensor, distances: Tensor) -> Tuple[Tensor, Tensor]:
    """
    creates an approximate symmetrization for the graph.

    For a given node_id, scan the beams for this node_id, and add each beam origin which contains it.

    """
    (batch_size, queue_size) = beams.size()
    queue_groups = int((queue_size + 1023) / 1024)
    queue_idx_size = int((queue_size + queue_groups - 1) / queue_groups)
    grid = (batch_size, 1, 1)
    block = (queue_idx_size, 1, 1)
    symmetrize_kernel[grid, block, numba_current_stream(), 0](
        beams, distances, output_beams, output_distances
    )
