import torch
import numba
import taichi
from taichi import types as ty

import numpy as np
from numba import cuda


@taichi.kernel
def dot_product(
    x: ty.ndarray(dtype=taichi.f32, ndim=1),
    y: ty.ndarray(dtype=taichi.f32, ndim=1),
    scratch: ty.ndarray(dtype=taichi.f32, ndim=1),
) -> taichi.f32:
    for i in x:
        scratch[i] = x[i] * y[i]
    taichi.sync()
    acc = 0.0
    for i in scratch:
        acc += scratch[i]
    return acc


@cuda.jit
def distance(vector, query_vector, vector_idx):
    if vector_idx == 0:
        pass
    pass


@cuda.jit(
    """
void(float32[:],
     int64,
     int64
    )
""",
    device=True,
)
def sum_part(vec, scale, idx):
    if idx < scale:
        vec[idx] += vec[idx + scale]


@cuda.jit(
    """
float32(float32[:],
        int64,
        int64
    )
""",
    device=True,
)
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


@cuda.jit(
    """
float32(float32[:],
        float32[:],
        float32[:],
        int64,
        int64
    )
""",
    device=True,
)
def cosine_distance(vec1, vec2, buf, vector_dimension, idx):
    buf[idx] = vec1[idx] * vec2[idx]
    numba.cuda.syncthreads()
    cos_theta = sum(buf, vector_dimension, idx)
    numba.cuda.syncthreads()
    result = 0.0
    if idx == 0:
        if cos_theta < 0.0:
            cos_theta = 0.0
        elif cos_theta > 1.0:
            cos_theta = 1.0
        result = (1 - cos_theta) / 2
    return result


@cuda.jit(
    """
void(float32[:],
     float32[:],
     float32[:]
    )
"""
)
def cosine_distance_kernel(vec1, vec2, out):
    scratch_buffer = numba.cuda.shared.array(2, numba.float32)
    dimension = vec1.shape[0]
    vector_idx = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
    if vector_idx < dimension:
        out[0] = cosine_distance(vec1, vec2, scratch_buffer, dimension, vector_idx)


INT32_MAX = 99
F32_MAX = 99.9


# @cuda.jit(
#    """
# void(float32[:, :],
#     int32[:, :],
#     int32[:, :],
#     float32[:, :],
#     int32[:, :],
#     float32[:, :])
# """
# )
def distance_from_seeds(
    query_vecs, neighbors_to_visit, neighborhoods, vectors, index_out, distance_out
):

    neighbor_queue_idx = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
    vector_idx = cuda.blockDim.y * cuda.blockIdx.y + cuda.threadIdx.y
    batch_idx = cuda.blockDim.z * cuda.blockIdx.z + cuda.threadIdx.z
    batch_size = query_vecs.shape[0]
    vector_count = vectors.shape[0]
    vector_dim = query_vecs.shape[1]
    neighborhood_size = neighborhoods.shape[1]
    queue_size = neighbors_to_visit.shape[1]

    queue_idx = neighbor_queue_idx % neighborhood_size
    neighborhood_idx = int(neighbor_queue_idx / neighborhood_size)

    if vector_idx >= vector_dim or queue_idx >= queue_size or batch_idx >= batch_size:
        return

    node_id = neighbors_to_visit[batch_idx, queue_idx]
    if node_id > vector_count:
        if vector_idx == 0:
            index_out[batch_idx][queue_idx] = INT32_MAX
            distance_out[batch_idx][queue_idx] = F32_MAX
    else:
        neighbor_vector_id = neighborhoods[node_id, neighborhood_idx]
        vector = vectors[neighbor_vector_id]
        query_vector = query_vecs[batch_idx]
        if vector_idx == 0:
            index_out[batch_idx][queue_idx] = neighbor_vector_id
            distance_out[batch_idx][queue_idx] = cosine_distance(vector, query_vector)


def do_dot_product():
    x = torch.tensor([1.0, 2.0])
    y = torch.tensor([3.0, 1.0])
    scratch = torch.tensor([0.0, 0.0])

    return (dot_product(x, y, scratch),)


def main():
    taichi.init(arch=taichi.cuda)
    print(do_dot_product())


import experiment


def run_cuda():
    (vecs, neighborhoods) = experiment.example_db()
    qvs = vecs.index_select(0, torch.tensor([1, 3, 5]))
    queue = torch.tensor([[0, 3], [4, 5], [6, 7]])
    result = torch.empty(1, dtype=torch.float32)
    v1 = qvs[0]
    v2 = qvs[1]

    grid = (1, 1, 1)
    block = (2, 1, 1)
    cosine_distance_kernel[grid, block, None, 4 * 2](v1, v2)


if __name__ == "__main__":
    # main()
    run_cuda()
