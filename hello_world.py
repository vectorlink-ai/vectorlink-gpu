import torch
import numba

import numpy as np
from numba import cuda, gdb_init, void, float32, int64, int32

import sys


@cuda.jit
def distance(vector, query_vector, vector_idx):
    if vector_idx == 0:
        pass
    pass


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
    vector_idx = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
    result = sum(vec, dim, vector_idx)
    numba.cuda.syncthreads()
    if vector_idx == 0:
        out[0] = result


@cuda.jit(float32(float32[::1], float32[::1], float32[::1], int64, int64), device=True)
def cosine_distance(vec1, vec2, buf, vector_dimension, idx):
    buf[idx] = vec1[idx] * vec2[idx]
    numba.cuda.syncthreads()
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


@cuda.jit(void(int32[::1]))
def silly_kernel(vec):
    vec[0] = 1


@cuda.jit(void(float32[::1], float32[::1], float32[::1]))
def cosine_distance_kernel(vec1, vec2, out):
    scratch_buffer = numba.cuda.shared.array(1536 * 4, numba.float32)
    dimension = vec1.shape[0]
    vector_idx = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
    if vector_idx < dimension:
        result = cosine_distance(vec1, vec2, scratch_buffer, dimension, vector_idx)
        if vector_idx == 0:
            out[0] = result


INT32_MAX = int32(2147483647)
FLOAT32_MAX = float32(3.4028235e38)


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
def distance_from_seeds(
    query_vecs, neighbors_to_visit, neighborhoods, vectors, index_out, distance_out
):
    distance_buffer = numba.cuda.shared.array(4 * 1536, float32)

    queue_idx = cuda.blockIdx.x
    neighborhood_idx = cuda.blockIdx.y
    vector_idx = cuda.threadIdx.x
    batch_idx = cuda.blockIdx.z
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
            index_out[batch_idx, queue_idx] = INT32_MAX
            distance_out[batch_idx, queue_idx] = FLOAT32_MAX
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


def do_dot_product():
    x = torch.tensor([1.0, 2.0])
    y = torch.tensor([3.0, 1.0])
    scratch = torch.tensor([0.0, 0.0])

    return (dot_product(x, y, scratch),)


def main():
    print(do_dot_product())


import experiment


def run_cuda():
    (vecs, neighborhoods) = experiment.example_db()
    qvs = vecs.index_select(
        0, torch.tensor([4, 6, 0], dtype=torch.int32, device="cuda")
    )
    queue = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.int32, device="cuda")

    # grid = (1, 1, 1)
    # block = (1, 1, 1)
    # result = torch.tensor([1], dtype=torch.int32, device="cuda")
    # silly_kernel[grid, block](result)
    # print("silly kernel result")
    # print(result)
    # grid = (1, 1, 1)
    # block = (4, 1, 1)
    # vec_in = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32, device="cuda")
    # out = torch.tensor([0.0], dtype=torch.float32, device="cuda")
    # sum_kernel[grid, block](vec_in, out)
    # print(out)
    # sys.exit(0)

    # v1 = vecs[0]
    # v2 = vecs[3]
    # grid = (1, 1, 1)
    # block = (2, 1, 1)
    # result = torch.tensor([4.0], dtype=torch.float32, device="cuda")
    # cosine_distance_kernel[grid, block](v1, v2, numba.cuda.as_cuda_array(result))
    # print("cosine kernel result")
    # print(result)
    # sys.exit(0)

    index_out = torch.empty((3, 4), dtype=torch.int32, device="cuda")
    distance_out = torch.empty((3, 4), dtype=torch.float32, device="cuda")

    batches = 3
    vec_dim = 2
    queue_dim = 2
    neighborhood_size = 2
    float_size = 4

    grid = (queue_dim, neighborhood_size, batches)
    block = (vec_dim, 1, 1)
    distance_from_seeds[grid, block, None, vec_dim * float_size](
        qvs, queue, neighborhoods, vecs, index_out, distance_out
    )
    print(index_out)
    print(distance_out)


if __name__ == "__main__":
    # main()
    # gdb_init()
    run_cuda()
