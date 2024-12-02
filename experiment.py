from torch import Tensor
import torch
import sys

MAXINT = sys.maxsize
MAXFLOAT = 10.0


class Queue:
    def __init__(self, num_queues: int, queue_length: int, capacity: int):
        self.length = queue_length
        self.indices = torch.full((num_queues, capacity), MAXINT)
        self.distances = torch.full((num_queues, capacity), MAXFLOAT)

    def insert(self, vector_id_batch: Tensor, distances_batch: Tensor):
        buf = torch.narrow_copy(self.indices, 1, 0, self.length)
        (batches, nhs) = vector_id_batch.size()
        indices_tail = self.indices.narrow(1, self.length, nhs)
        distances_tail = self.distances.narrow(1, self.length, nhs)
        indices_tail.copy_(vector_id_batch)
        distances_tail.copy_(distances_batch)
        (self.indices, self.distances) = queue_sort(self.indices, self.distances)

    def print(self):
        (_bs, dim) = self.indices.size()
        print(self.indices.narrow(1, 0, self.length))
        print(self.distances.narrow(1, 0, self.length))
        print("----------------")
        print(self.indices.narrow(1, self.length, dim))
        print(self.distances.narrow(1, self.length, dim))


def comparison(qvs, nvs):
    print(nvs)
    batch_size, queue_size, vector_dim = nvs.size()
    return (1 - nvs.bmm(qvs.resize(batch_size, 1, vector_dim).transpose(1, 2))) / 2


def search_from_seeds(
    query_vecs: Tensor,
    neighborhood_indices: Tensor,
    neighborhoods: Tensor,
    vectors: Tensor,
):
    (dim1, dim2) = neighborhood_indices.size()
    (batch_size, nhs) = neighborhoods.size()
    (_, vector_dimension) = vectors.size()

    index_list = neighborhoods.index_select(0, neighborhood_indices.flatten()).flatten()
    indexes_of_comparisons = index_list.view(dim1, dim2 * nhs)
    vectors_for_comparison = vectors.index_select(0, index_list).view(
        dim1, dim2 * nhs, vector_dimension
    )
    # return (query_vecs, vectors_for_comparison)
    distances_from_comparison = comparison(query_vecs, vectors_for_comparison)
    return (indexes_of_comparisons, distances_from_comparison)


def closest_vectors():
    pass


def punch_out_duplicates(ids: Tensor, distances: Tensor):
    dim1, dim2 = ids.size()
    shifted_ids = torch.hstack([ids[:, 1:], torch.full([dim1, 1], MAXINT)])
    mask = ids == shifted_ids
    ids[mask] = MAXINT
    distances[mask] = MAXFLOAT
    return index_sort(ids, distances)


def index_by_tensor(a: Tensor, b: Tensor):
    dim1, dim2 = a.size()
    a = a[
        torch.arange(dim1).unsqueeze(1).expand((dim1, dim2)).flatten(), b.flatten()
    ].view(dim1, dim2)
    return a


def index_sort(neighborhoods: Tensor, neighborhood_distances: Tensor):
    (ns, indices) = neighborhoods.sort(1)

    nds = index_by_tensor(neighborhood_distances, indices)

    (nds, indices) = nds.sort(stable=True)

    ns = index_by_tensor(ns, indices)

    return (ns, nds)


def queue_sort(neighborhoods: Tensor, neighborhood_distances: Tensor):
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
        ]
    )

    neighborhoods = torch.tensor(
        [[4, 6], [4, 5], [6, 7], [5, 7], [0, 1], [1, 3], [2, 0], [2, 3]]
    )

    return (vs, neighborhoods)


def two_hop(ns: Tensor):
    dim1, dim2 = ns.size()
    return ns.index_select(0, ns.flatten()).flatten().view(dim1, dim2 * dim2)


def distances(vs: Tensor, neighborhoods: Tensor):
    _, vec_dim = vs.size()
    dim1, dim2 = neighborhoods.size()
    qvi = torch.arange(dim1)
    qvs = vs.index_select(0, qvi).t().reshape(dim1, vec_dim, 1)
    nvs = vs.index_select(0, neighborhoods.flatten()).view(dim1, dim2, vec_dim)
    return comparison(qvs, nvs)


def prune(neighborhood: Tensor, neighborhood_distances: Tensor, vectors: Tensor):
    two_hop_neighbors = two_hop(neighborhood)

    pass


def primes():
    return torch.tensor([1, 3, 7, 9])


def generate_circulant_neighborhoods(num_vecs: Tensor, primes: Tensor):
    indices = torch.arange(num_vecs)
    (nhs,) = primes.size()
    repeated_indices = indices.expand(nhs, num_vecs).transpose(0, 1)
    repeated_primes = primes.expand(num_vecs, nhs)
    circulant_neighbors = repeated_indices + repeated_primes
    return circulant_neighbors.sort().values % num_vecs


def generate_hnsw():
    """ """
    pass


def generate_layer(primes: Tensor, vs: Tensor) -> Tensor:
    (num_vecs, vec_dim) = vs.size()
    nhi = generate_circulant_neighborhoods(num_vecs, primes)
    (dim1, dim2) = nhi.size()
    nhd = distances(vs, nhi)

    nhv = vs.index_select(0, nhi.flatten()).resize(dim1, dim2)
    pass


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
