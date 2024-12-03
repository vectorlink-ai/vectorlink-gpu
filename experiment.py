from typing import List, Optional
from torch import Tensor
import torch
import sys

MAXINT = 99
MAXFLOAT = 99.0


class Queue:
    def __init__(self, num_queues: int, queue_length: int, capacity: int):
        assert queue_length < capacity
        self.length = queue_length
        self.indices = torch.full((num_queues, capacity), MAXINT)
        self.distances = torch.full((num_queues, capacity), MAXFLOAT)

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
        bufs = torch.narrow_copy(self.indices, 1, 0, self.length)
        (batches, size_per_batch) = vector_id_batch.size()
        indices_tail = self.indices.narrow(1, self.length, size_per_batch)
        distances_tail = self.distances.narrow(1, self.length, size_per_batch)
        indices_tail.copy_(vector_id_batch)
        distances_tail.copy_(distances_batch)

        if not exclude is None:
            exclude_mask = indices_tail == exclude.expand(batches, size_per_batch)
            indices_tail[exclude_mask] = MAXINT
            distances_tail[exclude_mask] = MAXFLOAT

        (self.indices, self.distances) = queue_sort(self.indices, self.distances)
        did_something_mask = self.indices.narrow(1, 0, self.length) != bufs
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


def comparison(qvs, nvs):
    batch_size, queue_size, vector_dim = nvs.size()
    results = (1 - nvs.bmm(qvs.resize(batch_size, 1, vector_dim).transpose(1, 2))) / 2
    return results.resize(batch_size, queue_size)


def search_from_seeds(
    query_vecs: Tensor,
    neighbors_to_visit: Tensor,
    neighborhoods: Tensor,
    vectors: Tensor,
):
    (batch_size, visit_length) = neighbors_to_visit.size()
    (_, neighborhood_size) = neighborhoods.size()
    (_, vector_dimension) = vectors.size()

    filter_mask = neighbors_to_visit == MAXINT
    neighbors_to_visit[filter_mask] = 0  # set to 0 to get a valid element
    index_list = neighborhoods.index_select(0, neighbors_to_visit.flatten()).flatten()
    indexes_of_comparisons = index_list.view(
        batch_size, visit_length * neighborhood_size
    )
    vectors_for_comparison = vectors.index_select(0, index_list).view(
        batch_size, visit_length * neighborhood_size, vector_dimension
    )
    # return (query_vecs, vectors_for_comparison)
    distances_from_comparison = comparison(query_vecs, vectors_for_comparison)

    expanded_filter_mask = (
        filter_mask.reshape(batch_size, visit_length, 1)
        .expand(batch_size, visit_length, neighborhood_size)
        .reshape(batch_size, visit_length * neighborhood_size)
    )
    indexes_of_comparisons[expanded_filter_mask] = MAXINT
    distances_from_comparison[expanded_filter_mask] = MAXFLOAT
    return (indexes_of_comparisons, distances_from_comparison)


PARALLEL_VISIT_COUNT = 3
VISIT_QUEUE_LEN = 24 * 3


def closest_vectors(
    query_vecs: Tensor,
    search_queue: Queue,
    vectors: Tensor,
    neighborhoods: Tensor,
    exclude: Optional[Tensor] = None,
):
    (neighborhood_count, neighborhood_size) = neighborhoods.size()
    extra_capacity = neighborhood_size * PARALLEL_VISIT_COUNT
    (batch_size, queue_capacity) = search_queue.size()
    capacity = VISIT_QUEUE_LEN + extra_capacity
    visit_queue = Queue(batch_size, VISIT_QUEUE_LEN, capacity)
    visit_queue.initialize_from_queue(search_queue)
    did_something = torch.full([batch_size], True)
    seen = exclude if exclude is not None else torch.empty(batch_size, 0)
    while torch.any(did_something):
        # visit_queue.print()
        neighbors_to_visit = visit_queue.pop_n_ids(PARALLEL_VISIT_COUNT)
        (indexes_of_comparisons, distances_of_comparisons) = search_from_seeds(
            query_vecs,
            neighbors_to_visit,
            neighborhoods,
            vectors,
        )
        # Search queue
        did_something = search_queue.insert(
            indexes_of_comparisons, distances_of_comparisons, exclude=exclude
        )
        # Visit queue setup
        mask = rowwise_isin(indexes_of_comparisons, seen)
        # mask = torch.isin(indexes_of_comparisons, seen)
        indexes_of_comparisons[mask] = MAXINT
        distances_of_comparisons[mask] = MAXFLOAT

        visit_queue.insert(indexes_of_comparisons, distances_of_comparisons)

        seen = add_new_to_seen(seen, indexes_of_comparisons)


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
    max_col = nonzeroes[0].item()
    max_col = max_col - 1 if max_col > 0 else 0
    return seen.narrow(1, 0, max_col)


def punch_out_duplicates(ids: Tensor, distances: Tensor):
    dim1, dim2 = ids.size()
    shifted_ids = torch.hstack([ids[:, 1:], torch.full([dim1, 1], MAXINT)])
    mask = ids == shifted_ids
    ids[mask] = MAXINT
    distances[mask] = MAXFLOAT
    return index_sort(ids, distances)


# Does not appear to be stable :(
def index_by_tensor(a: Tensor, b: Tensor):
    dim1, dim2 = a.size()
    a = a[
        torch.arange(dim1).unsqueeze(1).expand((dim1, dim2)).flatten(), b.flatten()
    ].view(dim1, dim2)
    return a


def index_sort(neighborhoods: Tensor, neighborhood_distances: Tensor):
    (ns, indices) = neighborhoods.sort(1)

    nds = index_by_tensor(neighborhood_distances, indices)

    (nds, indices) = nds.sort(dim=1, stable=True)

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
        ],
        dtype=torch.float32,
    )

    neighborhoods = torch.tensor(
        [[4, 6], [4, 5], [6, 7], [5, 7], [0, 1], [1, 3], [0, 2], [2, 3]],
        dtype=torch.int32,
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
)


def primes(size: int):
    return PRIMES.narrow(0, 0, size)


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


NEIGHBORHOOD_QUEUE_FACTOR = 3
CAGRA_LOOPS = 1


def generate_ann(primes: Tensor, vectors: Tensor) -> Tensor:
    (num_vecs, vec_dim) = vectors.size()
    neighborhoods = generate_circulant_neighborhoods(num_vecs, primes)
    (_, neighborhood_size) = neighborhoods.size()
    neighborhood_distances = distances(vectors, neighborhoods)
    queue_length = neighborhood_size * NEIGHBORHOOD_QUEUE_FACTOR
    # we want to be able to add a 'big' neighborhood at the end, which happens to be queue_length
    remaining_capacity = queue_length * PARALLEL_VISIT_COUNT
    queue = Queue(
        num_vecs,
        queue_length,
        queue_length + remaining_capacity,
    )  # make que from neighborhoods + neigbhorhood_distances
    # queue.print()
    # print("neigbhorhoods")
    # print(neighborhoods)
    # print("neighborhood distances")
    # print(neighborhood_distances)
    queue.insert(neighborhoods, neighborhood_distances)
    exclude = torch.arange(num_vecs).unsqueeze(1)
    for i in range(0, CAGRA_LOOPS):
        closest_vectors(vectors, queue, vectors, neighborhoods, exclude)
        print(f"cagra loop {i}")
        # queue.print()
        neighborhoods = queue.indices.narrow_copy(1, 0, queue_length)
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
    vectors = torch.nn.functional.normalize(
        torch.randn(number_of_vectors, dimensions), dim=1
    )
    neighborhoods = generate_ann(primes(24), vectors)
    queue = initial_queue(vectors, neighborhood_size, 3 * neighborhood_size)

    closest_vectors(vectors, queue, vectors, neighborhoods)
    expected = torch.arange(number_of_vectors)
    actual = queue.indices.t()[0]
    found = (expected == actual).sum().item()

    print(found)
    print(found / number_of_vectors)
