from torch import Tensor
import torch
from typing import Optional
from torch import profiler

from .constants import MAXINT, MAXFLOAT
from .kernels import (
    punchout_excluded_,
    calculate_distances,
    punchout_duplicates_,
)
from .utils import primes, generate_circulant_beams, index_sort


def queue_sort(beams: Tensor, beam_distances: Tensor):
    with profiler.record_function("queue_sort"):
        (ns, nds) = index_sort(beams, beam_distances)
        return punchout_duplicates_(ns, nds)


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
