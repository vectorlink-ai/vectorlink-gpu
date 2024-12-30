from torch import Tensor
import torch
from .constants import PRIMES, DEVICE, DEBUG, MAXINT
from .kernels import punchout_duplicates_, dedup_tensor_, index_by_tensor

import time


def generate_random_vectors(number_of_vectors: int, dimensions: int = 1536) -> Tensor:
    vectors = torch.nn.functional.normalize(
        torch.randn(number_of_vectors, dimensions, dtype=torch.float32), dim=1
    )
    return vectors


def primes(size: int):
    return PRIMES.narrow(0, 0, size)


def generate_circulant_beams(num_vecs: int, primes: Tensor) -> Tensor:
    indices = torch.arange(num_vecs, device=DEVICE, dtype=torch.int32)
    (nhs,) = primes.size()
    repeated_indices = indices.expand(nhs, num_vecs).transpose(0, 1)
    repeated_primes = primes.expand(num_vecs, nhs)
    circulant_neighbors = repeated_indices + repeated_primes
    return circulant_neighbors.sort().values % num_vecs


def index_sort(beams: Tensor, beam_distances: Tensor):
    (ns, indices) = beams.sort(1)

    nds = index_by_tensor(beam_distances, indices)

    (nds, indices) = nds.sort(dim=1, stable=True)

    ns = index_by_tensor(ns, indices)

    return (ns, nds)


def queue_sort(beams: Tensor, beam_distances: Tensor):
    (ns, nds) = index_sort(beams, beam_distances)
    return punchout_duplicates_(ns, nds)


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
        "We still have space"
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


def dedup_sort(tensor: Tensor):
    (tensor, _) = tensor.sort()
    dedup_tensor_(tensor)
    (tensor, _) = tensor.sort()
    return tensor
