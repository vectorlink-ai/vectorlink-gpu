from torch import Tensor
import torch
import sys

MAXINT = sys.maxsize
MAXFLOAT = 10.0


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
