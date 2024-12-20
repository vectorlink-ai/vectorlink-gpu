from typing import List, Optional, Tuple
from torch import Tensor
import torch
from torch import profiler
import sys
from datetime import datetime
import sys
import os
import subprocess
import argparse
from typing import Dict
import json

import numba
from numba import cuda, gdb_init, void, float32, int64, int32

from torch.cuda import Stream

from .constants import PRIMES, MAXINT, MAXFLOAT, DEVICE
from .queue import Queue
from .kernels import search_from_seeds, dedup_tensor_, calculate_distances
from .utils import (
    log_time,
    index_sort,
    index_by_tensor,
    generate_circulant_beams,
    primes,
    add_new_to_seen_,
)


@log_time
def closest_vectors(
    query_vecs: Tensor,
    search_queue: Queue,
    vectors: Tensor,
    beams: Tensor,
    config: Dict,
    exclude: Optional[Tensor] = None,
):

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


def generate_ann(vectors: Tensor, config: Dict) -> Tuple[Tensor, Tensor]:
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

    for i in range(0, config["optimization_loops"]):
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
                for optimization_loops in [1, 2]:
                    for visit_queue_factor in [3, 4]:
                        config["prune"] = prune
                        config["beam_size"] = beam_size
                        config["parallel_visit_count"] = parallel_visit_count
                        config["visit_queue_factor"] = visit_queue_factor
                        config["optimization_loops"] = optimization_loops
                        result = main(vectors, config)
                        log_name = f"./grid-log/experiment-{time.time()}.log"
                        with open(log_name, "w") as w:
                            json.dump(result, w)


class ANN:
    def __init__(
        self,
        vectors: Tensor,
        prune: bool = True,
        beam_size: int = 24,
        parallel_visit_count: int = 4,
        beam_queue_factor: int = 3,
        visit_queue_factor: int = 3,
        exclude_factor: int = 5,
        optimization_loops: int = 3,
        batch_size: int = 10_000,
        recall_search_queue_factor: int = 3,
        speculative_readahead: int = 5,
    ):
        (count, dim) = vectors.size()
        self.configuration = {
            "vector_count": count,
            "vector_dimension": dim,
            "prune": prune,
            "beam_size": beam_size,
            "parallel_visit_count": parallel_visit_count,
            "beam_queue_factor": beam_queue_factor,
            "visit_queue_factor": visit_queue_factor,
            "exclude_factor": exclude_factor,
            "optimization_loops": optimization_loops,
            "batch_size": batch_size,
            "recall_search_queue_factor": recall_search_queue_factor,
            "speculative_readahead": speculative_readahead,
        }
        self.vectors = vectors
        (self.distances, self.beams) = generate_ann(self.vectors, self.configuration)

    def calculate_recall(self, sample_size: int = 1000) -> Tuple[float, float]:
        (count, _) = self.vectors.size()
        sample_size = min(sample_size, count)
        sample = self.vectors[0:sample_size]
        queue = initial_queue(sample, self.configuration)

        closest_vectors(sample, queue, self.vectors, self.beams, self.configuration)
        # print_timestamp("closest vectors calculated")
        expected = torch.arange(sample_size)
        actual = queue.indices.t()[0]
        found = (expected == actual).sum().item()

        return (found, found / sample_size)

    def search():
        pass


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
    parser.add_argument(
        "-l", "--optimization_loops", help="cagra loops", type=int, default=3
    )
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
        "optimization_loops": args.optimization_loops,
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
