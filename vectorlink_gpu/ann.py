from __future__ import annotations
from typing import List, Optional, Tuple, Type
from torch import Tensor
import torch
from torch import profiler
import sys
from datetime import datetime
import sys
import os
import subprocess
from typing import Dict
import json
import time
import datafusion as df

from .constants import MAXINT, MAXFLOAT, DEVICE
from .queue import Queue
from .kernels import (
    search_from_seeds,
    dedup_tensor_,
    calculate_distances,
    prune_,
    index_by_tensor,
)
from .utils import (
    index_sort,
    generate_circulant_beams,
    generate_random_vectors,
    primes,
    add_new_to_seen_,
)
from .log import log_time, CLOSEST_VECTORS_BATCH_TIME
from .datafusion import dataframe_to_tensor


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
        beams: Optional[Tensor] = None,
        distances: Optional[Tensor] = None,
    ):
        (count, dim) = vectors.size()
        torch.set_default_device(DEVICE)
        torch.set_float32_matmul_precision("high")

        self.configuration: Dict = {
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
        wall_start = time.time()
        if beams is not None:
            self.beams = beams
            if distances is None:
                self.distances = calculate_distances(self.vectors, self.beams)
            else:
                self.distances = distances
        else:
            (self.beams, self.distances) = generate_ann(
                self.vectors, self.configuration
            )
        wall_end = time.time()
        total_time: float = wall_end - wall_start
        self.log = self.configuration.copy()
        self.log["total_time"] = total_time
        self.log["vector_count"] = count
        self.log["vector_dimension"] = dim

    def load_from_dataframe(dataframe: df.DataFrame, **kwargs) -> ANN:
        "dataframe should have a vector_id, embedding and beams column, and an optional distances column"
        count = dataframe.count()
        assert "embedding" in dataframe.schema().names
        embedding_size = len(
            dataframe.select(df.col("embedding")).head(1).collect()[0]["embedding"][0]
        )
        assert "beams" in dataframe.schema().names
        beam_size = len(
            dataframe.select(df.col("beams")).head(1).collect()[0]["beams"][0]
        )

        dataframe = dataframe.sort(df.col("vector_id"))

        vectors_tensor = torch.empty(
            (count, embedding_size), dtype=torch.float32, device="cuda"
        )
        dataframe_to_tensor(dataframe.select(df.col("embedding")), vectors_tensor)

        beams_tensor = torch.empty((count, beam_size), dtype=torch.int32, device="cuda")
        dataframe_to_tensor(dataframe.select(df.col("beams")), beams_tensor)

        distances_tensor = None
        if "distances" in dataframe.schema().names:
            distances_tensor = torch.empty(
                (count, beam_size), dtype=torch.float32, device="cuda"
            )
            dataframe_to_tensor(dataframe.select(df.col("distances")), distances_tensor)

        return ANN(
            vectors_tensor,
            beam_size=beam_size,
            beams=beams_tensor,
            distances=distances_tensor,
            **kwargs,
        )

    def recall(self, sample_size: int = 1000) -> Tuple[float, float]:
        (count, _) = self.vectors.size()
        sample_size = min(sample_size, count)
        # This should be much more clever about selection
        sample = self.vectors[0:sample_size]

        return ann_calculate_recall(
            self.vectors, self.beams, self.configuration, sample
        )

    def clusters(self):
        pass

    def search(self, query: Tensor) -> Queue:
        """
        Search takes a query tensor, structured as a batch of vectors to search - it returns a Queue object.
        """
        search_queue = initial_queue(self.vectors, self.configuration)
        closest_vectors(
            query, search_queue, self.vectors, self.beams, self.configuration, None
        )
        return search_queue

    def dump_logs(self) -> Dict:
        global CLOSEST_VECTORS_BATCH_TIME
        log = self.log.copy()
        log["closest_vectors_batch_time"] = CLOSEST_VECTORS_BATCH_TIME
        gpu_arch = subprocess.check_output(["nvidia-smi", "-L"])
        log["gpu_arch"] = gpu_arch.decode("utf-8").strip()
        commit = subprocess.check_output(["git", "rev-parse", "--verify", "HEAD"])
        log["commit"] = commit.decode("utf-8").strip()
        (_, recall) = self.recall()
        log["recall"] = recall
        return log
