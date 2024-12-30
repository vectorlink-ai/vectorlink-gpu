import vectorlink
import argparse
import torch
import time

from vectorlink.ann import ANN
from vectorlink.constants import DEVICE
from vectorlink.utils import generate_random_vectors


def main(vectors, configuration):
    start = time.time()
    args = configuration.copy()
    args.pop("vector_count")
    args.pop("vector_dimension")
    ann = ANN(vectors=vectors, **args)
    recall = ann.recall()
    end = time.time()

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
