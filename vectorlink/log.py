import torch
import time


def timed(fn):
    wall_start = time.time()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    torch.cuda.synchronize()
    end.record()
    torch.cuda.synchronize()
    wall_end = time.time()
    return result, start.elapsed_time(end) / 1_000, (wall_end - wall_start)


CLOSEST_VECTORS_BATCH_TIME = 0.0


def log_time(func):
    if DEBUG is False:
        return func

    global CLOSEST_VECTORS_BATCH_TIME

    def wrapper(*args, **kwargs):
        global CLOSEST_VECTORS_BATCH_TIME

        def closure():
            return func(*args, **kwargs)

        (result, cuda_time, wall_time) = timed(closure)
        print(f"[{func.__name__}]\n\tCUDA time: {cuda_time}\n\tWALL time {wall_time}")
        if func.__name__ == "closest_vectors":
            CLOSEST_VECTORS_BATCH_TIME = max(wall_time, CLOSEST_VECTORS_BATCH_TIME)
        return result

    return wrapper
