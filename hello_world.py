import torch
import taichi
from taichi import types as ty


@taichi.kernel
def dot_product(
    x: ty.ndarray(dtype=taichi.f32, ndim=1),
    y: ty.ndarray(dtype=taichi.f32, ndim=1),
    scratch: ty.ndarray(dtype=taichi.f32, ndim=1),
) -> taichi.f32:
    for i in x:
        scratch[i] = x[i] * y[i]
    taichi.sync()
    acc = 0.0
    for i in scratch:
        acc += scratch[i]
    return acc


def do_dot_product():
    x = torch.tensor([1.0, 2.0])
    y = torch.tensor([3.0, 1.0])
    scratch = torch.tensor([0.0, 0.0])
    return dot_product(x, y, scratch)


def main():
    taichi.init(arch=taichi.vulkan)
    print(do_dot_product())
