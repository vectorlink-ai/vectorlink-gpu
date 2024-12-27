# VectorLink.py

VectorLink.py is a python library that exposes tools for managing vectors at scale. It is part of the VectorLink pipeline.

## Requirements

VectorLink.py *requires* an NVidia GPU of at least Lovelace or better architecture to function. The speed difference with GPU acceleration means that vector processing at scale is simply best left to GPUs for reasons of both cost and time.

You will also need an installation of [pytorch]() which works with your GPU.

## Examples

To make a new Approximate Nearest Neighbor graph (ANN) using VectorLink.py, simply import the library and pass it the vectors you want to index as a 2D tensor, the first dimension is the number of vectors and the second dimension being the vector length. Currently we only process `torch.float32` vectors.

### Indexing

```python
from vectorlink import ANN

ann = ANN(vectors=vectors)
sq = ann.search(a) # Searches for a vector or collection of vectors, and returns a search queue
c = ann.clusters() # Returns all search queues
```
