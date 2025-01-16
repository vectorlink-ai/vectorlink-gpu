from torch import Tensor
from numba import cuda
from datafusion import SessionContext, DataFrame
from datafusion.object_store import AmazonS3, LocalFileSystem, GoogleCloud
import numpy as np
import ctypes


def build_session_context(bucket_name: str) -> SessionContext:
    context = SessionContext()
    context.register_object_store(
        "obj://", GoogleCloud(bucket_name=bucket_name), "bucket"
    )

    return context


def build_local_session_context(directory: str) -> SessionContext:
    context = SessionContext()
    context.register_object_store("obj://", LocalFileSystem(directory), "bucket")

    return context


def dataframe_to_tensor(df: DataFrame, tensor: Tensor):
    # assuming 1536 length floats. this needs to be more properly done later
    # also assuming that the incoming dataframe is sorted!
    cuda_array = cuda.as_cuda_array(tensor)
    stream = df.execute_stream()
    offset = 0
    for batch in stream:
        embeddings = batch.to_pyarrow().column(0)
        embeddings_ptr = ctypes.cast(
            embeddings.buffers()[3].address, ctypes.POINTER(ctypes.c_float)
        )
        embeddings_numpy = np.ctypeslib.as_array(
            embeddings_ptr, (len(embeddings), 1536)
        )
        cuda_slice = cuda_array[offset : offset + len(embeddings)]
        cuda_slice.copy_to_device(embeddings_numpy)
        offset += len(embeddings)
