import time
import math

import numpy as np
from numpy.testing import assert_allclose
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import awkward as ak
import pandas as pd
import tensorflow as tf
import torch


def setup():
    rng = np.random.default_rng(123)
    ragged_data = []
    for i in range(1, 10_001):
        ragged_data.append(rng.random(size=i))
    ragged_data = np.asarray(ragged_data, dtype=object)
    return ragged_data


def check_result(orig_data, result):
    if not isinstance(result, tf.RaggedTensor):
        assert len(result) == len(orig_data)
    else:
        assert result.shape[0] == len(orig_data)
    assert_allclose(result[10][1], math.sqrt(orig_data[10][1]))


def raw_python_bench():
    """
    Raw Python on the ragged NumPy array, with no zero-filling.
    """
    ragged_data = setup()
    start = time.perf_counter()
    for row in range(len(ragged_data)):
        for col in range(len(ragged_data[row])):
            ragged_data[row][col] = math.sqrt(ragged_data[row][col])
    end = time.perf_counter()
    total_sec = end - start
    return [total_sec], ragged_data


def awkward_bench():
    """
    Using Awkward array to handle the sqrt calculation. The
    conversion to ak format is included in the timing.
    """
    ragged_data = setup()
    start = time.perf_counter()
    ragged_data = np.sqrt(ak.Array(ragged_data.tolist()))
    end = time.perf_counter()
    total_sec = end - start
    return [total_sec], ragged_data


def tf_bench(device):
    """
    Using tensorflow Ragged tensors for sqrt. Type/format
    conversions are included in the timing.
    """
    ragged_data = setup()
    start = time.perf_counter()
    with tf.device(device):
        ragged_data = tf.ragged.constant(ragged_data)
        ragged_data = tf.math.sqrt(ragged_data)
    end = time.perf_counter()
    total_sec = end - start
    return [total_sec], ragged_data


def torch_bench(device):
    """
    Using torch nested tensors for sqrt. Type/format
    conversions are included in the timing.
    """
    ragged_data = setup()
    start = time.perf_counter()
    with torch.device(device):
        ragged_data = torch.nested.nested_tensor(ragged_data.tolist())
        ragged_data = torch.sqrt(ragged_data)
    end = time.perf_counter()
    total_sec = end - start
    return [total_sec], ragged_data


def plot_results(bench_results):
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(8, 4)
    df = pd.DataFrame.from_dict(data=bench_results,
                                orient="index",
                                columns=["Time (s)"])
    df.plot.bar(ax=ax, legend=None)
    ax.set_ylabel("Elementwise sqrt time (s)")
    fig.tight_layout()
    fig.savefig("bench_sqrt_ragged.png", dpi=300)


def main_bench():
    orig_data = setup()
    bench_results = {}
    bench_results["raw_python"], result = raw_python_bench()
    check_result(orig_data, result)
    bench_results["awk_array"], result = awkward_bench()
    check_result(orig_data, result)
    bench_results["tf_ragged_gpu"], result = tf_bench(device="/device:GPU:0")
    check_result(orig_data, result)
    bench_results["tf_ragged_cpu"], result = tf_bench(device="/device:CPU:0")
    check_result(orig_data, result)
    # NOTE: torch nested_tensor does not support sqrt op at this time
    #bench_results["torch_nested_cpu"], result = torch_bench(device="cpu")
    #check_result(orig_data, result)
    #bench_results["torch_nested_gpu"], result = torch_bench(device="mps")
    #check_result(orig_data, result)
    plot_results(bench_results)


if __name__ == "__main__":
    main_bench()
