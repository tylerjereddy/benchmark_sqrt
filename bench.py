import time
import math

import numpy as np
from numpy.testing import assert_allclose
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import awkward as ak


def setup():
    rng = np.random.default_rng(123)
    ragged_data = []
    for i in range(1, 10_001):
        ragged_data.append(rng.random(size=i))
    ragged_data = np.asarray(ragged_data, dtype=object)
    return ragged_data


def check_result(orig_data, result):
    assert len(orig_data) == len(result)
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
    return total_sec, ragged_data


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
    return total_sec, ragged_data


def main_bench():
    orig_data = setup()
    bench_results = {}
    bench_results["raw_python"], result = raw_python_bench()
    check_result(orig_data, result)
    bench_results["awk_array"], result = awkward_bench()
    check_result(orig_data, result)
    print(bench_results)


if __name__ == "__main__":
    main_bench()