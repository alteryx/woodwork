import cProfile
import pstats
import time

import numpy as np
import pandas as pd

from woodwork import init_series


def kdd():
    # col = np.array([1] * 2 + [2] * 26 + [3] * 9 + [4] * 2 + [5] * 2 + [6] * 2 + [7] * 3 + [8, 8, 9, 9, 10, 11, 13, 16, 21, 28])
    # col = np.repeat(col, 8000)
    # col = pd.Series(col)
    np.random.seed(42)
    df = pd.read_csv("/Users/parthiv.naresh/Documents/Datasets/stones_encoded.csv")
    col = df["meas_depth"]
    sampling = col.sample(20000)
    sampling = init_series(sampling)
    mc_results = sampling.ww.medcouple_dict()
    print(mc_results)


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    kdd()
    profiler.disable()
    stats = pstats.Stats(profiler)
    sorted_stats = stats.sort_stats("cumtime")
    sorted_stats.print_stats(50)
    stats.dump_stats("/Users/parthiv.naresh/0_16_4_stat.txt")
