import cProfile
import pandas as pd
from evalml import AutoMLSearch


def kdd():
    X = pd.read_csv("/Users/parthiv.naresh/Documents/Datasets/Classification/Binary/KDDCup09_churn.csv")
    y = X.pop("CHURN")
    import time
    beginning = time.time()
    aml = AutoMLSearch(X_train=X, y_train=y, problem_type="binary", max_iterations=1)
    beginning_0 = time.time()
    aml.search()
    end = time.time()
    print(f"Total: {beginning_0-beginning}")
    print(f"Total: {end-beginning_0}")
    print("DONEEEEEEEEEEEEEEEEEEE")
    X.ww.init()
    print(X.ww.select(include="Double"))
    print(len(X.ww.select(include="Double")))

if __name__ == '__main__':
    import cProfile, pstats
    profiler = cProfile.Profile()
    profiler.enable()
    kdd()
    profiler.disable()
    stats = pstats.Stats(profiler)
    sorted_stats = stats.sort_stats('cumtime')
    sorted_stats.print_stats(50)
    stats.dump_stats('/Users/parthiv.naresh/0_16_4_stat.txt')
