import math
import numpy as np
import pandas as pd
import sys

def recompute_running_stats(scores):
    avgs = np.zeros_like(scores)
    std_devs = np.zeros_like(scores)

    N = scores.shape[0]

    prev_avg = 0
    prev_var = 0

    for n in range(N):

        x = scores[n]

        if n == 0:
            avg = x

        else:
            avg = prev_avg + ((x - prev_avg) / ((n+1)*1.0) )

        if n > 2:
            variance = ((n - 2) * prev_var + (x - avg) * (x - prev_avg)) / (n - 1)
            std_deviation = math.sqrt(variance)
            prev_var = variance
        else:
            std_deviation = 0

        avgs[n] = avg
        std_devs[n] = std_deviation

        prev_avg = avg

    return avgs, std_devs


def fix_broken_stats_file(filename):
    dataframe = pd.read_csv(filename)
    scores = dataframe["Score"]
    avg_score, std_devs = recompute_running_stats(scores)
    dataframe["AverageScore"] = avg_score
    dataframe["StandardDeviation"] = std_devs
    dataframe.to_csv(filename+".new")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"USAGE: python {sys.argv[0]} stats-file.csv")
        sys.exit(-1)

    fix_broken_stats_file(sys.argv[1])

