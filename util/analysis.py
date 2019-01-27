# %load analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from IPython.display import display, HTML

#% matplotlib inline


def display_table(title, data):
    print(title)
    display(pd.DataFrame(data=data[1:, 1:], index=data[1:, 0], columns=data[0, 1:]))


def display_play_stats(brains, games, training_episodes=150000, play_episodes=100):
    avg_scores_table_matrix = np.empty((len(brains) + 1, len(games) + 1), dtype=object)
    std_devs_table_matrix = np.empty((len(brains) + 1, len(games) + 1), dtype=object)

    avg_scores_table_matrix[0, 0] = ""
    std_devs_table_matrix[0, 0] = ""

    for idx, game in enumerate(games):
        column = idx + 1

        avg_scores_table_matrix[0][column] = game
        std_devs_table_matrix[0][column] = game

    for idx, brain in enumerate(brains):

        row = idx + 1

        avg_scores_table_matrix[row, 0] = brain + "net"
        std_devs_table_matrix[row, 0] = brain + "net"

        for idx, game in enumerate(games):

            column = idx + 1

            filename = f"Playing After {training_episodes}/{brain}net-playing-{game}-after-{training_episodes}.csv"

            try:
                dataframe = pd.read_csv(filename)

                avg_score = dataframe["AverageScore"][play_episodes - 1]
                std_dev = dataframe["StandardDeviation"][play_episodes - 1]
            except:
                avg_score = -1
                std_dev = -1
                
            avg_scores_table_matrix[row, column] = avg_score
            std_devs_table_matrix[row, column] = std_dev

    #display_table(f"Average Scores After {training_episodes} Training Episodes", avg_scores_table_matrix)
    #display_table(f"Standard Deviation After {training_episodes} Training Episodes", std_devs_table_matrix)


def display_learning_curve(brain, game, training_episodes=150000, color="g", minimum=None, maximum=None):
    filename = f"Learning Until {training_episodes}/{brain}net-learning-{game}-until-{training_episodes}.csv"
    dataframe = pd.read_csv(filename)
    avg_scores = dataframe["AverageScore"]
    
    if minimum == None or maximum == None:
        minimum = min(avg_scores)
        maximum = max(avg_scores)
        
    normalized_avg_scores = (avg_scores - minimum) / (maximum - minimum)
    
    plt.plot(normalized_avg_scores, color)
    plt.title(f"{brain}net learning {game} during {training_episodes} total episodes")
    plt.show()

    return minimum, maximum

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
            avg = prev_avg + ((x - prev_avg) / (n+1))

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