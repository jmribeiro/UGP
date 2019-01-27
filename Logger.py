import sys
import math
from queue import Empty

import Config
import pandas as pds
from multiprocessing import Process, Queue, Value

class Logger(Process):

    def __init__(self):

        super(Logger, self).__init__()

        self.logger = Queue(maxsize=100)

        self.environment_loggers = dict()
        for unique_environment in set(Config.ENVIRONMENTS):
            self.environment_loggers[unique_environment] = EnvironmentLogger()

        if Config.LOAD:

            try:

                self.stats = pds.read_csv(f"{Config.STATS_FILE}", index_col=0)
                self.total_episodes = Value('i', self.stats["Episode"].iloc[-1])

                for idx, row in self.stats.iterrows():
                    environment_stats = self.environment_loggers[row['Environment']]
                    environment_stats.add(row['Score'])

            except:
                self.stats = pds.DataFrame(columns=['Episode', 'Environment', 'Score', 'AverageScore', 'StandardDeviation'])
                self.total_episodes = Value('i', 0)
        else:

            self.stats = pds.DataFrame(columns=['Episode', 'Environment', 'Score', 'AverageScore', 'StandardDeviation'])
            self.total_episodes = Value('i', 0)
            
        if Config.PLAY_MODE: self.play_episodes = Value('i', 0)

        self.running = Value("b", True)

    def run(self):

        try:

            while self.running.value:

                try:
                    stat = self.logger.get(block=False)
                except Empty:
                    continue

                self.total_episodes.value += 1
                if Config.PLAY_MODE: self.play_episodes.value += 1

                actor_learner_id, environment, score, frames = stat
                environment_stats = self.environment_loggers[environment]

                environment_stats.add(score)

                episode = self.play_episodes.value if Config.PLAY_MODE else self.total_episodes.value

                print(f'ActorLearner #{actor_learner_id} '
                f'Episode #{episode}, '
                f'Score: {score} '
                f'(Avg. Score {environment_stats.average}) '
                f'(Std. Deviation: {environment_stats.std_deviation})'
                f'({environment})')

                self.stats = self.stats.append(
                    {'Episode': episode,
                     'Environment': environment,
                     'Score': score,
                     'AverageScore': environment_stats.average,
                     'StandardDeviation': environment_stats.std_deviation}, ignore_index=True)

                sys.stdout.flush()

        except KeyboardInterrupt:
            print("Killing Logger")
            self.stats.to_csv(Config.STATS_FILE)

    def log(self, actor_learner_id, environment, score, frames):
        self.logger.put((actor_learner_id, environment, score, frames))

####################
### Data Classes ###
####################

class EnvironmentLogger:

    def __init__(self):
        self.average = 0
        self.variance = 0
        self.std_deviation = 0
        self.N = 0

    def add(self, x):

        self.N += 1

        previous_average = self.average
        previous_variance = self.variance

        self.average = previous_average + ((x - previous_average) / self.N)

        if self.N > 2:

            self.variance = ((self.N - 2) * previous_variance + (x - self.average) * (x - previous_average)) / (self.N - 1)
            self.std_deviation = math.sqrt(self.variance)
