import os
import sys
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import Config
from Logger import Logger
from Trainer import Trainer
from threading import RLock
from Predictor import Predictor
from multiprocessing import Queue
from ActorLearner import ActorLearner
from ActorCriticNetwork import ActorCriticNetwork


class UniversalGamePlayer:

    def __init__(self):

        self.render_lock = RLock()

        self.model = ActorCriticNetwork(Config.ENVIRONMENTS)

        self.display_config()

        self.prediction_queue = Queue(maxsize=Config.PREDICTION_QUEUE_SIZE)
        self.training_queue = Queue(maxsize=Config.TRAINING_QUEUE_SIZE)

        self.actor_learners = [
            ActorLearner(i, self, Config.ENVIRONMENTS[i])
            for i in range(len(Config.ENVIRONMENTS))
        ]

        self.predictors = [
            Predictor(i, self)
            for i in range(Config.PREDICTORS)
        ]

        self.trainers = [
            Trainer(i, self)
            for i in range(Config.TRAINERS)
        ]

        self.logger = Logger()

        if Config.LOAD:
            try:
                self.logger.total_episodes.value = self.model.load()
                print(f"> Successfully loaded {Config.BRAIN} ({self.logger.total_episodes.value} total episodes)")
                print(f"> Network learned:")
                for environment in self.model.check_known_environments(self.logger.total_episodes.value):
                    print(f"\t- {environment}")
                print()
            except:
                print("No previous network found, starting from scratch")
                self.logger.total_episodes.value = 0

    def display_config(self):

        unique_environments = set(Config.ENVIRONMENTS)

        print(f"\n### Starting '{Config.BRAIN}' ###\n")

        if not Config.PLAY_MODE:
            print("> Mode: [Training]/Play")
        else:
            print("> Mode: Training/[Play]")

        print(f"> Device used: {Config.DEVICE}")
        print(f"> Episodes: {Config.EPISODES}")

        if not Config.PLAY_MODE:
            print(f"> Actor Learning: {len(Config.ENVIRONMENTS)}")

        print(f"> Environments ({int(len(Config.ENVIRONMENTS)/len(unique_environments))} Actor-Learners on each):")
        [print(f"\t - {environment}") for environment in unique_environments]

        if not Config.PLAY_MODE:
            print(f"> Saving every {Config.SAVE_EPISODE_INTERVAL} episodes")

        print()

    def run(self):

        self.start()

        try:

            if Config.PLAY_MODE:

                while self.logger.play_episodes.value < Config.EPISODES:
                    continue

            else:

                last_save = self.logger.total_episodes.value
                ended = False
                current_episode = self.logger.total_episodes.value

                while not ended:
                    last_save = self.checkpoint(last_save, current_episode)
                    if Config.PREPARE_EWC: self.check_sampling(current_episode)
                    current_episode = self.logger.total_episodes.value
                    ended = current_episode > Config.EPISODES

            print(f"Reached {Config.EPISODES}/{Config.EPISODES}, stopping!")

        except KeyboardInterrupt:

            print("Manually stopping Universal Game Player")

        self.stop()

    def start(self):
        for trainer in self.trainers: trainer.start()
        for predictor in self.predictors: predictor.start()
        for actor_learner in self.actor_learners: actor_learner.start()
        self.logger.start()

    def stop(self):

        stop_episode = self.logger.total_episodes.value

        if Config.PREPARE_EWC and self.model.sampling.value:
            self.model.sampling.value = False
            self.model.sampling_finished.value = True

        if not Config.PLAY_MODE:
            self.model.save(stop_episode)

        self.kill_actor_learners()
        self.kill_predictors()
        self.kill_trainers()

        self.logger.running.value = False
        time.sleep(2)

        print("Finished execution")

    def kill_actor_learners(self):
        for actor_learner in self.actor_learners:
            actor_learner.running.value = False
            actor_learner.join()
        print("Stopped Actor Learners")

    def kill_predictors(self):
        for predictor in self.predictors:
            predictor.running.value = False
        print("Stopped Predictors")

    def kill_trainers(self):
        for trainer in self.trainers:
            trainer.running.value = False
        print("Stopped Trainers")

    def checkpoint(self, last_save, this_save):

        should_save = this_save % Config.SAVE_EPISODE_INTERVAL == 0

        if last_save != this_save and should_save:
            self.model.save(self.logger.total_episodes.value)
            last_save = this_save
            print("Checkpoint: Model saved")

        return last_save

    def check_sampling(self, current_episode):
        if not self.model.sampling.value and current_episode > (Config.EPISODES - Config.SAMPLING_EPISODES):
            print("EWC: Started Sampling for Fisher and Star Values")
            self.model.sampling.value = True


if __name__ == '__main__':

    if Config.PLAY_MODE:

        Config.ENVIRONMENTS = [Config.ENVIRONMENTS[0]] * Config.PLAY_ACCELERATOR
        Config.TRAINERS = 0
        Config.LOAD = True

    agent = UniversalGamePlayer()
    agent.run()
