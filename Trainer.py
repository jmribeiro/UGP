from multiprocessing import Value

import numpy as np
from threading import Thread

import Config


class Trainer(Thread):

    """
        Trainer

            This class updates the global network using large batches to take advantage of GPU
            Unlike the original A3C, where each individual actor learner updated the network, the GA3C
            receives the batches from each actor learner and submits a large bundle through the optimizer.
            However, unlike the original GA3C, the UGP's Trainers trains the network using batches from each environment

    """

    def __init__(self, trainer_id, ugp):
        super(Trainer, self).__init__()

        self.setDaemon(True)

        self.ugp = ugp
        self.id = trainer_id
        self.running = Value("b", True)

    def run(self):

        print(f"Trainer #{self.id} started")

        while self.running.value:

            batch_size = 0
            env_batches = dict()

            while batch_size <= Config.TRAINING_MIN_BATCH:

                s, a, r, target_environment = self.ugp.training_queue.get()

                if target_environment not in env_batches:
                    states = s
                    actions = a
                    rewards = r
                    env_batches[target_environment] = states, actions, rewards
                else:
                    states, actions, rewards = env_batches[target_environment]
                    states = np.concatenate((states, s))
                    actions = np.concatenate((actions, a))
                    rewards = np.concatenate((rewards, r))
                    env_batches[target_environment] = states, actions, rewards

                batch_size += s.shape[0]

            for environment, batch in env_batches.items():
                states, actions, rewards = batch
                self.ugp.model.train(environment, states, actions, rewards)

        print(f"Stopping Trainer #{self.id}")
