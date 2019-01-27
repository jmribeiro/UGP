from multiprocessing import Value

import numpy as np
from threading import Thread

import Config


class Predictor(Thread):

    """
    Predictor

        This class receives prediction requests from the actor learners,
        submits them through the model and returns the policy/value prediction to the requesting actor learner

    """

    def __init__(self, predictor_id, ugp):

        super(Predictor, self).__init__()

        self.setDaemon(True)

        self.ugp = ugp

        self.id = predictor_id
        self.running = Value("b", True)

    def run(self):

        print(f"Predictor #{self.id} started")

        while self.running.value:

            env_batches = dict()

            for i in range(Config.PREDICTION_MIN_BATCH):

                actor_learner_id, target_environment, state_request = self.ugp.prediction_queue.get()

                if target_environment not in env_batches:
                    actor_learner_ids = [actor_learner_id]
                    state_requests = np.zeros(
                        (Config.PREDICTION_MIN_BATCH,
                         Config.IMAGE_HEIGHT,
                         Config.IMAGE_WIDTH,
                         Config.STACKED_FRAMES), dtype=np.float32)
                    state_requests[0] = state_request
                    env_batches[target_environment] = actor_learner_ids, state_requests
                else:
                    actor_learner_ids, state_requests = env_batches[target_environment]
                    actor_learner_ids.append(actor_learner_id)
                    state_requests[len(actor_learner_ids)-1] = state_request
                    env_batches[target_environment] = actor_learner_ids, state_requests

                if self.ugp.prediction_queue.empty():
                    break

            for environment, batch in env_batches.items():

                actor_learner_ids, state_requests = batch
                policies, values = self.ugp.model.predict(environment, state_requests[:len(actor_learner_ids)])

                for prediction_idx in range(len(actor_learner_ids)):
                    actor_learner_id = actor_learner_ids[prediction_idx]
                    policy, value = policies[prediction_idx], values[prediction_idx]
                    self.ugp.actor_learners[actor_learner_id].requested_predictions.put((policy, value))

        print(f"Stopping Predictor #{self.id}")
