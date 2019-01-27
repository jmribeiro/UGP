import gym
import time
import Config
import numpy as np
from EnvironmentHandler import EnvironmentHandler
from multiprocessing import Process, Queue, Value

class ActorLearner(Process):

    """
         Actor Learner
             This class asynchronously:
             1) Interacts with its instance of the environment
             2) Updates the global network
     """

    def __init__(self, actor_learner_id, ugp, environment):
        super(ActorLearner, self).__init__()

        self.ugp = ugp
        self.id = actor_learner_id

        render = True if (Config.RENDER_MODE == 1 and self.id == 0) or Config.RENDER_MODE == 2 else False

        self.environment = environment

        env = gym.make(environment)
        self.action_space = env.action_space.n
        env.close()

        self.env = EnvironmentHandler(environment, render, self.ugp.render_lock)
        self.action_space = self.env.action_space
        self.actions = np.arange(self.action_space)

        self.requested_predictions = Queue(maxsize=1)
        self.running = Value('b', True)

    def run(self):

        time.sleep(1.0)
        np.random.seed(np.int32(time.time() % 1 * self.id * 5555))

        try:
            while self.running.value:

                total_score = 0
                total_frames = 0

                for states, actions, rewards, score in self.run_episode():

                    total_score += score
                    total_frames += len(actions) + 1

                    if not Config.PLAY_MODE: self.ugp.training_queue.put((states, actions, rewards, self.environment))

                self.ugp.logger.log(self.id, self.environment, total_score, total_frames)

        except KeyboardInterrupt:

            print(f"Manually Stopping Actor Learner {self.id}")

    def run_episode(self):

        state = self.env.reset()

        terminal = False
        batch = []
        t = 0
        score = 0.0

        while not terminal and self.running.value:

            action, policy, value = self.action(state)
            next_state, reward, terminal = self.env.step(action)

            score += reward

            batch.append(Datapoint(state, action, policy, value, reward, terminal))

            if terminal or t == Config.T_MAX:

                if Config.PLAY_MODE: yield [], [dp.action for dp in batch], [], score
                else:
                    states, actions, rewards = self.prepare_batch(batch)
                    yield states, actions, rewards, score

                batch = [batch[-1]]
                t = 0
                score = 0.0

            t += 1
            state = next_state

    def accumulate_rewards(self, batch):

        last_datapoint = batch[-1]

        accumulator = last_datapoint.value if not last_datapoint.terminal else 0

        for t in reversed(range(0, len(batch) - 1)):
            reward = np.clip(batch[t].reward, Config.REWARD_MIN_CLIP, Config.REWARD_MAX_CLIP)
            accumulator = reward + Config.DISCOUNT_FACTOR * accumulator
            batch[t].reward = accumulator

        return batch[:-1]

    def prepare_batch(self, batch):
        batch = self.accumulate_rewards(batch)
        states = np.array([exp.state for exp in batch])
        actions = np.eye(self.action_space)[np.array([exp.action for exp in batch])].astype(np.float32)
        rewards = np.array([exp.reward for exp in batch])
        return states, actions, rewards

    def action(self, state):
        self.ugp.prediction_queue.put((self.id, self.environment, state))
        policy, value = self.requested_predictions.get()
        action = self.action_from_policy(policy)
        return action, policy, value

    def action_from_policy(self, policy):

        if Config.PLAY_MODE:
            return self.exploit(policy)
        else:
            return np.random.choice(self.actions, p=policy)

    def exploit(self, policy):

        actions = []
        action_max_idx = np.argmax(policy)
        max_prob = policy[action_max_idx]
        actions.append(action_max_idx)

        for idx, prob in enumerate(policy):

            if idx == action_max_idx:
                continue

            if max_prob - Config.EXPLOITATION_DISTANCE < prob:
                actions.append(idx)

        return np.random.choice(actions)

####################
### Data Classes ###
####################

class Datapoint:
    def __init__(self, state, action, policy, value, reward, terminal):
        self.state = state
        self.action = action
        self.policy = policy
        self.value = value
        self.reward = reward
        self.terminal = terminal
