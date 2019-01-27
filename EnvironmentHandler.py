import gym
import Config
import numpy as np
from queue import Queue
import scipy.misc as misc

class EnvironmentHandler:

    def __init__(self, environment, render, render_lock):
        self.environment = gym.make(environment)
        self.action_space = self.environment.action_space.n
        self.lookback_memory = Queue(maxsize=Config.STACKED_FRAMES)
        self.should_render = render
        self.render_lock = render_lock

    def reset(self):

        self.lookback_memory.queue.clear()

        state = self.state(self.environment.reset())
        while state is None:
            state, _, _ = self.step(0)

        return state

    def step(self, action):

        if self.should_render:
            with self.render_lock:
                self.environment.render()

        observation, reward, done, _ = self.environment.step(action)
        next_state = self.state(observation)

        return next_state, reward, done

    def state(self, observation):

        # Already had full depth, remove the oldest
        if self.lookback_memory.full(): self.lookback_memory.get()

        # Add the new one
        self.lookback_memory.put(self.preprocess(observation))

        # Game hasn't stacked enough frames yet
        if not self.lookback_memory.full():
            return None
        else:
            # Stack state
            state = np.array(self.lookback_memory.queue)
            return np.transpose(state, [1, 2, 0])

    def preprocess(self, observation):
        grayscale_image = np.dot(observation[..., :3], [0.299, 0.587, 0.114])
        resized_image = misc.imresize(grayscale_image, [Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH], 'bilinear')
        processed_image = resized_image.astype(np.float32) / 128.0 - 1.0
        return processed_image

    def close(self):
        self.lookback_memory.queue.clear()
        self.environment.close()
