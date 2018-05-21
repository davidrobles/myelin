import numpy as np

from .environment import Environment


class GymEnvironment(Environment):

    def __init__(self, gym_env):
        self.gym_env = gym_env
        self.state = self.gym_env.reset()
        self.done = False

    # @property
    # def action_space(self):
    #     pass

    def get_actions(self, state):
        n_actions = self.gym_env.action_space.n
        return list(np.arange(n_actions))

    def get_state(self):
        return self.state

    def step(self, action):
        next_state, reward, done, info = self.gym_env.step(action)
        self.state = next_state
        self.done = done
        return next_state, reward, done, info

    def is_terminal(self):
        return self.done

    def reset(self):
        self.done = False
        self.state = self.gym_env.reset()
