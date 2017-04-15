import random
from abc import ABCMeta, abstractmethod
from myelin.core import Policy
from myelin.utils import check_random_state


class RandomPolicy(Policy):
    '''
    A uniformly random policy.
    # Arguments
        action_space: a callable that returns a list of available
            actions from a given state.
    '''

    def __init__(self, action_space):
        self.action_space = action_space

    def get_action(self, state):
        actions = self.action_space(state)
        if not actions:
            raise ValueError('Must have at least one available action.')
        return random.choice(actions)

    def get_action_prob(self, state, action):
        return 1.0 / len(self.action_space(state))

    def get_action_probs(self, state):
        return 1.0 / len(self.action_space(state))


class FixedPolicy(Policy):

    def __init__(self, action_space):
        self.action_space = action_space
        self.vf = {}

    def get_action(self, state):
        actions = self.action_space(state)
        if not actions:
            raise ValueError('Must have at least one available action.')
        if state in self.vf:
            return self.vf[state]
        return random.choice(actions)

    def get_action_prob(self, state, action):
        return 1.0 / len(self.action_space(state))
