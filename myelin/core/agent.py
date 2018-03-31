from abc import abstractmethod

from .policy import Policy


class Agent(Policy):
    """A reinforcement learning agent that learns by interacting with an environment."""

    def __init__(self, policy):
        self.policy = policy

    @abstractmethod
    def update(self, experience):
        """An experience is a tuple (state, action, reward, next_state, done)"""

    def get_action(self, state):
        return self.policy.get_action(state)

    def get_action_prob(self, state, action):
        return self.policy.get_action_prob(state, action)
