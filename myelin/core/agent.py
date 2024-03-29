from .policy import Policy
from ..utils import Experience


class Agent(Policy):
    """A reinforcement learning agent that learns by interacting with an environment."""

    def __init__(self, policy):
        self.policy = policy
        self.action_space = policy.action_space

    def update(self, experience: Experience):
        """An experience is a tuple (state, action, reward, next_state, done)"""
        pass

    ##########
    # Policy #
    ##########

    def get_action(self, state):
        return self.policy.get_action(state)

    def get_action_prob(self, state, action):
        return self.policy.get_action_prob(state, action)
