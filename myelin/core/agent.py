from abc import abstractmethod

from .policy import Policy


class Agent(Policy):
    """A reinforcement learning agent that learns by interacting with an environment."""

    @abstractmethod
    def update(self, experience):
        """An experience is a tuple (state, action, reward, next_state)"""
