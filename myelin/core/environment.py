from abc import ABC, abstractmethod


class Environment(ABC):
    """An environment for reinforcement learning interactions."""

    @property
    def action_space(self):
        pass

    @abstractmethod
    def get_actions(self, state):
        """Returns the available actions in the given state."""

    @abstractmethod
    def get_state(self):
        """Returns the current state."""

    @abstractmethod
    def do_action(self, action):
        """
        Performs the given action in the current state.
        Returns (reward, next_state).
        """

    @abstractmethod
    def is_terminal(self):
        """Returns True if the current state is terminal"""

    @abstractmethod
    def reset(self):
        """Resets the current state to the start state."""
