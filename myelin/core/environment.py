import abc


class Environment:
    """An environment for reinforcement learning interactions."""

    @property
    def action_space(self):
        pass

    @abc.abstractmethod
    def get_actions(self, state):
        """Returns the available actions in the given state."""

    @abc.abstractmethod
    def get_state(self):
        """Returns the current state."""

    @abc.abstractmethod
    def do_action(self, action):
        """
        Performs the given action in the current state.
        Returns (reward, next_state).
        """

    def is_terminal(self):
        return self._mdp.is_terminal(self.get_state())

    @abc.abstractmethod
    def reset(self):
        """Resets the current state to the start state."""
