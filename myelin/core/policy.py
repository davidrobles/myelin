from abc import ABCMeta, abstractmethod


class Policy(metaclass=ABCMeta):
    """
    A policy is a mapping from perceived states of the environment
    to actions to be taken when in those states.
    """

    @abstractmethod
    def get_action(self, state):
        """
        Returns one of the available actions from the given state.
        """
