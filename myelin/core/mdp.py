from abc import ABCMeta, abstractmethod


class MDP(metaclass=ABCMeta):
    """Markov Decision Process"""

    @abstractmethod
    def get_actions(self, state):
        """Returns a list of possible actions in the given state."""

    @abstractmethod
    def get_reward(self, state, action, next_state):
        """
        Returns the reward of being in 'state', taking 'action', and ending up
        in 'next_state'.
        Not available in reinforcement learning.
        """

    @abstractmethod
    def get_start_state(self):
        """Returns the initial state."""

    @abstractmethod
    def get_states(self):
        """
        Returns a list of all states.
        Not generally possible for large MDPs.
        """

    @abstractmethod
    def get_transitions(self, state, action):
        """
        Returns a dict of (next_state: probability) key/values, where
        'next_state' is reachable from 'state' by taking 'action'. The sum of
        all probabilities should be 1.0.
        Note that in Q-Learning and reinforcement learning in general, we do
        not know these probabilities nor do we directly model them.
        """

    @abstractmethod
    def is_terminal(self, state):
        """
        Returns true if the given state is terminal. By convention, a terminal
        state has zero future rewards. Sometimes the terminal state(s) may have
        no possible actions. It is also common to think of the terminal state
        as having a self-loop action 'pass' with zero reward; the formulations
        are equivalent.
        """


class MarkovGame(metaclass=ABCMeta):
    """Markov Game"""


class AlternatingMarkovGame(metaclass=ABCMeta):
    """Alternating Markov Game"""


class POMPDP(metaclass=ABCMeta):
    """Partially Observable Markov Decision Process"""
