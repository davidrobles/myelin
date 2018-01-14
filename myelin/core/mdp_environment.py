from myelin.core.environment import Environment


class MDPEnvironment(Environment):
    """An reinforcement learning environment based on a markov decision process."""

    def __init__(self, mdp):
        self._mdp = mdp
        self._cur_state = self._mdp.get_start_state()

    def is_terminal(self):
        return self._mdp.is_terminal(self.get_state())

    @property
    def action_space(self):
        return self._mdp.get_actions

    ###############
    # Environment #
    ###############

    def get_actions(self, state):
        """Returns the available actions in the given state."""
        return self._mdp.actions(state)

    def get_state(self):
        """Returns the current state."""
        return self._cur_state

    def do_action(self, action):
        """
        Performs the given action in the current state.
        Returns (reward, next_state).
        """
        prev = self.get_state()
        transitions = self._mdp.get_transitions(self.get_state(), action)
        for next_state, prob in transitions:
            self._cur_state = next_state
        reward = self._mdp.get_reward(prev, action, self.get_state())
        return reward, self.get_state()

    def reset(self):
        """Resets the current state to the start state."""
        self._cur_state = self._mdp.get_start_state()
