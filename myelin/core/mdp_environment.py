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
        return self._mdp.get_actions(state)

    def get_state(self):
        return self._cur_state

    def step(self, action):
        prev = self.get_state()
        transitions = self._mdp.get_transitions(self.get_state(), action)
        for next_state, prob in transitions:
            self._cur_state = next_state
        reward = self._mdp.get_reward(prev, action, self.get_state())
        return self.get_state(), reward, self.is_terminal(), {}

    def reset(self):
        self._cur_state = self._mdp.get_start_state()
