from myelin.core import Policy


class Greedy(Policy):
    """Greedy Policy"""

    def __init__(self, action_space, qfunction):
        if qfunction is None:
            raise ValueError('Requires a Q-function')
        self.action_space = action_space
        self.qfunc = qfunction

    ##########
    # Policy #
    ##########

    def get_action(self, state):
        actions = self.action_space(state)
        if not actions:
            raise ValueError('Must have at least one available action.')
        state_actions = [(state, action) for action in actions]
        _, best_action = max(state_actions, key=lambda state_action: self.qfunc[state_action])
        return best_action

    def get_action_prob(self, state, action):
        actions = self.action_space(state)
        if action not in actions:
            raise ValueError('Invalid action')
        state_actions = [(state, action) for action in actions]
        _, first_best_action = max(state_actions, key=lambda state_action: self.qfunc[state_action])
        best_value = self.qfunc[state, first_best_action]
        best_actions = [action for state, action in state_actions if self.qfunc[state, action] == best_value]
        if action not in best_actions:
            return 0
        return 1 / len(best_actions)
