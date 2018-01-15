from myelin.core import Policy


class Greedy(Policy):
    """Greedy Policy"""

    def __init__(self, action_space, qfunction):
        if qfunction is None:
            raise ValueError('Requires either a Q-function')
        self.action_space = action_space
        self.qfunction = qfunction

    def get_action(self, state):
        actions = self.action_space(state)
        if not actions:
            raise ValueError('Must have at least one available action.')
        state_actions = [(state, action) for action in actions]
        _, best_action = max(state_actions, key=lambda s_or_sa: self.qfunction[s_or_sa])
        return best_action
