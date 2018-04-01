from myelin.core import Agent


def action_expected_value(state, action, policy, qf):
    return policy.get_action_prob(state, action) * qf[state, action]


def state_expected_value(state, actions, policy, qf):
    return sum([action_expected_value(state, action, policy, qf) for action in actions])


class ExpectedSARSA(Agent):
    """
    Expected SARSA.

    # Arguments
        policy: behavior policy.
        qfunction: a state-action value function.
        learning_rate: float >= 0.
        discount_factor: float >= 0.
    """

    def __init__(self, action_space, policy, qfunction, learning_rate=0.1, discount_factor=1.0):
        super().__init__(policy)
        self.action_space = action_space
        self.qfunction = qfunction
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

    #########
    # Agent #
    #########

    def update(self, experience):
        state, action, reward, next_state, done = experience
        if done:
            target = reward
        else:
            next_actions = self.action_space(next_state)
            expected_value = state_expected_value(next_state, next_actions, self.policy, self.qfunction)
            target = reward + (self.discount_factor * expected_value)
        td_error = target - self.qfunction[state, action]
        self.qfunction[state, action] += self.learning_rate * td_error
