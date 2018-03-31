from myelin.core import Agent


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
            s = 0
            for next_action in next_actions:
                action_prob = self.policy.get_action_prob(next_state, next_action)
                next_value = self.qfunction[next_state, next_action]
                s += action_prob * next_value
            target = reward + (self.discount_factor * s)
        td_error = target - self.qfunction[state, action]
        self.qfunction[state, action] += self.learning_rate * td_error
