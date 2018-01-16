from myelin.core import Agent


class SARSA(Agent):
    """
    Tabular SARSA.

    # Arguments
        policy: behavior policy.
        qfunction: a state-action value function.
        learning_rate: float >= 0.
        discount_factor: float >= 0.
    """

    def __init__(self, policy, qfunction, learning_rate=0.1, discount_factor=1.0):
        self.policy = policy
        self.qfunction = qfunction
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

    #########
    # Agent #
    #########

    def update(self, experience):
        """Called after an action has been taken"""
        state, action, reward, next_state, done = experience
        if done:
            target = reward
        else:
            next_action = self.policy.action(next_state)
            target = reward + (self.discount_factor * self.qfunction[next_state, next_action])
        td_error = target - self.qfunction[state, action]
        self.qfunction[state, action] += self.learning_rate * td_error

    ##########
    # Policy #
    ##########

    def get_action(self, state):
        return self.policy.get_action(state)
