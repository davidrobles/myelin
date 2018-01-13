from myelin.core import Agent


class QLearning(Agent):
    """
    Tabular Q-learning.

    # Arguments
        env: environment.
        policy: behavior policy.
        qfunction: a state-action value function.
        learning_rate: float >= 0.
        discount_factor: float >= 0.
    """

    def __init__(self, action_space, policy, qfunction, learning_rate=0.1, discount_factor=1.0, max_value=True):
        self.action_space = action_space
        self.policy = policy
        self.qfunction = qfunction
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.max_value = max_value

    #########
    # Agent #
    #########

    def update(self, experience):
        """Called after an action has been taken"""
        state, action, reward, next_state, done = experience
        if done:
            target = reward
        else:
            next_actions = self.action_space(next_state)
            q_values = [self.qfunction[next_state, next_action] for next_action in next_actions]
            best_q_value = max(q_values) if self.max_value else min(q_values)
            target = reward + (self.discount_factor * best_q_value)
        td_error = target - self.qfunction[state, action]
        self.qfunction[state, action] += self.learning_rate * td_error

    ##########
    # Policy #
    ##########

    def get_action(self, state):
        return self.policy.get_action(state)
