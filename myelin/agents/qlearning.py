from myelin.core import Agent


class QLearning(Agent):
    """
    Tabular Q-learning.

    # Arguments
        policy: behavior policy.
        qfunction: a state-action value function.
        learning_rate: float >= 0. Determines to what extent newly acquired information overrides old information. A
            factor of 0 makes the agent learn nothing (exclusively exploiting prior knowledge), while a factor of 1
            makes the agent consider only the most recent information (ignoring prior knowledge to explore
            possibilities).
        discount_factor: float >= 0. Determines the importance of future rewards. A factor of 0 will make the agent
            "myopic" (or short-sighted) by only considering current rewards, while a factor approaching 1 will make it
            strive for a long-term high reward.
    """

    def __init__(self, policy, qfunction, learning_rate=0.1, discount_factor=1.0, max_value=True):
        super().__init__(policy)
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
