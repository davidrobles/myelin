from myelin.core import Agent


class TabularTD0(Agent):
    """
    Tabular TD0.

    # Arguments
        action_space: action space of the environment.
        policy: behavior policy.
        vfunction: a state value function.
        learning_rate: float >= 0.
        discount_factor: float >= 0.
    """

    def __init__(self, action_space, policy, vfunction, learning_rate=0.1, discount_factor=1.0, max_value=True):
        super().__init__(policy)
        self.action_space = action_space
        self.policy = policy
        self.vfunction = vfunction
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
            target = reward + (self.discount_factor * self.vfunction[next_state])
        td_error = target - self.vfunction[state]
        self.vfunction[state] += self.learning_rate * td_error
