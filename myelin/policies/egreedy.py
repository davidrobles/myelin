from myelin.core import Policy
from myelin.policies.greedy import Greedy
from myelin.policies.random_policy import RandomPolicy
from myelin.utils import check_random_state


class EGreedy(Policy):

    def __init__(self, action_space, qfunction, epsilon=0.1, random_state=None):
        if qfunction is None:
            raise ValueError('Requires a Q-function')
        self.action_space = action_space
        self.qfunction = qfunction
        self.epsilon = epsilon
        self.random_state = check_random_state(random_state)
        self.rand_policy = RandomPolicy(action_space)
        self.greedy_policy = Greedy(action_space, self.qfunction)

    ##########
    # Policy #
    ##########

    def get_action(self, state):
        policy = self.rand_policy if self.random_state.rand() < self.epsilon else self.greedy_policy
        return policy.get_action(state)

    def get_action_prob(self, state, action):
        pass
