from myelin.mixins import EpisodicLearnerMixin
from myelin.utils import max_qvalue, min_qvalue


class QLearning(EpisodicLearnerMixin):
    """
    Tabular Q-learning.

    # Arguments
        env: environment.
        policy: behavior policy.
        qfunction: a state-action value function.
        learning_rate: float >= 0.
        discount_factor: float >= 0.
        selfplay: boolean. Whether to use the same policy
    """

    def __init__(self, env, policy, qfunction, learning_rate=0.1,
                 discount_factor=1.0, selfplay=False):
        self.env = env
        self.policy = policy
        self.qfunction = qfunction
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.selfplay = selfplay

    def best_qvalue(self, state):
        if self.selfplay:
            best_func = max_qvalue if state.cur_player() == 0 else min_qvalue
            return best_func(state, self.env.get_actions(state), self.qfunction)
        return max_qvalue(state, self.env.get_actions(state), self.qfunction)

    ###########
    # Learner #
    ###########

    def episode(self):
        while not self.env.is_terminal():
            state = self.env.get_state()
            action = self.policy.get_action(state)
            reward, next_state = self.env.do_action(action)
            best_qvalue = self.best_qvalue(next_state)
            target = reward + (self.discount_factor * best_qvalue)
            td_error = target - self.qfunction[state, action]
            self.qfunction[state, action] += self.learning_rate * td_error