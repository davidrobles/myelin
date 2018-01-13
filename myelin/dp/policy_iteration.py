import numpy as np


class PolicyIteration():
    """Policy Iteration"""

    def __init__(self, mdp, gamma, policy, vf, theta):
        self.mdp = mdp
        self.gamma = gamma
        self.policy = policy
        self.vf = vf
        self.theta = theta

    def prob_target(self, state, action):
        """Find a better name for this"""
        value = 0
        for next_state, prob in self.mdp.get_transitions(state, action):
            reward = self.mdp.get_reward(state, action, next_state)
            target = reward + (self.gamma * self.vf[next_state])
            value += prob * target
        return value

    def state_value(self, state):
        """Find a better name for this"""
        value = 0
        for action in self.mdp.get_actions(state):
            action_prob = self.policy.get_action_prob(state, action)
            prob_target = self.prob_target(state, action)
            value += action_prob * prob_target
        return value

    def policy_eval(self):
        """Policy evaluation"""
        delta = 0
        for state in self.mdp.get_states():
            old_value = self.vf[state]
            self.vf[state] = self.state_value(state)
            delta = max(delta, abs(old_value - self.vf[state]))
        return delta

    def iter_policy_eval(self):
        """Iterative policy evaluation"""
        delta = np.inf
        while delta >= self.theta:
            delta = self.policy_eval()
            print('Delta: {:.5f}'.format(delta))

    def policy_improvement(self):
        """Policy improvement"""
        print('Policy improvement...')
        policy_stable = True
        self.policy.vf = {}
        for state in self.mdp.get_states():
            old_action = self.policy.get_action(state)
            best_action = None
            best_value = -np.inf
            for action in self.mdp.get_actions(state):
                action_prob = self.policy.get_action_prob(state, action)
                prob_target = self.prob_target(state, action)
                value = action_prob * prob_target
                if value > best_value:
                    best_value = value
                    best_action = action
            self.policy.vf[state] = best_action

# TODO: there can be multiple best actions
