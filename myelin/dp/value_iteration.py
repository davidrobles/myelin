class ValueIteration():
    """Value Iteration"""

    def __init__(self, mdp, theta, gamma, vfunction):
        self.mdp = mdp
        self.theta = theta
        self.gamma = gamma
        self.vf = vfunction

    def iteration(self):
        delta = 0.0
        for state in self.mdp.get_states():
            old_val = self.vf[state]
            new_val = -100000.0
            for action in self.mdp.get_actions(state):
                tot = 0.0
                for next_state, prob in self.mdp.get_transitions(state, action):
                    reward = self.mdp.get_reward(state, action, next_state)
                    next_value = self.vf[next_state]
                    if self.mdp.is_terminal(next_state):
                        assert next_value == 0
                    tot += prob * (reward + (self.gamma * next_value))
                if tot > new_val:
                    new_val = tot
            self.vf[state] = new_val
            delta = max(delta, abs(old_val - self.vf[state]))
        return delta

    def learn(self):
        print('Value Iteration started...')
        delta = 1000000
        while delta >= self.theta:
            delta = self.iteration()
            print('Delta: %.4f' % (delta))
        print('DP Value Iteration finished!')
