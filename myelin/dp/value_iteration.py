
class ValueIterationCallback:
    def on_learning_begin(self, vf):
        """Called at the beginning of learning process."""

    def on_learning_end(self, vf):
        """Called at the end of learning process."""

    def on_iteration_begin(self, vf):
        """Called at the beginning of every iteration (sweep)."""

    def on_iteration_end(self, vf):
        """Called at the end of every iteration (sweep)."""


class ValueIterationCallbackList:
    def __init__(self, callbacks=None):
        self.callbacks = callbacks or []

    def on_learning_begin(self, vf):
        for callback in self.callbacks:
            callback.on_learning_begin(vf)

    def on_learning_end(self, vf):
        for callback in self.callbacks:
            callback.on_learning_end(vf)

    def on_iteration_begin(self, vf):
        for callback in self.callbacks:
            callback.on_iteration_begin(vf)

    def on_iteration_end(self, vf):
        for callback in self.callbacks:
            callback.on_iteration_end(vf)


class ValueIteration:
    """
    Value iteration (Bellman 1957), also called backward induction, is an algorithm that computes the values for all
    the states in the markov decision process. In value iteration no policy is used.
    """

    def __init__(self, mdp, theta, gamma, vfunction, callbacks=None):
        self.mdp = mdp
        self.theta = theta
        self.gamma = gamma
        self.vf = vfunction
        self.callbacks = ValueIterationCallbackList(callbacks)

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
        delta = 1000000
        self.callbacks.on_learning_begin(self.vf)
        while delta >= self.theta:
            self.callbacks.on_iteration_begin(self.vf)
            delta = self.iteration()
            print('Delta: %.4f' % delta)
            self.callbacks.on_iteration_end(self.vf)
        self.callbacks.on_learning_end(self.vf)
