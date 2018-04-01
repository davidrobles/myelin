from collections import defaultdict

import numpy as np

from myelin.core import Agent


class FirstVisitMonteCarlo(Agent):
    """
    First-visit Monte Carlo

    # Arguments
        policy: behavior policy.
        vfunction: a state value function.
    """

    def __init__(self, policy, vfunction):
        super().__init__(policy)
        self.vfunction = vfunction
        self.visited = []
        self.rewards = []
        self.returns = defaultdict(list)

    def reset(self):
        self.visited = []
        self.rewards = []

    #########
    # Agent #
    #########

    def update(self, experience):
        """Called after an action has been taken"""
        state, action, reward, next_state, done = experience
        self.rewards.append(reward)
        self.visited.append(next_state)
        if done:
            for s in set(self.visited):
                idx = self.visited.index(s)
                return_ = np.sum(self.rewards[idx:])
                self.returns[s].append(return_)
                self.vfunction[s] = np.mean(np.array(self.returns[s]))
            self.reset()
