from collections import defaultdict

import numpy as np

from myelin.core import Agent


class EveryVisitMonteCarlo(Agent):
    """
    Every-visit Monte Carlo

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
        _, _, reward, next_state, done = experience
        self.rewards.append(reward)
        self.visited.append(next_state)
        if done:
            for i, visited_state in enumerate(self.visited):
                return_ = np.sum(self.rewards[i:])
                self.returns[visited_state].append(return_)
                self.vfunction[visited_state] = np.mean(np.array(self.returns[visited_state]))
            self.reset()
