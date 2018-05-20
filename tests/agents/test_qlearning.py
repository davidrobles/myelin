import unittest

from myelin.agents import QLearning
from myelin.policies import RandomPolicy
from myelin.utils import Experience
from myelin.value_functions import TabularQF


class TestQLearning(unittest.TestCase):
    def setUp(self):
        self.qf = TabularQF(init=False)
        self.policy = RandomPolicy(lambda s: [1])
        self.learning_rate = 1.0
        self.discount_factor = 1.0
        self.qlearning = QLearning(policy=self.policy, qfunction=self.qf, learning_rate=self.learning_rate,
                                   discount_factor=self.discount_factor)

    def test_update(self):
        # settings
        self.qlearning.discount_factor = 1.0
        self.qlearning.learning_rate = 1.0
        # default
        state = 1
        action = 'north'
        next_state = 2
        done = False
        # step 1
        reward = 1.0
        experience = Experience(state, action, reward, next_state, done)
        self.qlearning.update(experience)
        self.assertEqual(self.qf[state, action], 1.0)
        # step 2
        reward = 1.0
        experience = Experience(state, action, reward, next_state, done)
        self.qlearning.update(experience)
        self.assertEqual(self.qf[state, action], 1.0)
        # # step 3
        reward = 0.0
        experience = Experience(state, action, reward, next_state, done)
        self.qlearning.update(experience)
        self.assertEqual(self.qf[state, action], 0.0)
