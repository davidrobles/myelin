import unittest

from myelin.agents import TabularTD0
from myelin.policies import RandomPolicy
from myelin.utils import Experience
from myelin.value_functions.tabular_vf import TabularVF


class TestTD0(unittest.TestCase):
    def setUp(self):
        self.vf = TabularVF(init=False)
        self.policy = RandomPolicy(lambda s: [1])
        self.learning_rate = 1.0
        self.discount_factor = 1.0
        self.td0 = TabularTD0(policy=self.policy, vfunction=self.vf, learning_rate=self.learning_rate,
                              discount_factor=self.discount_factor)

    def test_update(self):
        # settings
        self.td0.discount_factor = 1.0
        self.td0.learning_rate = 1.0
        # default
        state = 1
        action = 'north'
        next_state = 2
        done = False
        # step 1
        reward = 1.0
        experience = Experience(state, action, reward, next_state, done)
        self.td0.update(experience)
        self.assertEqual(self.vf[state], 1.0)
        # step 2
        reward = 1.0
        experience = Experience(state, action, reward, next_state, done)
        self.td0.update(experience)
        self.assertEqual(self.vf[state], 1.0)
        # step 3
        reward = 0.0
        experience = Experience(state, action, reward, next_state, done)
        self.td0.update(experience)
        self.assertEqual(self.vf[state], 0.0)

    def test_update_with_no_learning(self):
        self.td0.learning_rate = 0.0
        state = 1
        action = 'north'
        next_state = 2
        done = False
        self.vf[state] = 23
        reward = 1.0
        experience = Experience(state, action, reward, next_state, done)
        self.td0.update(experience)
        self.assertEqual(self.vf[state], 23)
