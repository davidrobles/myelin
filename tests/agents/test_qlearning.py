import unittest

from myelin.agents import QLearning
from myelin.policies import RandomPolicy
from myelin.utils import Experience
from myelin.value_functions import TabularQF


class TestQLearning(unittest.TestCase):
    def setUp(self):
        self.qf = TabularQF(init=False)
        self.policy = RandomPolicy(lambda s: ['north'])
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

    def test_update_with_no_learning(self):
        self.qlearning.learning_rate = 0.0
        state = 1
        action = 'north'
        next_state = 2
        done = False
        self.qf[state, action] = 23
        reward = 1.0
        experience = Experience(state, action, reward, next_state, done)
        self.qlearning.update(experience)
        self.assertEqual(self.qf[state, action], 23)

    def test_update_with_discount_factor(self):
        self.qlearning.discount_factor = 0.99
        state = 1
        action = 'north'
        next_state = 2
        done = False
        # step 1
        self.qf[state, action] = 0.0
        self.qf[next_state, action] = 1.0
        reward = 1.0
        experience = Experience(state, action, reward, next_state, done)
        self.qlearning.update(experience)
        self.assertEqual(self.qf[state, action], 1.99)
        # step 2
        reward = 1.0
        experience = Experience(state, action, reward, next_state, done)
        self.qlearning.update(experience)
        self.assertEqual(self.qf[state, action], 1.99)
        # step 3
        reward = 2.0
        experience = Experience(state, action, reward, next_state, done)
        self.qlearning.update(experience)
        self.assertEqual(self.qf[state, action], 2.99)

    def test_update_with_low_learning_rate_and_high_discount_factor(self):
        # settings
        self.qlearning.learning_rate = 0.1
        self.qlearning.discount_factor = 0.99
        # experience
        state = 1
        action = 'north'
        next_state = 2
        done = False
        # step 1
        self.qf[state, action] = 0.0
        self.qf[next_state, action] = 0.0
        reward = 1.0
        experience = Experience(state, action, reward, next_state, done)
        self.qlearning.update(experience)
        self.assertEqual(self.qf[state, action], 0.1)
        # step 2
        reward = 1.0
        experience = Experience(state, action, reward, next_state, done)
        self.qlearning.update(experience)
        self.assertEqual(self.qf[state, action], 0.19)
        # step 3
        reward = 1.0
        experience = Experience(state, action, reward, next_state, done)
        self.qlearning.update(experience)
        self.assertEqual(self.qf[state, action], 0.271)

    def test_update_with_low_learning_rate_and_high_discount_factor_and_non_zero_next_state(self):
        # settings
        self.qlearning.learning_rate = 0.1
        self.qlearning.discount_factor = 0.99
        # experience
        state = 1
        action = 'north'
        next_state = 2
        done = False
        # step 1
        self.qf[state, action] = 0.0
        self.qf[next_state, action] = 1.0
        reward = 1.0
        experience = Experience(state, action, reward, next_state, done)
        self.qlearning.update(experience)
        self.assertEqual(self.qf[state, action], 0.199)
        # step 2
        reward = 1.0
        experience = Experience(state, action, reward, next_state, done)
        self.qlearning.update(experience)
        self.assertEqual(self.qf[state, action], 0.3781)
        # # step 3
        reward = 1.0
        experience = Experience(state, action, reward, next_state, done)
        self.qlearning.update(experience)
        self.assertEqual(self.qf[state, action], 0.53929)
