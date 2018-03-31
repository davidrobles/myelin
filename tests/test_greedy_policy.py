import unittest

from myelin.policies import Greedy
from myelin.value_functions.tabular import TabularVF


class TestGreedyPolicy(unittest.TestCase):
    def setUp(self):
        def action_space(state):
            actions = {
                'north': [1, 4, 5, 8],
                'south': [2, 3, 4]
            }
            return actions[state]

        self.qfunction = TabularVF()
        self.qfunction['north', 1] = 5
        self.qfunction['north', 4] = 23
        self.qfunction['north', 5] = 73
        self.qfunction['north', 8] = 33
        self.qfunction['south', 2] = 2
        self.qfunction['south', 3] = 45
        self.qfunction['south', 4] = 45
        self.policy = Greedy(action_space, self.qfunction)

    def test_get_action_returns_best_action(self):
        self.assertEqual(self.policy.get_action('north'), 5)
        self.assertIn(self.policy.get_action('south'), (3, 4))

    def test_get_action_prob_returns_correct_probabilities(self):
        self.assertEqual(self.policy.get_action_prob('north', 1), 0)
        self.assertEqual(self.policy.get_action_prob('north', 4), 0)
        self.assertEqual(self.policy.get_action_prob('north', 5), 1)
        self.assertEqual(self.policy.get_action_prob('north', 8), 0)
        self.assertEqual(self.policy.get_action_prob('south', 2), 0)
        self.assertEqual(self.policy.get_action_prob('south', 3), 0.5)
        self.assertEqual(self.policy.get_action_prob('south', 4), 0.5)

    def test_get_action_prob_raises_exception_if_invalid_action(self):
        with self.assertRaises(ValueError):
            self.policy.get_action_prob('north', 'invalid_action')
