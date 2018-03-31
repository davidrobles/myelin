import unittest

from myelin.policies import Greedy
from myelin.value_functions.tabular import TabularVF


class TestGreedy(unittest.TestCase):
    def setUp(self):
        def action_space(state):
            actions = {
                'north': [1, 4, 5, 8],
                'south': [2, 3]
            }
            return actions[state]

        self.qfunction = TabularVF()
        self.qfunction['north', 1] = 5
        self.qfunction['north', 4] = 23
        self.qfunction['north', 5] = 73
        self.qfunction['north', 8] = 33
        self.qfunction['south', 2] = 2
        self.qfunction['south', 3] = 45
        self.policy = Greedy(action_space, self.qfunction)

    def test_max_action(self):
        action = self.policy.get_action('north')
        self.assertEqual(action, 5)
        action = self.policy.get_action('south')
        self.assertEqual(action, 3)

    def test_get_action_prob(self):
        self.assertEqual(self.policy.get_action_prob('north', 1), 0)
        self.assertEqual(self.policy.get_action_prob('north', 4), 0)
        self.assertEqual(self.policy.get_action_prob('north', 5), 1)
        self.assertEqual(self.policy.get_action_prob('north', 8), 0)

    # def test_raises_value_error_if_no_actions_available(self):
    #     state = 1
    #     actions = []
    #     with self.assertRaises(ValueError):
    #         self.policy.get_action(state, actions)
