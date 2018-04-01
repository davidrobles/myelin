import unittest

from myelin.policies import RandomPolicy


class TestRandomPolicy(unittest.TestCase):
    def setUp(self):
        def action_space(state):
            actions = {
                'north': [1, 4, 5, 8],
                'south': [2, 3, 4]
            }
            return actions[state]
        self.policy = RandomPolicy(action_space)

    def test_get_action_returns_best_action(self):
        self.assertIn(self.policy.get_action('north'), [1, 4, 5, 8])
        self.assertIn(self.policy.get_action('south'), [2, 3, 4])

    def test_get_action_prob_returns_correct_probabilities(self):
        self.assertEqual(self.policy.get_action_prob('north', 1), 0.25)
        self.assertEqual(self.policy.get_action_prob('north', 4), 0.25)
        self.assertEqual(self.policy.get_action_prob('north', 5), 0.25)
        self.assertEqual(self.policy.get_action_prob('north', 8), 0.25)
        self.assertAlmostEqual(self.policy.get_action_prob('south', 2), 0.333, 3)
        self.assertAlmostEqual(self.policy.get_action_prob('south', 3), 0.333, 3)
        self.assertAlmostEqual(self.policy.get_action_prob('south', 4), 0.333, 3)
