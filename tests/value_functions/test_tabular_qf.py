import unittest

from myelin.value_functions import TabularQF


class TestTabularQF(unittest.TestCase):
    def setUp(self):
        self.qf = TabularQF()

    def test_save_and_retrieve_state_action_values(self):
        state = 1
        action = 1
        self.qf[state, action] = 10
        self.assertEqual(self.qf[state, action], 10)

    def test_should_return_zero_for_non_existent_action_values(self):
        qf = TabularQF()
        state, action = 0, 0
        self.assertEqual(qf[state, action], 0)
        state, action = 1, 1
        self.assertEqual(qf[state, action], 0)

    def test_should_return_randomly_initialized_values_for_non_existent_values(self):
        from unittest.mock import MagicMock
        qf = TabularQF()
        qf.random_initializer = MagicMock(return_value=0.837)
        state, action = 0, 0
        self.assertEqual(qf[state, action], 0.837)
        qf.random_initializer = MagicMock(return_value=0.184)
        state, action = 1, 1
        self.assertEqual(qf[state, action], 0.184)

    def test_discretizer_for_state_action_values(self):
        import math
        qf = TabularQF(discretizer=math.ceil)
        state = 0.3
        action = 1
        qf[state, action] = 5
        discretized_state = 1.0
        self.assertEqual(qf[state, action], 5)
        self.assertEqual(qf[discretized_state, action], 5)
        state = 3.3
        action = 2
        qf[state, action] = 10
        discretized_state = 4.0
        self.assertEqual(qf[discretized_state, action], 10)
