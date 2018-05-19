import unittest

from myelin.value_functions.tabular import TabularVF


class TestTabularVF(unittest.TestCase):
    def setUp(self):
        self.vf = TabularVF(init=True)

    def test_save_and_retrieve_state_values(self):
        state = 1
        self.vf[state] = 5
        self.assertEqual(self.vf[state], 5)

    def test_save_and_retrieve_state_action_values(self):
        state = 1
        action = 1
        self.vf[state, action] = 10
        self.assertEqual(self.vf[state, action], 10)

    def test_should_return_zero_for_non_existent_state_values(self):
        vf = TabularVF(init=False)
        state = 0
        self.assertEqual(vf[state], 0)
        state = 1
        self.assertEqual(vf[state], 0)

    def test_should_return_zero_for_non_existent_action_values(self):
        vf = TabularVF(init=False)
        state, action = 0, 0
        self.assertEqual(vf[state, action], 0)
        state, action = 1, 1
        self.assertEqual(vf[state, action], 0)

    def test_should_return_randomly_initialized_values_for_non_existent_values(self):
        from unittest.mock import MagicMock
        vf = TabularVF(init=True)
        vf.random_initializer = MagicMock(return_value=0.837)
        state, action = 0, 0
        self.assertEqual(vf[state, action], 0.837)
        vf.random_initializer = MagicMock(return_value=0.184)
        state, action = 1, 1
        self.assertEqual(vf[state, action], 0.184)

    def test_discretizer_for_state_values(self):
        import math
        vf = TabularVF(discretizer=math.ceil, init=False)
        state = 0.3
        vf[state] = 5
        discretized_state = 1
        self.assertEqual(vf[discretized_state], 5)
        state = 3.3
        vf[state] = 10
        discretized_state = 4
        self.assertEqual(vf[discretized_state], 10)