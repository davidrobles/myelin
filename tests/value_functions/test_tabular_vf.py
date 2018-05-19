import unittest

from myelin.value_functions.tabular_vf import TabularVF


class TestTabularVF(unittest.TestCase):
    def setUp(self):
        self.vf = TabularVF(init=True)

    def test_save_and_retrieve_state_values(self):
        state = 1
        self.vf[state] = 5
        self.assertEqual(self.vf[state], 5)

    def test_zero_initialized_state_values(self):
        vf = TabularVF(init=False)
        state = 0
        self.assertEqual(vf[state], 0)
        state = 1
        self.assertEqual(vf[state], 0)

    def test_randomly_initialized_state_values(self):
        from unittest.mock import MagicMock
        vf = TabularVF(init=True)
        vf.random_initializer = MagicMock(return_value=0.837)
        state = 0
        self.assertEqual(vf[state], 0.837)
        vf.random_initializer = MagicMock(return_value=0.184)
        state = 1
        self.assertEqual(vf[state], 0.184)

    def test_discretizer_for_state_values(self):
        import math
        vf = TabularVF(discretizer=math.ceil, init=False)
        state = 0.3
        vf[state] = 5
        discretized_state = 1
        self.assertEqual(vf[state], 5)
        self.assertEqual(vf[discretized_state], 5)
        state = 3.3
        vf[state] = 10
        discretized_state = 4
        self.assertEqual(vf[state], 10)
        self.assertEqual(vf[discretized_state], 10)
