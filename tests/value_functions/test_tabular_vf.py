import unittest

from myelin.value_functions.tabular_vf import TabularVF


class TestTabularVF(unittest.TestCase):
    def setUp(self):
        self.vf = TabularVF(init=True)

    def test_save_and_retrieve_state_values(self):
        state = 1
        self.vf[state] = 5
        self.assertEqual(self.vf[state], 5)

    def test_should_return_zero_for_non_existent_state_values(self):
        vf = TabularVF(init=False)
        state = 0
        self.assertEqual(vf[state], 0)
        state = 1
        self.assertEqual(vf[state], 0)

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
