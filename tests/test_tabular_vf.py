import unittest

from myelin.value_functions.tabular import TabularVF


class TestTabularVF(unittest.TestCase):
    def setUp(self):
        self.vf = TabularVF(init=False)

    def test_save_and_retrieve(self):
        state = 1
        action = 1
        self.vf[state] = action
        self.assertEqual(self.vf[state], action)

    def test_should_return_zero_for_non_existent_values(self):
        vf = TabularVF(init=False)
        self.assertEqual(vf[0], 0)

    def test_should_return_randomly_initialized_values_for_non_existent_values(self):
        from unittest.mock import MagicMock
        vf = TabularVF(init=True)
        vf.random_initializer = MagicMock(return_value=0.837)
        self.assertEqual(vf[0], 0.837)
        vf.random_initializer = MagicMock(return_value=0.184)
        self.assertEqual(vf[1], 0.184)
