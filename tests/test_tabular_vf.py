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

    def test_retrieve_should_return_zero(self):
        vf = TabularVF(init=False)
        self.assertEqual(vf[0], 0)

