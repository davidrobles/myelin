from myelin.core import ValueFunction
from myelin.utils import check_random_state


class TabularVF(ValueFunction):
    """
    Tabular Value Function for v(s) values.
    # Arguments
        init: boolean. Whether to return a randomly initialized value
            when accessed and does not exist in the table.
    """

    def __init__(self, discretizer=None, init=True, mean=0.0, std=0.3, random_state=None):
        self.discretizer = discretizer or (lambda state: state)
        self.init = init
        self.mean = mean
        self.std = std
        self.random_state = check_random_state(random_state)
        self._table = {}

    def random_initializer(self):
        return self.random_state.normal(self.mean, self.std)

    def __setitem__(self, state, value):
        """
        Sets the state value.
        # Arguments
            key: `state`.
            value: a scalar.
        """
        discretized_state = self.discretizer(state)
        self._table[discretized_state] = value

    #################
    # ValueFunction #
    #################

    def __getitem__(self, state):
        """
        Returns the state value.
        # Arguments
            key: `state`.
        # Returns
            a scalar value.
        """
        discretized_state = self.discretizer(state)
        if discretized_state not in self._table:
            if self.init:
                self._table[discretized_state] = self.random_initializer()
            else:
                self._table[discretized_state] = 0
        return self._table[discretized_state]
