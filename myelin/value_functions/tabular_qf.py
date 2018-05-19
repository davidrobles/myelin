from myelin.core import ValueFunction
from myelin.utils import check_random_state


class TabularQF(ValueFunction):
    """
    Tabular Q-Function for q(s, a) values.
    # Arguments
        init: boolean. Whether to return a randomly initialized value
            when accessed and does not exist in the table.
        ```
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

    def __setitem__(self, key, value):
        """
        Sets the state-action value.
        # Arguments
            key: `(state, action)`.
            value: a scalar.
        """
        state, action = key
        state_action = self.discretizer(state), action
        self._table[state_action] = value

    #################
    # ValueFunction #
    #################

    def __getitem__(self, key):
        """
        Returns the state-action value.
        # Arguments
            key: `(state, action)`.
        # Returns
            a scalar value.
        """
        state, action = key
        state_action = self.discretizer(state), action
        if state_action not in self._table:
            if self.init:
                self._table[state_action] = self.random_initializer()
            else:
                self._table[state_action] = 0
        return self._table[state_action]
