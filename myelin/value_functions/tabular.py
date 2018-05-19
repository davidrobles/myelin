from myelin.core import ValueFunction
from myelin.utils import check_random_state


class TabularVF(ValueFunction):
    """
    Tabular Value Function.
    Can be used for both state V(s) and state-action Q(s, a) values.
    # Arguments
        init: boolean. Whether to return a randomly initialized value
            when accessed and does not exist in the table.
    # Example
        ```python
        >>> v = TabularVF()
        >>> v[state] = 3.2
        >>> print(q[state])
        3.2
        >>> q = TabularVF()
        >>> q[state, action] = 1.8
        >>> print(q[state, action])
        1.8
        ```
    """

    def __init__(self, init=True, mean=0.0, std=0.3, random_state=None):
        self.init = init
        self.mean = mean
        self.std = std
        self.random_state = check_random_state(random_state)
        self._table = {}

    def random_initializer(self):
        return self.random_state.normal(self.mean, self.std)

    def __setitem__(self, key, value):
        """
        Sets the state or state-action value.
        # Arguments
            key: `state` or `(state, action)`.
            value: a scalar.
        """
        self._table[key] = value

    #################
    # ValueFunction #
    #################

    def __getitem__(self, key):
        """
        Returns the state or state-action value.
        # Arguments
            key: `state` or `(state, action)`.
        # Returns
            a scalar value.
        """
        if key not in self._table:
            if self.init:
                self._table[key] = self.random_initializer()
            else:
                self._table[key] = 0
        return self._table[key]
