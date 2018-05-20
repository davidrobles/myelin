from myelin import initializers
from myelin.core import ValueFunction
from myelin.utils import check_random_state


class TabularQF(ValueFunction):
    """
    Tabular Q-Function for q(s, a) values.
    # Arguments
        initializer: used to lazy initialize non-existent Q-values.
    """

    def __init__(self, initializer='zeros', discretizer=None, random_state=None):
        self.initializer = initializers.get(initializer)
        self.discretizer = discretizer or (lambda state: state)
        self.random_state = check_random_state(random_state)
        self._table = {}

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
            self._table[state_action] = self.initializer()
        return self._table[state_action]
