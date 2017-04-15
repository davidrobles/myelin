import numpy as np
from myelin.core import MDP


STAY  = ( 0,  0)
NORTH = (-1,  0)
EAST  = ( 0,  1)
SOUTH = ( 1,  0)
WEST  = ( 0, -1)


class GridWorld(MDP):

    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.terminal_state = self.rand_cell()

    def __str__(self):
        return ('<GridWorld n_rows={} n_cols={} terminal_state={}>'
                .format(self.rows, self.cols, self.terminal_state))

    def is_state_illegal(self, state):
        return state[0] < 0 or state[0] == self.rows or state[1] < 0 or state[1] == self.cols

    def rand_cell(self):
        return (np.random.randint(self.rows), np.random.randint(self.cols))

    #######
    # MDP #
    #######

    def get_actions(self, state):
        return (STAY,) if self.is_terminal(state) else (NORTH, EAST, SOUTH, WEST)

    def get_reward(self, state, action, next_state):
        return 0 if self.is_terminal(state) else -1

    def get_start_state(self):
        return self.rand_cell()

    def get_states(self):
        return [(row, col) for row in range(self.rows) for col in range(self.cols)]

    def get_transitions(self, state, action):
        ns = tuple(np.array(state) + np.array(action))
        return [(state, 1.0)] if self.is_state_illegal(ns) else [(ns, 1.0)]

    def is_terminal(self, state):
        return self.terminal_state == state
