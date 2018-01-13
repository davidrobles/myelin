import numpy as np


def check_random_state(seed):
    if seed is None or isinstance(seed, int):
        return np.random.RandomState(seed)
    elif isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('Seed should be None, int or np.random.RandomState')


def max_qvalue(state, actions, qf):
    if not actions or actions[0] is None:
        return 0
    return max([qf[state, action] for action in actions])


def min_qvalue(state, actions, qf):
    if not actions or actions[0] is None:
        return 0
    return min([qf[state, action] for action in actions])
