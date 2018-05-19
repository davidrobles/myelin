import gym
import numpy as np
from myelin.agents import QLearning
from myelin.policies import RandomPolicy, EGreedy
from myelin.utils import Experience

import math


def action_space(state):
    return [0, 1]


class PoleBalancingQFunction:
    def __init__(self):
        self._table = {}

    def __setitem__(self, state_action, value):
        # state, action = state_action
        self._table[state_action] = value

    def __getitem__(self, key):
        # return self._table[state_action]
        _MEAN = 0.0
        _STD = 0.3

        init = True
        if key not in self._table:
            if init:
                self._table[key] = np.random.normal(_MEAN, _STD)
            else:
                self._table[key] = 0
        return self._table[key]


qfunction = PoleBalancingQFunction()

# policy = RandomPolicy(action_space)
policy = EGreedy(action_space, qfunction, epsilon=0.1)

agent = QLearning(policy, qfunction, learning_rate=0.1, discount_factor=0.99)

env = gym.make('CartPole-v0')
# state = env.reset()

# Number of discrete states (bucket) per state dimension
NUM_BUCKETS = (1, 1, 6, 3)  # (x, x', theta, theta')
# Number of discrete actions
NUM_ACTIONS = env.action_space.n # (left, right)


# Bounds for each discrete state
STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))
STATE_BOUNDS[1] = [-0.5, 0.5]
STATE_BOUNDS[3] = [-math.radians(50), math.radians(50)]


def discretize(state):
    bucket_indice = []
    for i in range(len(state)):
        if state[i] <= STATE_BOUNDS[i][0]:
            bucket_index = 0
        elif state[i] >= STATE_BOUNDS[i][1]:
            bucket_index = NUM_BUCKETS[i] - 1
        else:
            # Mapping the state bounds to the bucket array
            bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
            offset = (NUM_BUCKETS[i]-1)*STATE_BOUNDS[i][0]/bound_width
            scaling = (NUM_BUCKETS[i]-1)/bound_width
            bucket_index = int(round(scaling*state[i] - offset))
        bucket_indice.append(bucket_index)
    return tuple(bucket_indice)


def print_state(state):
    x, x_dot, theta, theta_dot = state
    print("x: {}".format(x))
    print("x_dot: {}".format(x_dot))
    print("theta: {}".format(theta))
    print("theta_dot: {}".format(theta_dot))
    print('-' * 100)


for episode in range(1000):
    # print('-' * 100)
    print('Episode: {}'.format(episode))
    # print('-' * 100)
    state = env.reset()
    # print_state(state)
    env.render()
    for step in range(1000):
        # print('Step: {}'.format(step))
        # d = discretize(state)
        action = agent.get_action(discretize(state))
        next_state, reward, done, info = env.step(action)
        env.render()
        # print('Reward: {}'.format(reward))
        # print_state(next_state)
        experience = Experience(discretize(state), action, reward, discretize(next_state), done)
        agent.update(experience)
        if done:
            print('Steps: {}'.format(step))
            print('-' * 100)
            break
        state = next_state
