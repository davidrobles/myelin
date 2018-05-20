import math

import gym

from myelin.agents import QLearning
from myelin.core import RLInteraction, GymEnvironment
from myelin.policies import EGreedy
from myelin.utils import Callback
from myelin.value_functions.tabular_qf import TabularQF


def action_space(state):
    return [0, 1]


gym_env = gym.make('CartPole-v0')
env = GymEnvironment(gym_env)


def discretizer(state):
    # Number of discrete states (bucket) per state dimension
    NUM_BUCKETS = (1, 1, 6, 3)  # (x, x', theta, theta')
    # Bounds for each discrete state
    STATE_BOUNDS = list(zip(gym_env.observation_space.low, gym_env.observation_space.high))
    STATE_BOUNDS[1] = [-0.5, 0.5]
    STATE_BOUNDS[3] = [-math.radians(50), math.radians(50)]
    bucket_indice = []
    for i in range(len(state)):
        if state[i] <= STATE_BOUNDS[i][0]:
            bucket_index = 0
        elif state[i] >= STATE_BOUNDS[i][1]:
            bucket_index = NUM_BUCKETS[i] - 1
        else:
            # Mapping the state bounds to the bucket array
            bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
            offset = (NUM_BUCKETS[i] - 1) * STATE_BOUNDS[i][0] / bound_width
            scaling = (NUM_BUCKETS[i] - 1) / bound_width
            bucket_index = int(round(scaling * state[i] - offset))
        bucket_indice.append(bucket_index)
    return tuple(bucket_indice)


qfunction = TabularQF(discretizer=discretizer)
agent = QLearning(
    policy=EGreedy(action_space, qfunction, epsilon=0.1),
    qfunction=qfunction,
    learning_rate=0.1,
    discount_factor=0.99
)


class Monitor(Callback):
    def on_episode_end(self, episode, step):
        print('Episode: {}'.format(episode))
        print('Steps: {}'.format(step))
        print('-' * 100)


RLInteraction(env, agent, callbacks=[Monitor()]).train(1000)
