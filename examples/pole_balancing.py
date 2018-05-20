import math

import gym

from myelin.agents import QLearning
from myelin.policies import EGreedy
from myelin.utils import Experience
from myelin.value_functions.tabular_qf import TabularQF


def action_space(state):
    return [0, 1]


def discretizer(state):
    # Number of discrete states (bucket) per state dimension
    NUM_BUCKETS = (1, 1, 6, 3)  # (x, x', theta, theta')
    # Bounds for each discrete state
    STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))
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


env = gym.make('CartPole-v0')
qfunction = TabularQF(discretizer=discretizer)
agent = QLearning(
    policy=EGreedy(action_space, qfunction, epsilon=0.1),
    qfunction=qfunction,
    learning_rate=0.1,
    discount_factor=0.99
)

for episode in range(1000):
    print('Episode: {}'.format(episode))
    state = env.reset()
    env.render()
    for step in range(1000):
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        env.render()
        experience = Experience(state, action, reward, next_state, done)
        agent.update(experience)
        if done:
            print('Steps: {}'.format(step))
            print('-' * 100)
            break
        state = next_state
