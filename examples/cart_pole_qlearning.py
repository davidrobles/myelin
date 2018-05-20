import math

import gym

from myelin.agents import QLearning
from myelin.core import RLInteraction, GymEnvironment, Agent
from myelin.core.termination import MaxEpisodes, Convergence
from myelin.policies import EGreedy, Greedy
from myelin.utils import Callback
from myelin.value_functions.tabular_qf import TabularQF

#################
# Configuration #
#################


MAX_N_EPISODES = 1000
EPSILON_DECAY = 0.0001
LEARNING_RATE_DECAY = 0.0001


################
# Action Space #
################


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
egreedy = EGreedy(action_space, qfunction, epsilon=1.)
agent = QLearning(
    policy=egreedy,
    qfunction=qfunction,
    learning_rate=0.1,
    discount_factor=0.99
)


#############
# Callbacks #
#############


class Monitor(Callback):
    def on_episode_end(self, episode, step):
        print('Episode: {}'.format(episode))
        print('Steps: {}'.format(step))
        print('-' * 100)
        egreedy.epsilon = egreedy.epsilon * (1. / (1. + EPSILON_DECAY * episode))
        print('Epsilon: {}'.format(egreedy.epsilon))
        agent.learning_rate = agent.learning_rate * (1. / (1. + LEARNING_RATE_DECAY * episode))
        print('Learning rate: {}'.format(agent.learning_rate))


#################################
# Agent-Environment Interaction #
#################################


rl_interaction = RLInteraction(
    env=env,
    agent=agent,
    callbacks=[Monitor()],
    termination_conditions=[
        MaxEpisodes(n_episodes=1000),
        Convergence(n_episodes=5, n_steps=200)
    ]
)

rl_interaction.train()


#################################
# Agent-Environment Interaction #
#################################


class Monitor2(Callback):
    def on_step(self, step):
        gym_env.render()


greedy = Greedy(action_space, qfunction)
env.reset()

rl_interaction = RLInteraction(
    env=env,
    agent=Agent(greedy),
    callbacks=[Monitor2()]
)

rl_interaction.train()
