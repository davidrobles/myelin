import gym

from myelin.agents import QLearning
from myelin.core import RLInteraction, GymEnvironment, Agent
from myelin.core.termination import MaxEpisodes, LessThanStepsConvergence
from myelin.policies import EGreedy, Greedy
from myelin.utils import Callback
from myelin.value_functions import TabularQF

#################
# Configuration #
#################

MAX_N_EPISODES = 1000

INITIAL_EPSILON = 1.0
MIN_EPSILON = 0.01
EPSILON_DECAY = 0.000001

INITIAL_LEARNING_RATE = 1.0
MIN_LEARNING_RATE = 0.01
LEARNING_RATE_DECAY = 0.000001


################
# Action Space #
################


def action_space(state):
    return [0, 1, 2]


gym_env = gym.make('MountainCar-v0')
env = GymEnvironment(gym_env)


n_states = 40


def discretizer(obs):
    """ Maps an observation to state """
    env_low = gym_env.observation_space.low
    env_high = gym_env.observation_space.high
    env_dx = (env_high - env_low) / n_states
    a = int((obs[0] - env_low[0])/env_dx[0])
    b = int((obs[1] - env_low[1])/env_dx[1])
    return a, b


qfunction = TabularQF(initializer='random_uniform', discretizer=discretizer)
egreedy = EGreedy(action_space, qfunction, epsilon=INITIAL_EPSILON)
agent = QLearning(
    policy=egreedy,
    qfunction=qfunction,
    learning_rate=INITIAL_LEARNING_RATE,
    discount_factor=0.99
)


#########
# Learn #
#########


class LearningMonitor(Callback):
    def on_episode_end(self, episode, step):
        print('Episode: {}'.format(episode))
        print('Steps: {}'.format(step))
        print('-' * 100)
        egreedy.epsilon = max(MIN_EPSILON, egreedy.epsilon * (1. / (1. + EPSILON_DECAY * episode)))
        print('Epsilon: {}'.format(egreedy.epsilon))
        agent.learning_rate = max(MIN_LEARNING_RATE, agent.learning_rate * (1. / (1. + LEARNING_RATE_DECAY * episode)))
        print('Learning rate: {}'.format(agent.learning_rate))


RLInteraction(
    env=env,
    agent=agent,
    callbacks=[LearningMonitor()],
    termination_conditions=[
        MaxEpisodes(n_episodes=MAX_N_EPISODES),
        LessThanStepsConvergence(n_episodes=10, n_steps=200)
    ]
).start()


###########
# Perform #
###########


class PerformanceMonitor(Callback):
    def on_step_end(self, step):
        gym_env.render()

    def on_episode_end(self, episode, step):
        print('Steps: {}'.format(step))


greedy = Greedy(action_space, qfunction)
env.reset()

RLInteraction(
    env=env,
    agent=Agent(greedy),
    callbacks=[PerformanceMonitor()],
    termination_conditions=[
        MaxEpisodes(n_episodes=10)
    ]
).start()
