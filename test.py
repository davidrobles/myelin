import gym
env = gym.make('CartPole-v0')
env.reset()

for _ in range(1000):
    env.render()
    actions = env.action_space
    action = env.action_space.sample()  # take a random action
    observation, reward, done, info = env.step(1)
    print('')
