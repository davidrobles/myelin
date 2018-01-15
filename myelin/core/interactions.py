from myelin.utils import Experience
from myelin.utils.callbacks import CallbackList


class RLInteraction:
    """An episodic interaction between an agent and an environment."""

    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

    def train(self, n_episodes, callbacks=None):
        """Trains the model for a fixed number of episodes."""
        callbacks = CallbackList(callbacks)
        callbacks.on_train_begin()
        for episode in range(n_episodes):
            callbacks.on_episode_begin(episode)
            self.episode()
            callbacks.on_episode_end(episode)
        callbacks.on_train_end(n_episodes)

    def episode(self):
        self.env.reset()
        while not self.env.is_terminal():
            state = self.env.get_state()
            action = self.agent.get_action(state)
            reward, next_state = self.env.do_action(action)
            done = self.env.is_terminal()
            experience = Experience(state, action, reward, next_state, done)
            self.agent.update(experience)
