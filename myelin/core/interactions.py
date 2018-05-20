from myelin.utils import CallbackList, Experience


class RLInteraction:
    """An episodic interaction between an agent and an environment."""

    def __init__(self, env, agent, callbacks=None):
        self.env = env
        self.agent = agent
        self.callbacks = CallbackList(callbacks)
        self.episode = 0
        self.step = 0

    def train(self, n_episodes):
        """Trains the model for a fixed number of episodes."""
        self.callbacks.on_train_begin()
        for _ in range(n_episodes):
            self.callbacks.on_episode_begin(self.episode)
            self.env.reset()
            self.step = 0
            while not self.env.is_terminal():
                state = self.env.get_state()
                action = self.agent.get_action(state)
                reward, next_state = self.env.do_action(action)
                done = self.env.is_terminal()
                experience = Experience(state, action, reward, next_state, done)
                self.agent.update(experience)
                self.step += 1
                self.callbacks.on_step(self.step)
            self.callbacks.on_episode_end(self.episode, self.step)
            self.episode += 1
        self.callbacks.on_train_end(self.episode)
