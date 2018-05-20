class MaxEpisodes:
    def __init__(self, n_episodes):
        self.n_episodes = n_episodes

    def __call__(self, info):
        return info['episode'] == self.n_episodes

    def __str__(self):
        return 'Reached max number of episodes: {}'.format(self.n_episodes)


class Convergence:
    def __init__(self, n_episodes, n_steps):
        self.n_episodes = n_episodes
        self.n_steps = n_steps
        self.count = 0

    def __call__(self, info):
        if info['step'] == self.n_steps:
            self.count += 1
        else:
            self.count = 0
        return self.count == self.n_episodes

    def __str__(self):
        return 'Converged after {} consecutive episodes reaching {} steps'.format(self.n_episodes, self.n_steps)
