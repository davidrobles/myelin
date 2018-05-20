class CallbackList:
    """Container abstracting a list of callbacks (inspired by Keras)."""

    def __init__(self, callbacks=None):
        self.callbacks = callbacks or []

    def on_episode_begin(self, episode):
        """Called at the beginning of every episode."""
        for callback in self.callbacks:
            callback.on_episode_begin(episode)

    def on_episode_end(self, episode, step):
        """Called at the end of every episode."""
        for callback in self.callbacks:
            callback.on_episode_end(episode, step)

    def on_train_begin(self):
        """Called at the beginning of model training."""
        for callback in self.callbacks:
            callback.on_train_begin()

    def on_train_end(self, episode):
        """Called at the end of model training."""
        for callback in self.callbacks:
            callback.on_train_end(episode)

    def on_step(self, step):
        """Called after every time step."""
        for callback in self.callbacks:
            callback.on_step(step)


class Callback:
    """Abstract base class used to build new callbacks (inspired by Keras)."""

    def on_episode_begin(self, episodes):
        """Called at the beginning of every episode."""

    def on_episode_end(self, episode, step):
        """Called at the end of every episode."""

    def on_train_begin(self):
        """Called at the beginning of the agent-environment interaction."""

    def on_train_end(self, episode):
        """Called at the end of the agent-environment interaction."""

    def on_step(self, step):
        """Called after every time step"""
