from myelin.core import Agent


class Dummy(Agent):
    """
    Dummy Agent

    # Arguments
        policy: behavior policy.
        vfunction: a state value function.
    """

    def __init__(self, policy):
        super().__init__(policy)

    #########
    # Agent #
    #########

    def update(self, experience):
        pass
