from myelin.utils import check_random_state


##########################################################
# Inspired by Keras: https://github.com/keras-team/keras #
##########################################################


class Initializer:
    """Initializer base class: all initializers inherit from this class."""

    def __call__(self):
        raise NotImplementedError


class Zeros(Initializer):
    """Initializer that generates values initialized to 0."""

    def __call__(self):
        return 0


class Ones(Initializer):
    """Initializer that generates values initialized to 1."""

    def __call__(self):
        return 1


class Constant(Initializer):
    """Initializer that generates values initialized to a constant value.
    # Arguments
        value: float; the value of the generator values.
    """

    def __init__(self, value=0):
        self.value = value

    def __call__(self):
        return self.value


class RandomNormal(Initializer):
    """Initializer that generates values with a normal distribution.
    # Arguments
        mean: a python scalar. Mean of the random values to generate.
        stddev: a python scalar. Standard deviation of the random values to generate.
    """

    def __init__(self, mean=0., stddev=0.05, random_state=None):
        self.mean = mean
        self.stddev = stddev
        self.random_state = check_random_state(random_state)

    def __call__(self):
        return self.random_state.normal(self.mean, self.stddev)


class RandomUniform(Initializer):
    """Initializer that generates values with a uniform distribution.
    # Arguments
        minval: A python scalar. Lower bound of the range of random values to generate.
        maxval: A python scalar. Upper bound of the range of random values to generate.
            Defaults to 1 for float types.
        seed: A Python integer. Used to seed the random generator.
    """

    def __init__(self, minval=-0.05, maxval=0.05, random_state=None):
        self.minval = minval
        self.maxval = maxval
        self.random_state = check_random_state(random_state)

    def __call__(self):
        return self.random_state.uniform(self.minval, self.maxval)


def get(identifier):
    if isinstance(identifier, str):
        return {
            'zeros': Zeros(),
            'ones': Ones(),
            'constant': Constant(),
            'random_normal': RandomNormal(),
            'random_uniform': RandomUniform()
        }[identifier]
    elif callable(identifier):
        return identifier
    else:
        raise ValueError('Could not interpret initializer identifier: {}'.format(identifier))
