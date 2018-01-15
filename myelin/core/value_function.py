from abc import ABCMeta, abstractmethod


class ValueFunction(metaclass=ABCMeta):

    @abstractmethod
    def __getitem__(self, s_or_sa):
        """Takes a state or a stat-action tuple"""
