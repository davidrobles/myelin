from abc import ABC, abstractmethod


class ValueFunction(ABC):

    @abstractmethod
    def __getitem__(self, s_or_sa):
        """Takes a state or a stat-action tuple"""
