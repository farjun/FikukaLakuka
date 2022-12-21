import abc
from abc import abstractmethod

from fikuka_lakuka.fikuka_lakuka.models import History, Action


class Agent(abc.ABC):

    @abstractmethod
    def act(self, history: History)->Action:
        pass

    @abstractmethod
    def update(self, reward: float, history: History):
        pass

    @abstractmethod
    def reward(self, action: Action):  #
        pass