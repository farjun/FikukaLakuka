import abc
from abc import abstractmethod

from fikuka_lakuka.fikuka_lakuka.models import History, ActionSpace
from fikuka_lakuka.fikuka_lakuka.models.i_state import IState


class Agent(abc.ABC):

    @abstractmethod
    def act(self, state: IState, history: History, action_space: ActionSpace)->ActionSpace:
        pass

    @abstractmethod
    def update(self, reward: float, history: History):
        pass

    @abstractmethod
    def reward(self, action: ActionSpace):  #
        pass