import abc
from abc import abstractmethod

from fikuka_lakuka.models import History, Action


class Agent(abc.ABC):

    @abstractmethod
    def act(self, history: History):
        pass

    @abstractmethod
    def forward(self, history: History):
        pass


    def reward(self, action: Action):  #
        return action.value - self.reservation_price - self.fee * self.offer_counter

    def utility_function(self, action):
        pass