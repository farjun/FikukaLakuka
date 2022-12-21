import abc
from abc import abstractmethod

from fikuka_lakuka.fikuka_lakuka.models import History, Action
from fikuka_lakuka.fikuka_lakuka.models.agent.base import Agent


class RandomAgent(Agent):

    def __init__(self, config_params:dict):
        self.config_params = config_params

    def act(self, history: History) -> Action:
        pass

    def update(self, reward: float, history: History):
        pass

    def reward(self, action: Action):
        pass
