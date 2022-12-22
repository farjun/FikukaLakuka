import abc
import random
from abc import abstractmethod

from fikuka_lakuka.fikuka_lakuka.models import History, ActionSpace
from fikuka_lakuka.fikuka_lakuka.models.agent.base import Agent
from fikuka_lakuka.fikuka_lakuka.models.i_state import IState


class RandomAgent(Agent):

    def __init__(self, config_params:dict):
        self.config_params = config_params

    def act(self, state: IState, history: History, action_space: ActionSpace) -> ActionSpace:
        return random.randint(0, action_space.num_of_actions)

    def update(self, reward: float, history: History):
        pass

    def reward(self, action: ActionSpace):
        pass
