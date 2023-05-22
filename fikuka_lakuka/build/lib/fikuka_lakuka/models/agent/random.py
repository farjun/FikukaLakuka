import abc
import random
from abc import abstractmethod

from fikuka_lakuka.fikuka_lakuka.models import History, ActionSpace
from fikuka_lakuka.fikuka_lakuka.models.agent.base import Agent
from fikuka_lakuka.fikuka_lakuka.models.i_state import IState
from fikuka_lakuka.fikuka_lakuka.models.action_space import Action


class RandomAgent(Agent):

    def __init__(self, config_params: dict):
        self.config_params = config_params
        self.action_space = ActionSpace()

    def act(self, state: IState, history: History) -> Action:
        return random.randint(1, self.action_space.num_of_actions)

    def update(self, reward: float, history: History):
        pass
