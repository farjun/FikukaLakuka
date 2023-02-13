import abc
import random
from abc import abstractmethod
from typing import Tuple

import numpy as np

from fikuka_lakuka.fikuka_lakuka.models import History, ActionSpace
from fikuka_lakuka.fikuka_lakuka.models.agent.base import Agent
from fikuka_lakuka.fikuka_lakuka.models.i_state import IState
from fikuka_lakuka.fikuka_lakuka.models.action_space import Action


class AlgoAgent(Agent):

    def __init__(self, config_params: dict):
        self.config_params = config_params
        self.action_space = ActionSpace()

    def calc_rock_distances(self, state: IState):
        return np.linalg.norm(state.rocks_arr - state.cur_agent_location(), axis=1)

    def go_towards(self, state: IState, target: np.ndarray)->Action:
        cur_loc = state.cur_agent_location()
        if target[0] != cur_loc[0]:
            if target[0] > cur_loc[0]:
                return Action.RIGHT

            elif target[0] < cur_loc[0]:
                return Action.LEFT

        if target[1] != cur_loc[1]:
            if target[1] > cur_loc[1]:
                return Action.DOWN

            elif target[1] < cur_loc[1]:
                return Action.UP

        return None

    def act(self, state: IState, history: History) -> int:
        rock_distances = self.calc_rock_distances(state)
        min_rock = state.rocks_arr[np.argmin(rock_distances)]
        action = self.go_towards(state, min_rock)
        if action:
            return action.value
        else:
            return self.go_towards(state, state.end_pt).value

    def update(self, reward: float, history: History):
        pass
