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
        return np.linalg.norm(np.asarray(state.rocks_arr) - state.cur_agent_location(), axis=1)

    def go_towards(self, state: IState, target: np.ndarray)->Action:
        cur_loc = state.cur_agent_location()
        if target[0] != cur_loc[0]:
            if target[0] > cur_loc[0]:
                return Action.DOWN

            elif target[0] < cur_loc[0]:
                return Action.UP

        if target[1] != cur_loc[1]:
            if target[1] > cur_loc[1]:
                return Action.RIGHT

            elif target[1] < cur_loc[1]:
                return Action.LEFT

        return None

    def act(self, state: IState, history: History) -> int:
        if state.rocks_set:
            rock_distances = self.calc_rock_distances(state)
            rock_distances[np.asarray(state.collected_rocks, dtype=bool)] = state.grid_size[0] * state.grid_size[1]
            min_rock = state.rocks_arr[np.argmin(rock_distances)]
            action = self.go_towards(state, min_rock)

            return action.value
        else:
            return self.go_towards(state, state.end_pt).value

    def update(self, reward: float, history: History):
        pass
