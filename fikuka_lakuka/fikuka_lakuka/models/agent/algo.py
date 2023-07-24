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

    def act(self, state: IState, history: History) -> int:
        if not state.rocks_set:
            return self.go_to_exit(state)

        rock_distances = self.calc_rock_distances(state)
        rock_distances[np.asarray(state.collected_rocks, dtype=bool)] = state.grid_size[0] * state.grid_size[1]
        min_rock = state.rocks_arr[np.argmin(rock_distances)]
        action = Action(action_type=self.go_towards(state, min_rock))

        return action


