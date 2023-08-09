import abc
from abc import abstractmethod
from typing import Tuple, List

import numpy as np

from fikuka_lakuka.fikuka_lakuka.models import History, ActionSpace
from fikuka_lakuka.fikuka_lakuka.models.i_state import IState
from fikuka_lakuka.fikuka_lakuka.models.action_space import Action, Actions
from config import config

class Agent(abc.ABC):

    @abstractmethod
    def act(self, state: IState, history: History)->Action:
        pass

    def update(self, state: IState, reward: float, history: History)->List[float]:
        return []

    def calc_rock_distances(self, state: IState):
        return np.linalg.norm(np.asarray(state.rocks_arr) - state.cur_agent_location(), axis=1)

    def go_to_exit(self, state):
        return Action(action_type=self.go_towards(state, state.end_pt))

    def go_towards(self, state: IState, target: np.ndarray)->Actions:
        cur_loc = state.cur_agent_location()
        if target[0] != cur_loc[0]:
            if target[0] > cur_loc[0]:
                return Actions.DOWN

            elif target[0] < cur_loc[0]:
                return Actions.UP

        if target[1] != cur_loc[1]:
            if target[1] > cur_loc[1]:
                return Actions.RIGHT

            elif target[1] < cur_loc[1]:
                return Actions.LEFT

        return None

    def get_rock_distances(self, state: IState)->List[int]:
        dists = list()
        for rock in state.rocks:
            agnet_location = state.cur_agent_location()
            dists.append(abs(agnet_location[0] -rock.loc[0]) + abs(agnet_location[1] -rock.loc[1]))

        return dists

