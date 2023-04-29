import abc
from abc import abstractmethod

import numpy as np

from fikuka_lakuka.fikuka_lakuka.models import History, ActionSpace
from fikuka_lakuka.fikuka_lakuka.models.i_state import IState
from fikuka_lakuka.fikuka_lakuka.models.action_space import Action


class Agent(abc.ABC):

    @abstractmethod
    def act(self, state: IState, history: History)->ActionSpace:
        pass

    @abstractmethod
    def update(self, reward: float, history: History):
        pass

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