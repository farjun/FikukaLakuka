import abc
from abc import abstractmethod
from typing import Tuple, List
from scipy.sparse import csr_matrix
import numpy as np
from scipy.sparse.csgraph import dijkstra, shortest_path
from scipy.spatial.distance import euclidean, cdist

from fikuka_lakuka.fikuka_lakuka.models import History, ActionSpace
from fikuka_lakuka.fikuka_lakuka.models.i_state import IState
from fikuka_lakuka.fikuka_lakuka.models.action_space import Action, Actions
from config import config

class Agent(abc.ABC):

    @abstractmethod
    def act(self, state: IState, history: History)->Action:
        pass

    def update(self, state: IState, reward: float, last_action: Action, observation, history: History)->List[float]:
        return []

    def calc_rock_distances(self, state: IState):
        return np.linalg.norm(np.asarray(state.rocks_arr) - state.cur_agent_location(), axis=1)

    def calc_dijkstra_distance(self, graph_matrix: np.array):
        csr_graph_matrix = csr_matrix(graph_matrix)
        dist_matrix, predecessors = shortest_path(csgraph=csr_graph_matrix, return_predecessors=True, indices=0)
        return predecessors

    def get_graph_matrix(self, state: IState)->np.ndarray:
        state_rocks_arr_not_picked = [r for r in state.rocks_arr if r in state.rocks_set]
        graph_nodes_num = len(state_rocks_arr_not_picked) + 2
        graph_matrix = np.zeros((graph_nodes_num, graph_nodes_num))
        locations = [state.cur_agent_location()] + state_rocks_arr_not_picked + [state.end_pt]
        for i, loc in enumerate(locations):
            graph_matrix[i, :] = np.array(cdist([loc], locations, metric='cityblock')[0])

        return graph_matrix

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

