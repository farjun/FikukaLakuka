import abc
from abc import abstractmethod
from typing import Tuple, List, Dict

import numpy as np
from dijkstar import Graph
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import cdist

from Multi_Agent_Robot.multi_agent_robot.env.history import History
from Multi_Agent_Robot.multi_agent_robot.env.types import Action, RobotActions, SampleObservation, State
from config import config


class Agent(abc.ABC):

    @abstractmethod
    def act(self, state, history: History) -> Action:
        pass

    def update(self, state, reward: float, last_action: Action, rock_observation, history: History) -> Tuple[List[str], List[str]]:
        return [], [], None

    def calc_rock_distances(self, state):
        return np.linalg.norm(np.asarray(list(state["rocks_dict"].keys())) - np.asarray(state["current_agent_location"]), axis=1, ord=1)

    def calc_distance(self, state, loc: Tuple[int, int]) -> np.float64:
        return np.linalg.norm(np.asarray(loc) - state.current_agent_location(), ord=1)

    def calc_dijkstra_distance(self, graph_matrix: np.array):
        csr_graph_matrix = csr_matrix(graph_matrix)
        # Use dijkstra to find the distance matrix and paths
        dist_matrix, predecessors = shortest_path(csgraph=csr_graph_matrix, return_predecessors=True, indices=0, method='D')
        return dist_matrix, predecessors

    def get_graph_matrix(self, state) -> np.ndarray:
        state_rocks_arr_not_picked = [rock.loc for rock in state.rocks if not rock.picked]
        graph_nodes_num = len(state_rocks_arr_not_picked) + 2
        graph_matrix = np.zeros((graph_nodes_num, graph_nodes_num))
        locations = [state.current_agent_location()] + state_rocks_arr_not_picked + [state.end_pt]
        for i, loc in enumerate(locations):
            graph_matrix[i, :] = np.array(cdist([loc], locations, metric='cityblock')[0])
        graph_matrix *= state.gas_fee
        graph_matrix[:, -1] -= config.get_in_game_context("environment", "end_pt_reward")
        return graph_matrix

    def get_graph_obj(self, state, rock_beliefs = None) -> Graph:
        graph = Graph()
        graph_matrix = self.get_graph_matrix(state)

        state_rocks_arr_not_picked = [rock.loc for rock in state.rocks if not rock.picked]
        if rock_beliefs:
            for rock, i in zip(state_rocks_arr_not_picked, range(1, graph_matrix.shape[1] - 1)):
                graph_matrix[:, i] -= (rock_beliefs[rock][SampleObservation.GOOD_ROCK] - 0.5) * 30

        for i in range(graph_matrix.shape[0]):
            for j in range(graph_matrix.shape[1]):
                graph.add_edge(i, j, graph_matrix[i, j])

        return graph

    def go_to_exit(self, state):
        return Action(action_type=self.go_towards(state, state.end_pt))

    def go_towards(self, state, target: np.ndarray) -> RobotActions:
        cur_loc = state.current_agent_location()

        if target[1] != cur_loc[1]:
            if target[1] > cur_loc[1]:
                return RobotActions.RIGHT

            elif target[1] < cur_loc[1]:
                return RobotActions.LEFT

        if target[0] != cur_loc[0]:
            if target[0] > cur_loc[0]:
                return RobotActions.DOWN

            elif target[0] < cur_loc[0]:
                return RobotActions.UP

        return None

    def get_rock_distances(self, state) -> List[int]:
        dists = list()
        for rock in state.rocks:
            agent_location = state.cur_agent_location()
            dists.append(abs(agent_location[0] - rock.loc[0]) + abs(agent_location[1] - rock.loc[1]))
        return dists

    def get_rock_beliefs(self):
        raise NotImplementedError

    @staticmethod
    def calc_good_sample_prob(state, rock_loc: Tuple[int, int], observation: SampleObservation) -> (float, float):
        location = state.current_agent_location()
        # sensor quality
        # distance to rock
        distance_to_rock = np.linalg.norm(np.array(location) - np.array(rock_loc))
        # measurement error function
        sample_prob_with_distance = 1 / 2 * (1 + np.exp(-(distance_to_rock / 3) * np.log(2) / state.sample_prob))
        if observation == SampleObservation.GOOD_ROCK:
            return sample_prob_with_distance, 1 - sample_prob_with_distance
        if observation == SampleObservation.BAD_ROCK:
            return 1 - sample_prob_with_distance, sample_prob_with_distance

    def get_bu_rock_probs(self, rock_sample_loc:tuple[int,int], rock_prob:dict, observation: SampleObservation, state):
        if observation == SampleObservation.GOOD_ROCK:
            likelihood = self.calc_good_sample_prob(state, rock_sample_loc, SampleObservation.GOOD_ROCK)
            likelihood_of_good_observation_from_a_good_rock = likelihood[0] * rock_prob[SampleObservation.GOOD_ROCK]
            likelihood_of_good_observation_from_a_bad_rock = likelihood[1] * rock_prob[SampleObservation.BAD_ROCK]
            posterior_good_rock_given_good_observation = likelihood_of_good_observation_from_a_good_rock / \
                                                         (
                                                                 likelihood_of_good_observation_from_a_good_rock + likelihood_of_good_observation_from_a_bad_rock)
            good_rock_prob = max([posterior_good_rock_given_good_observation, 0])
            bad_rock_prob = 1 - good_rock_prob

        else:  # observation ==SampleObservation.BAD_ROCK
            likelihood = self.calc_good_sample_prob(state, rock_sample_loc, SampleObservation.BAD_ROCK)
            likelihood_of_bad_observation_from_a_good_rock = likelihood[0] * rock_prob[SampleObservation.GOOD_ROCK]
            likelihood_of_bad_observation_from_a_bad_rock = likelihood[1] * rock_prob[SampleObservation.BAD_ROCK]
            posterior_good_rock_given_bad_observation = likelihood_of_bad_observation_from_a_good_rock / \
                                                        (
                                                                likelihood_of_bad_observation_from_a_bad_rock + likelihood_of_bad_observation_from_a_bad_rock)
            good_rock_prob = max([posterior_good_rock_given_bad_observation, 0])
            bad_rock_prob = 1 - good_rock_prob
        return bad_rock_prob, good_rock_prob

    def get_beliefs_as_db_repr(self, state:State, rock_probs:Dict) -> List[str]:
        beliefs = list()
        for rock in state.rocks:
            rock_beliefs = rock_probs[rock.loc]
            beliefs.append(f"{rock.loc}:{rock_beliefs[SampleObservation.GOOD_ROCK]}")

        return beliefs