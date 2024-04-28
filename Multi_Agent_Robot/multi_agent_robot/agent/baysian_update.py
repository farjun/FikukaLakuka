import random
from typing import List, Tuple

import numpy as np
from dijkstar import find_path

from Multi_Agent_Robot.multi_agent_robot.agent.oracle import OracleAgent
from Multi_Agent_Robot.multi_agent_robot.env.history import History
from Multi_Agent_Robot.multi_agent_robot.env.types import SampleObservation, Action, RobotActions
from config import config


def norm_mat(x: np.ndarray) -> np.ndarray:
    np_min = np.min(x)
    norm_factor = np.abs(np_min)
    x = x + norm_factor
    x[-1, :] = 0
    return x, norm_factor


class BayesianBeliefAgent(OracleAgent):

    def __init__(self, config_params: dict):
        super().__init__(config_params)
        rocks = config.get_in_game_context("environment", "rocks")
        self.rock_probs = dict(
            (tuple(x), {SampleObservation.GOOD_ROCK: 0.5, SampleObservation.BAD_ROCK: 0.5}) for x in rocks)
        self.sample_count = dict((tuple(x), 0) for x in rocks)

    def get_graph_matrix(self, state, norm_matrix=False) -> np.ndarray:
        graph_matrix = super().get_graph_matrix(state)
        state_rocks_arr_not_picked = [loc for loc, rock in state["rocks_dict"].items() if not rock.picked]
        for rock, i in zip(state_rocks_arr_not_picked, range(1, graph_matrix.shape[1] - 1)):
            graph_matrix[:, i] -= (self.rock_probs[rock][SampleObservation.GOOD_ROCK] - 0.5) * 30

        graph_matrix[:, -1] -= 15
        if norm_matrix:
            graph_matrix, norm_factor = norm_mat(graph_matrix)

        return graph_matrix

    def act(self, state, history: History) -> Action:
        self.oracle_act(state, history)

        if all(map(lambda x: x.picked, state["rocks_dict"].values())):
            return self.go_to_exit(state)
        graph = self.get_graph_obj(state)
        shortest_path = find_path(graph, 0, graph.node_count - 1)
        next_best_idx = shortest_path.nodes[1] - 1
        state_rocks_arr_not_picked = [r.loc for r in state["rocks_dict"].values() if not r.picked] + [state["end_pt"]]
        target_loc = state_rocks_arr_not_picked[next_best_idx]

        if random.random() < 0.5:
            target_loc, sample_count = min(self.sample_count.items(), key=lambda x: x[1])
            return Action(action_type=RobotActions.SAMPLE, rock_sample_loc=target_loc)

        return Action(action_type=self.go_towards(state, target_loc))

    def update(self, state, reward: float, last_action: Action, observation, history: History) -> List[str]:
        if not history.past:
            return self.get_rock_beliefs_as_db_repr(state)

        if last_action.action_type == RobotActions.SAMPLE:
            self.sample_count[last_action.rock_sample_loc] += 1
            rock_prob = self.rock_probs[last_action.rock_sample_loc]
            if SampleObservation == SampleObservation.GOOD_ROCK:
                likelihood = self.calc_good_sample_prob(state, last_action.rock_sample_loc, SampleObservation.GOOD_ROCK)
                likelihood_of_good_observation_from_a_good_rock = likelihood[0] * rock_prob[SampleObservation.GOOD_ROCK]
                likelihood_of_good_observation_from_a_bad_rock = likelihood[1] * rock_prob[SampleObservation.BAD_ROCK]
                posterior_good_rock_given_good_observation = likelihood_of_good_observation_from_a_good_rock / \
                                                             (
                                                                     likelihood_of_good_observation_from_a_good_rock + likelihood_of_good_observation_from_a_bad_rock)
                good_rock_prob = max([posterior_good_rock_given_good_observation, 0])
                bad_rock_prob = 1 - good_rock_prob

            else:  # observation ==SampleObservation.BAD_ROCK
                likelihood = self.calc_good_sample_prob(state, last_action.rock_sample_loc, SampleObservation.BAD_ROCK)
                likelihood_of_bad_observation_from_a_good_rock = likelihood[0] * rock_prob[SampleObservation.GOOD_ROCK]
                likelihood_of_bad_observation_from_a_bad_rock = likelihood[1] * rock_prob[SampleObservation.BAD_ROCK]
                posterior_good_rock_given_bad_observation = likelihood_of_bad_observation_from_a_good_rock / \
                                                            (
                                                                    likelihood_of_bad_observation_from_a_bad_rock + likelihood_of_bad_observation_from_a_bad_rock)
                good_rock_prob = max([posterior_good_rock_given_bad_observation, 0])
                bad_rock_prob = 1 - good_rock_prob

            self.rock_probs[last_action.rock_sample_loc] = {SampleObservation.GOOD_ROCK: good_rock_prob,
                                                            SampleObservation.BAD_ROCK: bad_rock_prob}

        rock_probs_sorted = [tuple(self.rock_probs[r].values()) for r in state["rocks_dict"].keys()]
        self.data_api.write_agent_state("bbu", history.cur_step(), np.asarray(rock_probs_sorted))
        return self.get_rock_beliefs_as_db_repr(state)

    @staticmethod
    def calc_good_sample_prob(state, rock_loc: Tuple[int, int], observation: SampleObservation) -> (float, float):
        location = state["current_agent_location"]
        # sensor quality
        # distance to rock
        distance_to_rock = np.linalg.norm(np.array(location) - np.array(rock_loc))
        # measurement error function
        sample_prob_with_distance = 1 / 2 * (1 + np.exp(-(distance_to_rock / 3) * np.log(2) / state["sample_prob"]))
        if observation == SampleObservation.GOOD_ROCK:
            return sample_prob_with_distance, 1 - sample_prob_with_distance
        if observation == SampleObservation.BAD_ROCK:
            return 1 - sample_prob_with_distance, sample_prob_with_distance

    def get_rock_beliefs_as_db_repr(self, state) -> List[str]:
        beliefs = list()
        for rock in state["rocks_dict"].values():
            rock_beliefs = self.rock_probs[rock.loc]
            beliefs.append(f"{rock.loc}:{rock_beliefs[SampleObservation.GOOD_ROCK]}")

        return beliefs

    def get_rock_beliefs(self) -> List[str]:
        return self.rock_probs

    def update_beliefs(self, rock_loc, is_good):
        if is_good:
            self.rock_probs[rock_loc] = {SampleObservation.GOOD_ROCK: 1, SampleObservation.BAD_ROCK: 0}
        else:
            self.rock_probs[rock_loc] = {SampleObservation.GOOD_ROCK: 0, SampleObservation.BAD_ROCK: 1}

# implement both offline and online
# add offline calc for resilience factor
#
