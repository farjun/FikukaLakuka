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
        state_rocks_arr_not_picked = [rock.loc for rock in state.rocks if not rock.picked]
        for rock, i in zip(state_rocks_arr_not_picked, range(1, graph_matrix.shape[1] - 1)):
            graph_matrix[:, i] -= (self.rock_probs[rock][SampleObservation.GOOD_ROCK] - 0.5) * 30

        if norm_matrix:
            graph_matrix, norm_factor = norm_mat(graph_matrix)

        return graph_matrix

    def act(self, state, history: History) -> Action:
        if all(state.collected_rocks()):
            return self.go_to_exit(state)

        graph = self.get_graph_obj(state, self.get_rock_beliefs())
        shortest_path = find_path(graph, 0, graph.node_count - 1)
        next_best_idx = shortest_path.nodes[1] - 1
        possible_targets = [rock.loc for rock in state.rocks if not rock.picked] + [state.end_pt]
        target_loc = possible_targets[next_best_idx]
        agent_confidenc = abs(sum([self.rock_probs[r][SampleObservation.GOOD_ROCK] for r in self.rock_probs]) / len(self.rock_probs) - 0.5)
        if agent_confidenc < 0.2:
            target_loc, sample_count = min(self.sample_count.items(), key=lambda x: x[1] + abs(0.5-self.rock_probs[x[0]][SampleObservation.GOOD_ROCK])*100)
            return Action(action_type=RobotActions.SAMPLE, rock_sample_loc=target_loc)

        return Action(action_type=self.go_towards(state, target_loc))

    def update(self, state, reward: float, last_action: Action, observation, history: History) -> Tuple[List[str], List[str], Action]:
        oracle_action = self.oracle_act(state, last_action, history)

        if not history.past:
            return *self.get_beliefs_as_db_repr(state), oracle_action

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

        return *self.get_beliefs_as_db_repr(state), oracle_action

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

    def get_beliefs_as_db_repr(self, state) -> Tuple[List[str], List[str]]:
        oracle_beliefs = self.get_oracles_beliefs_as_db_repr(state)
        beliefs = list()
        for rock in state.rocks:
            rock_beliefs = self.rock_probs[rock.loc]
            beliefs.append(f"{rock.loc}:{rock_beliefs[SampleObservation.GOOD_ROCK]}")

        return beliefs, oracle_beliefs

    def get_rock_beliefs(self) -> List[str]:
        return self.rock_probs



# implement full graph dijstra
# find a way to consider sample prob,

