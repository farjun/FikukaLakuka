from typing import List

import numpy as np

from config import config
from fikuka_lakuka.fikuka_lakuka.data.api import DataApi
from fikuka_lakuka.fikuka_lakuka.models import History
from fikuka_lakuka.fikuka_lakuka.models.action_space import Action, Actions, Observation
from fikuka_lakuka.fikuka_lakuka.models.agent.base import Agent
from fikuka_lakuka.fikuka_lakuka.models.i_state import IState

def norm_mat(x:np.ndarray)->np.ndarray:
    norm_factor = np.abs(np.min(x))
    x = x + norm_factor
    x[:, 0] = 0
    x[-1, :] = 0
    return x, norm_factor

class BaysianBeliefAgent(Agent):


    def __init__(self, config_params: dict):
        self.config_params = config_params
        rocks = config.get_in_game_context("environment", "rocks")
        self.rock_probs = dict((tuple(x), {Observation.GOOD_ROCK: 0.5, Observation.BAD_ROCK: 0.5}) for x in rocks)
        self.gas_fee = config.get_in_game_context("environment", "gas_fee")
        self.sample_count = [1] * len(rocks)
        self.data_api = DataApi()

    def get_graph_matrix(self, state: IState)->np.ndarray:
        graph_matrix = super().get_graph_matrix(state)
        state_rocks_arr_not_picked = [r for r in state.rocks_arr if r in state.rocks_set]
        for rock, i in zip(state_rocks_arr_not_picked, range(1, graph_matrix.shape[1]-1)):
            graph_matrix[:,i] -= np.tan(self.rock_probs[rock][Observation.GOOD_ROCK]-0.5)*20

        normed_mat, norm_factor = norm_mat(graph_matrix)

        return normed_mat, norm_factor

    def act(self, state: IState, history: History) -> Action:
        if not state.rocks_set:
            return self.go_to_exit(state)
        graph_matrix, norm_factor = self.get_graph_matrix(state)
        dist_matrix, predecessors = self.calc_dijkstra_distance(graph_matrix)
        next_best_idx = np.argmin(dist_matrix[1:])
        next_best_idx_score = dist_matrix[next_best_idx+1]
        state_rocks_arr_not_picked = [r.loc for r in state.rocks if r.loc in state.rocks_set] + [state.end_pt]
        target_loc = state_rocks_arr_not_picked[next_best_idx]
        if next_best_idx_score >= norm_factor and target_loc != state.end_pt:
            return Action(action_type=Actions.SAMPLE, rock_sample_loc = target_loc)

        return Action(action_type=self.go_towards(state, target_loc))

    def update(self, state: IState, reward: float, last_action: Action, observation, history: History) -> List[float]:
        if not history.past:
            return self.get_rock_beliefs(state)

        history_step = history.past[-1]
        if last_action.action_type == Actions.SAMPLE:
            rock_prob = self.rock_probs[last_action.rock_sample_loc]
            if observation == Observation.GOOD_ROCK:
                likelihood = state.calc_good_sample_prob(last_action.rock_sample_loc, Observation.GOOD_ROCK)
                likelihood_of_good_observation_from_a_good_rock = likelihood[0] * rock_prob[Observation.GOOD_ROCK]
                likelihood_of_good_observation_from_a_bad_rock = likelihood[1] * rock_prob[Observation.BAD_ROCK]
                posterior_good_rock_given_good_observation = likelihood_of_good_observation_from_a_good_rock / \
                                                             (
                                                                         likelihood_of_good_observation_from_a_good_rock + likelihood_of_good_observation_from_a_bad_rock)
                good_rock_prob = max([posterior_good_rock_given_good_observation,0])
                bad_rock_prob = 1 - good_rock_prob

            else: # observation == Observation.BAD_ROCK
                likelihood = state.calc_good_sample_prob(last_action.rock_sample_loc, Observation.BAD_ROCK)
                likelihood_of_bad_observation_from_a_good_rock = likelihood[0] * rock_prob[Observation.GOOD_ROCK]
                likelihood_of_bad_observation_from_a_bad_rock = likelihood[1] * rock_prob[Observation.BAD_ROCK]
                posterior_good_rock_given_bad_observation = likelihood_of_bad_observation_from_a_good_rock / \
                                                            (
                                                                        likelihood_of_bad_observation_from_a_bad_rock + likelihood_of_bad_observation_from_a_bad_rock)
                good_rock_prob = max([posterior_good_rock_given_bad_observation,0])
                bad_rock_prob = 1 - good_rock_prob


            self.rock_probs[last_action.rock_sample_loc] = {Observation.GOOD_ROCK: good_rock_prob,
                                                            Observation.BAD_ROCK: bad_rock_prob}

        rock_probs_sorted = [tuple(self.rock_probs[r].values()) for r in state.rocks_arr]
        self.data_api.write_agent_state("bbu", history.cur_step(), np.asarray(rock_probs_sorted))
        return self.get_rock_beliefs(state)

    def get_rock_beliefs(self, state: IState) -> List[float]:
        beliefs = list()
        for rock in state.rocks:
            rock_beliefs = self.rock_probs[rock.loc]
            beliefs.extend([rock_beliefs[Observation.GOOD_ROCK], rock_beliefs[Observation.BAD_ROCK]])
        return beliefs
