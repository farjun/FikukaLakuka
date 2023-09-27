from typing import List

import numpy as np

from config import config
from fikuka_lakuka.fikuka_lakuka.data.api import DataApi
from fikuka_lakuka.fikuka_lakuka.models import History
from fikuka_lakuka.fikuka_lakuka.models.action_space import Action, Actions, Observation
from fikuka_lakuka.fikuka_lakuka.models.agent.base import Agent
from fikuka_lakuka.fikuka_lakuka.models.i_state import IState


class BaysianBeliefAgent(Agent):

    def __init__(self, config_params: dict):
        self.config_params = config_params
        rocks = config.get_in_game_context("environment", "rocks")
        self.rock_probs = dict((tuple(x), {Observation.GOOD_ROCK: 0.5, Observation.BAD_ROCK: 0.5}) for x in rocks)
        self.gas_fee = config.get_in_game_context("environment", "gas_fee")
        self.sample_count = [1] * len(rocks)
        self.data_api = DataApi()

    def act(self, state: IState, history: History) -> Action:
        if not state.rocks_set:
            return self.go_to_exit(state)
        tracks = self.calc_tracks_distances(state)
        rock_dists = self.get_rock_distances(state)
        rock_scores = list()
        for i, (dist, rock) in enumerate(zip(rock_dists, state.rocks)):
            if rock.picked:
                rock_scores.append(0)
            else:
                rock_good_prob = self.rock_probs[rock.loc][Observation.GOOD_ROCK]
                rock_bad_prob = self.rock_probs[rock.loc][Observation.BAD_ROCK]
                # Expected utility: P(Rock is good) * R(Rock is good) + P(Rock is bad) * R(Rock is bad)
                expected_utility_from_rock = rock_good_prob * 10 + rock_bad_prob * -10 + self.gas_fee * dist
                rock_exploration_bonus = np.sqrt(np.log(np.sum(self.sample_count)) / self.sample_count[i]) / 2
                rock_scores.append(expected_utility_from_rock + rock_exploration_bonus)

        max_score_rock_idx = rock_scores.index(max(rock_scores))
        # Decide if you should sample a rock
        if rock_scores[max_score_rock_idx] < 5.0:
            self.sample_count[max_score_rock_idx] += 1
            return Action(action_type=Actions.SAMPLE, rock_sample_loc=state.rocks[max_score_rock_idx].loc)

        return Action(action_type=self.go_towards(state, state.rocks[max_score_rock_idx].loc))

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

            else:
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
