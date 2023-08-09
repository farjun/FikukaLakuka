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
        self.data_api = DataApi()

    def act(self, state: IState, history: History) -> Action:
        if not state.rocks_set:
            return self.go_to_exit(state)
        rock_dists = self.get_rock_distances(state)
        rock_scores = list()
        for dist, rock in zip(rock_dists, state.rocks):
            if rock.picked:
                rock_scores.append(-10)
            else:
                rock_good_prob = self.rock_probs[rock.loc][Observation.GOOD_ROCK]
                rock_scores.append(rock_good_prob - dist*0.1)

        max_score_rock_idx = rock_scores.index(max(rock_scores))
        if rock_scores[max_score_rock_idx] < 0.5:
            return Action(action_type=Actions.SAMPLE, rock_sample_loc=state.rocks[max_score_rock_idx].loc)

        return Action(action_type=self.go_towards(state, state.rocks[max_score_rock_idx].loc))

    def update(self, state: IState, reward: float, history: History)->List[float]:
        history_step = history.past[-1]
        last_action = history_step.action
        if last_action.action_type == Actions.SAMPLE:
            rock_prob = self.rock_probs[last_action.rock_sample_loc]
            if history_step.observation == Observation.GOOD_ROCK:
                good_rock_prob = rock_prob[Observation.GOOD_ROCK] * state.calc_good_sample_prob(last_action.rock_sample_loc, Observation.GOOD_ROCK) + \
                                 rock_prob[Observation.BAD_ROCK] * state.calc_good_sample_prob(last_action.rock_sample_loc, Observation.BAD_ROCK)
                bad_rock_prob = 1 - good_rock_prob

            else:
                # todo - change this to calc_good_sample_prob?
                bad_rock_prob = rock_prob[Observation.GOOD_ROCK] * state.calc_good_sample_prob(last_action.rock_sample_loc, Observation.GOOD_ROCK) + \
                                rock_prob[Observation.BAD_ROCK] * state.calc_good_sample_prob(last_action.rock_sample_loc, Observation.BAD_ROCK)
                good_rock_prob = 1 - bad_rock_prob

            self.rock_probs[last_action.rock_sample_loc] = {Observation.GOOD_ROCK: good_rock_prob,
                                                            Observation.BAD_ROCK: bad_rock_prob}

        rock_probs_sorted = [tuple(self.rock_probs[r].values()) for r in state.rocks_arr]
        self.data_api.write_agent_state("bbu", history.cur_step(), np.asarray(rock_probs_sorted))
        return self.get_rock_beliefs(state)


    def get_rock_beliefs(self, state: IState)->List[float]:
        beliefs = list()
        for rock in state.rocks:
            rock_beliefs = self.rock_probs[rock.loc]
            beliefs.extend([rock_beliefs[Observation.GOOD_ROCK], rock_beliefs[Observation.BAD_ROCK]])
        return beliefs