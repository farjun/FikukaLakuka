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

    def act(self, state: IState, history: History) -> int:
        return Action.sample()

    def update(self, state: IState, reward: float, history: History):
        history_step = history.past[-1]
        last_action = history_step.action
        if last_action.action_type == Actions.SAMPLE:
            rock_prob = self.rock_probs[last_action.rock_sample_loc]
            if history_step.observation == Observation.GOOD_ROCK:
                good_rock_prob = rock_prob[Observation.GOOD_ROCK] * state.calc_sample_prob(last_action.rock_sample_loc) + \
                rock_prob[Observation.BAD_ROCK] * (1-state.calc_sample_prob(last_action.rock_sample_loc))
                bad_rock_prob = 1 - good_rock_prob
            else:
                bad_rock_prob = rock_prob[Observation.BAD_ROCK] * state.calc_sample_prob(last_action.rock_sample_loc)+ \
                rock_prob[Observation.GOOD_ROCK] * (1-state.calc_sample_prob(last_action.rock_sample_loc))
                good_rock_prob = 1 - bad_rock_prob

            self.rock_probs[last_action.rock_sample_loc] = {Observation.GOOD_ROCK: good_rock_prob,
                                                            Observation.BAD_ROCK: bad_rock_prob}

        rock_probs_sorted = [tuple(self.rock_probs[r].values()) for r in state.rocks_arr]
        self.data_api.write_agent_state("bbu", history.cur_step(), np.asarray(rock_probs_sorted))
