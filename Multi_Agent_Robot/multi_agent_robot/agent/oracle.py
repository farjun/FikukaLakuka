import random
from typing import List, Tuple

import numpy as np
from dijkstar import Graph, find_path

from Multi_Agent_Robot.multi_agent_robot.agent.base import Agent
from Multi_Agent_Robot.multi_agent_robot.data.api import DataApi
from Multi_Agent_Robot.multi_agent_robot.env.history import History
from Multi_Agent_Robot.multi_agent_robot.env.types import SampleObservation, Action, RobotActions, OracleActions
from config import config


def norm_mat(x: np.ndarray) -> np.ndarray:
    np_min = np.min(x)
    norm_factor = np.abs(np_min)
    x = x + norm_factor
    x[-1, :] = 0
    return x, norm_factor


class OracleAgent(Agent):
    """
    1. Keep track of personal beliefs about the other agents beliefs
    2. Update beliefs based on the other agents' actions
    3. Use the updated beliefs to make decisions, calculate approximate value of information and decide whether to send a message to the
        other agents
    4. Implement the act and update methods
    5. Implement the get_rock_beliefs method
    6. Implement the calc_good_sample_prob method
    7. Implement the update method
    """
    def __init__(self, config_params: dict):
        self.config_params = config_params
        rocks = config.get_in_game_context("environment", "rocks")
        self.rock_probs = dict((tuple(x), {SampleObservation.GOOD_ROCK: 0.5, SampleObservation.BAD_ROCK: 0.5}) for x in rocks)
        self.gas_fee = config.get_in_game_context("environment", "gas_fee")
        self.sample_count = dict((tuple(x), 0) for x in rocks)
        self.data_api = DataApi()

    def oracle_act(self, state, history: History) -> Action:
        if all(map(lambda x: x.picked, state["rocks_dict"].values())):
            return Action(action_type=OracleActions.DONT_SEND_DATA)

        return Action(action_type=OracleActions.DONT_SEND_DATA)

    def update(self, state, reward: float, last_action: Action, observation, history: History) -> List[str]:
        return []
