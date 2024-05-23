import random
from typing import List, Tuple, Dict

import numpy as np
from dijkstar import Graph, find_path
from dijkstar.algorithm import PathInfo

from Multi_Agent_Robot.multi_agent_robot.agent.base import Agent
from Multi_Agent_Robot.multi_agent_robot.data.api import DataApi
from Multi_Agent_Robot.multi_agent_robot.env.history import History
from Multi_Agent_Robot.multi_agent_robot.env.multi_agent_robot import MultiAgentRobotEnv, State, run_one_episode
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

    INTERVEEN_THRESHOLD = -20

    def __init__(self, config_params: dict):
        super().__init__()
        self.config_params = config_params
        self.data_api = DataApi()
        self.gas_fee = config.get_in_game_context("environment", "gas_fee")
        self.inner_simulations = True
        rocks = config.get_in_game_context("environment", "rocks")
        rocks_reward = config.get_in_game_context("environment", "rocks_reward")
        self.rock_probs= dict((tuple(x), {SampleObservation.GOOD_ROCK: 0.5, SampleObservation.BAD_ROCK: 0.5}) for x in rocks)
        self.agents_rock_probs = dict((tuple(x), {SampleObservation.GOOD_ROCK: 0.5, SampleObservation.BAD_ROCK: 0.5}) for x in rocks)
        self.sample_count = dict((tuple(x), 0) for x in rocks)
        self.real_rock_probs = dict((tuple(rock_loc), {SampleObservation.GOOD_ROCK: 1 if reward > 0 else 0,
                                                       SampleObservation.BAD_ROCK: 1 if reward <= 0 else 0}) for rock_loc, reward in zip(rocks, rocks_reward))

    def oracle_act(self, state, last_action: Action, history: History) -> Action:
        """
        the oracle preforms the following:
        1. if not sample -
                simulate some belief vectors
                run a simulation of the robot and see what results in the action given
                set the best belief as the given belief
            if sample -
                we can verify our belief using conclusion as to why the robot sent a sample

        """
        if not self.inner_simulations:
            return Action(action_type=OracleActions.DONT_SEND_DATA)

        optimal_graph = self.get_graph_obj(state, self.real_rock_probs)
        optimal_path = find_path(optimal_graph, 0, optimal_graph.node_count - 1)

        if all(state.collected_rocks()):
            return Action(action_type=OracleActions.DONT_SEND_DATA)

        if last_action.action_type in [RobotActions.SAMPLE]:
            shortest_path, cost_diff = self.get_agent_graph_path(state, optimal_path)
            if cost_diff <= OracleAgent.INTERVEEN_THRESHOLD:
                for path_rock in shortest_path.nodes:
                    rock_loc = state.rocks[path_rock].loc
                    if self.real_rock_probs[rock_loc] == 0:
                        return Action(action_type=OracleActions.SEND_GOOD_ROCK, rock_sample_loc = rock_loc)

        return Action(action_type=OracleActions.DONT_SEND_DATA)
    def enter_inner_simulation_mode(self, beliefs: Dict[tuple, Dict[SampleObservation,float]]):
        self.inner_simulations = False
        self._real_agents_beliefs = self.rock_probs
        self.rock_probs = beliefs

    def exit_inner_simulation_mode(self):
        self.inner_simulations = True
        self.rock_probs = self._real_agents_beliefs

    def get_agent_graph_path(self, state: State, optimal_path: PathInfo) -> float:
        self.enter_inner_simulation_mode(self.agents_rock_probs)
        inner_env = MultiAgentRobotEnv(state.agents)
        sum_reward = run_one_episode(inner_env)
        self.exit_inner_simulation_mode()
        return None, None

