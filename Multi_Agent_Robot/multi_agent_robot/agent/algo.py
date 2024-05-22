import numpy as np
from Multi_Agent_Robot.multi_agent_robot.agent.base import Agent
from Multi_Agent_Robot.multi_agent_robot.env.agent_action_space import AgentActionSpace
from Multi_Agent_Robot.multi_agent_robot.env.history import History
from Multi_Agent_Robot.multi_agent_robot.env.types import Action
from config import config


class AlgoAgent(Agent):

    def __init__(self, config_params: dict):
        self.config_params = config_params
        rocks = config.get_in_game_context("environment", "rocks")
        self.action_space = AgentActionSpace(agent_type="robot", n_rocks=len(rocks))

    def act(self, state, history: History) -> Action:
        if all(map(lambda x: x.picked, state.rocks)):
            return self.go_to_exit(state)

        rock_distances = self.calc_rock_distances(state)
        rock_distances[np.asarray(state["collected_rocks"], dtype=bool)] = state["grid_size"][0] * state["grid_size"][1]
        min_rock = list(state["rocks_dict"].keys())[np.argmin(rock_distances)]
        action = Action(action_type=self.go_towards(state, min_rock))
        return action
