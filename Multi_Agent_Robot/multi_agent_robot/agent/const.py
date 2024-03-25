from Multi_Agent_Robot.multi_agent_robot.agent.base import Agent
from Multi_Agent_Robot.multi_agent_robot.env.agent_action_space import AgentActionSpace
from Multi_Agent_Robot.multi_agent_robot.env.history import History
from Multi_Agent_Robot.multi_agent_robot.env.types import RobotActions, Action
from config import config


class ConstAgent(Agent):

    def __init__(self, config_params: dict):
        self.config_params = config_params
        rocks = config.get_in_game_context("environment", "rocks")
        self.action_space = AgentActionSpace(agent_type="robot", n_rocks=len(rocks))
        self.actions_iter = iter([RobotActions.DOWN] * 10)

    def act(self, state, history: History) -> Action:
        try:
            return Action(action_type=next(self.actions_iter))
        except StopIteration:
            return Action(action_type=RobotActions.SAMPLE, rock_sample_loc=state["rocks_dict"].values()[0].loc)
