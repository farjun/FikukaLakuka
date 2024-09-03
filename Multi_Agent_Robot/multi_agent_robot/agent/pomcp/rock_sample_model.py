from Multi_Agent_Robot.multi_agent_robot.env.types import State
from config import config
from .util import draw_arg


class RockSampleModel(object):
    def __init__(self):
        self.costs = {}
        self.discount_reward = config.get_in_game_context("environment", "discount_reward")
        self.observations = None
        self.max_depth = None

    def num_states(self, state: State) -> int:
        return state.num_of_possible_states()

    def cost_function(self, action):
        if not self.costs:
            return 0
        return self.costs[action]
