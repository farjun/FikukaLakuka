import random
from typing import Optional, Any

import gym
from gym.spaces.space import T_cov

from config import config


class Action(gym.spaces.Space):

    def __init__(self):
        self.rocks_arr = config.get_in_game_context("environment", "rocks")
        self.num_of_actions = len(self.rocks_arr) + 4
        super(Action, self).__init__((1, self.num_of_actions), dtype=int)
    def sample(self, mask: Optional[Any] = None) -> T_cov:
        return random.randint(1, self.num_of_actions)

    def contains(self, x) -> bool:
        pass
    @property
    def space(self):
        return self

    def __contains__(self, item: int):
        return 1 <= item <= self.num_of_actions
