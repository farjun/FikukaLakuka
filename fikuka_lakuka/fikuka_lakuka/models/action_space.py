import random
from enum import Enum
from typing import Optional, Any

import gym
from gym.spaces.space import T_cov

from config import config


class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    SAMPLE = 4


class ActionSpace(gym.spaces.Space):

    def __init__(self):
        self.rocks_arr = config.get_in_game_context("environment", "rocks")
        self.num_of_actions = len(self.rocks_arr) + 4
        super(ActionSpace, self).__init__((1, self.num_of_actions), dtype=int)

    def sample(self, mask: Optional[Any] = None) -> T_cov:
        return random.randint(1, self.num_of_actions)

    def contains(self, x) -> bool:
        pass

    @property
    def space(self):
        return self

    def __contains__(self, item: int):
        return 1 <= item <= self.num_of_actions
