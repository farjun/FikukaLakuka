import random
from enum import Enum
from typing import Tuple, Optional
from pydantic import BaseModel

from config import config


class CellType(Enum):
    EMPTY = 0
    START = 1
    END = 2
    ROCK = 3
    ROBOT2 = -2
    ROBOT1 = -1


class SampleObservation(Enum):
    NO_OBS = -1
    BAD_ROCK: int = 0
    GOOD_ROCK: int = 1


class RockTile(BaseModel):
    loc: Tuple[int, int]
    reward: float
    picked: bool = False

    def is_good(self):
        return self.reward > 0


class RobotActions(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    SAMPLE = 4

    NUM_OF_ACTIONS = 5


class Action(BaseModel):
    action_type: RobotActions
    rock_sample_loc: Optional[Tuple[int, int]] = None

    @staticmethod
    def sample(rock_sample_loc=None):
        rocks_arr = config.get_in_game_context("environment", "rocks")
        action_type = RobotActions(random.randint(0, int(RobotActions.NUM_OF_ACTIONS.value) - 1))
        if action_type == RobotActions.SAMPLE:
            rock_sample_loc = rock_sample_loc or rocks_arr[random.randint(0, len(rocks_arr) - 1)]
            return Action(action_type=action_type, rock_sample_loc=rock_sample_loc)
        else:
            return Action(action_type=action_type)
