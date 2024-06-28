import random
from enum import Enum
from itertools import product
from typing import Tuple, Optional, Union, List, Dict

import numpy as np
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

    def __str__(self):
        return f"RockTile: {self.loc}, {self.reward}, {self.picked}"

    def __eq__(self, other):
        return self.loc == other.loc


class RobotActions(Enum):
    UP = "Up"
    DOWN = "Down"
    LEFT = "Left"
    RIGHT = "Right"
    SAMPLE = "Sample"
    NUM_OF_ACTIONS = "Num of Actions"


class OracleActions(Enum):
    DONT_SEND_DATA = 0
    SEND_GOOD_ROCK = 1
    SEND_BAD_ROCK = 2
    NUM_OF_ACTIONS = 3


class Action(BaseModel):
    action_type: Union[RobotActions, OracleActions]
    rock_sample_loc: Optional[Tuple[int, int]] = None

    @staticmethod
    def sample(rock_sample_loc=None):
        rocks_arr = config.get_in_game_context("environment", "rocks")
        action_type = list(iter(RobotActions))[random.randint(0, 4)]
        if action_type == RobotActions.SAMPLE:
            rock_sample_loc = rock_sample_loc or rocks_arr[random.randint(0, len(rocks_arr) - 1)]
            return Action(action_type=action_type, rock_sample_loc=rock_sample_loc)
        else:
            return Action(action_type=action_type)

    @staticmethod
    def all_actions(state=None):
        cur_agent_loc = state.agent_locations[state.agent_selection]

        all_actions = []
        if cur_agent_loc[0] > 0:
            all_actions.append(Action(action_type=RobotActions.UP))
        if cur_agent_loc[0] < len(state.board) - 1:
            all_actions.append(Action(action_type=RobotActions.DOWN))
        if cur_agent_loc[1] > 0:
            all_actions.append(Action(action_type=RobotActions.LEFT))
        if cur_agent_loc[1] < len(state.board) - 1:
            all_actions.append(Action(action_type=RobotActions.RIGHT))

        for rock in state.rocks:
            if state and not rock.picked:
                all_actions.append(Action(action_type=RobotActions.SAMPLE, rock_sample_loc=rock.loc))

        return all_actions

    def __str__(self):
        return f"Action: {self.action_type}, {self.rock_sample_loc}"

    def ui_repr(self):
        return f"{self.action_type.value}{self.rock_sample_loc or ''}"

    def __hash__(self):
        return hash(str(self))

    def __lt__(self, other):
        return self.action_type.value > other.action_type.value

    def __gt__(self, other):
        return self.action_type.value < other.action_type.value

    def __eq__(self, other):
        return self.action_type == other.action_type and self.rock_sample_loc == other.rock_sample_loc


class State(BaseModel):
    cur_step: int
    board: np.ndarray
    grid_size: Tuple[int, int]
    sample_prob: float
    agents: List[object]  # Agents
    agent_locations: List[Tuple[int, int]]
    agent_selection: int
    rocks: List[RockTile]
    gas_fee: float
    start_pt: Tuple[int, int]
    end_pt: Tuple[int, int]

    @property
    def rocks_map(self) -> Dict[Tuple[int, int], RockTile]:
        return {r.loc: r for r in self.rocks}

    class Config:
        arbitrary_types_allowed = True

    def current_agent_location(self):
        return self.agent_locations[self.agent_selection]

    def collected_rocks(self) -> List[bool]:
        return [rt.picked for rt in self.rocks]

    def __str__(self):
        return f"State: {self.cur_step}, {self.grid_size}, {self.sample_prob}, {self.agent_locations}, {self.agent_selection}, {self.rocks}"

    def __hash__(self):
        res = hash(str(self))
        return res

    def get_all_possible_belief_states(self) -> List["State"]:
        possible_states = []
        items = [1, -1]
        not_picked_rocks = [r for r in self.rocks if not r.picked]
        for rock_beliefs in product(items, repeat=len(not_picked_rocks)):
            s_dict = self.dict()
            s_dict["rocks"] = [RockTile(loc=r.loc, reward=rb*r.reward) for rb, r in zip(rock_beliefs,self.rocks)]
            s = State(**s_dict)
            possible_states.append(s)

        return possible_states

    def num_of_possible_states(self) -> int:
        return 2 ** len([r for r in self.rocks if not r.picked])

    def get_state_index(self, state):
        return state

    def deep_copy(self):
        self_copy = self.copy(deep=True, exclude={'board', 'agents'})
        self_copy.board = self.board
        self_copy.agents = self.agents
        return self_copy


