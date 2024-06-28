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


class RockTile(object):
    loc: Tuple[int, int]
    reward: float
    picked: bool

    def __init__(self, loc, reward, picked=False):
        self.loc = loc
        self.reward = reward
        self.picked = picked

    def is_good(self):
        return self.reward > 0

    def __str__(self):
        return f"RockTile: {self.loc}, {self.reward}, {self.picked}"

    def __eq__(self, other):
        return self.loc == other.loc

    def __copy__(self):
        return RockTile(loc=self.loc, reward=self.reward, picked=self.picked)


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
        if cur_agent_loc[0] < state.grid_size[1] - 1:
            all_actions.append(Action(action_type=RobotActions.DOWN))
        if cur_agent_loc[1] > 0:
            all_actions.append(Action(action_type=RobotActions.LEFT))
        if cur_agent_loc[1] < state.grid_size[0] - 1:
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


class State(object):
    cur_step: int
    grid_size: Tuple[int, int]
    sample_prob: float
    agents: List[object]  # Agents
    agent_locations: List[Tuple[int, int]]
    agent_selection: int
    rocks: List[RockTile]
    gas_fee: float
    start_pt: Tuple[int, int]
    end_pt: Tuple[int, int]

    def __init__(self, cur_step, grid_size, sample_prob,agents, agent_locations, agent_selection, rocks, gas_fee, start_pt, end_pt):
        self.cur_step = cur_step
        self.grid_size = grid_size
        self.sample_prob = sample_prob
        self.agents = agents
        self.agent_locations = agent_locations
        self.agent_selection = agent_selection
        self.rocks = rocks
        self.gas_fee = gas_fee
        self.start_pt = start_pt
        self.end_pt = end_pt

    def dict(self):
        return {
            "cur_step": self.cur_step,
            "grid_size": self.grid_size,
            "sample_prob": self.sample_prob,
            "agents": self.agents,
            "agent_locations": self.agent_locations,
            "agent_selection": self.agent_selection,
            "rocks": self.rocks,
            "gas_fee": self.gas_fee,
            "start_pt": self.start_pt,
            "end_pt": self.end_pt,
        }

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
        return State(**{
            "cur_step": self.cur_step,
            "grid_size": self.grid_size,
            "sample_prob": self.sample_prob,
            "agents": self.agents,
            "agent_locations": self.agent_locations.copy(),
            "agent_selection": self.agent_selection,
            "rocks": self.rocks.copy(),
            "gas_fee": self.gas_fee,
            "start_pt": self.start_pt,
            "end_pt": self.end_pt,
        })


