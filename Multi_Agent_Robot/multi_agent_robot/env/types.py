import random
from enum import Enum
from functools import lru_cache
from itertools import product
from typing import Tuple, Optional, Union, List, Dict

import numpy as np
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


class RockTile:


    def __init__(self, loc, reward, picked=False, rock_number=None):
        self.loc = loc
        self.reward = reward
        self.picked = picked
        self.rock_number=rock_number
        self.rock_number: int = rock_number


    @staticmethod
    def from_rock_tile(other_rock_tile)->"RockTile":
        return RockTile(loc=other_rock_tile.loc, reward=other_rock_tile.reward, picked=other_rock_tile.picked, rock_number=other_rock_tile.rock_number)

    def is_good(self):
        return self.reward > 0

    def __str__(self):
        return f"RockTile: {self.loc}, {self.reward}, {self.picked}"

    def __eq__(self, other):
        return self.loc == other.loc

    def __copy__(self):
        return RockTile(loc=self.loc, reward=self.reward, picked=self.picked, rock_number=self.rock_number)

class RobotActions(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    SAMPLE = 4
    COLLECT_ROCK= 5
    BUY_INFORMATION = 6


class OracleActions(Enum):
    DONT_SEND_DATA = 0
    SEND_GOOD_ROCK = 1
    SEND_BAD_ROCK = 2
    ORACLE_DID_NOT_RUN = 3


class Action:

    def __init__(self, action_type: Union[RobotActions, OracleActions],  rock_sample_loc: Optional[Tuple[int, int]] = None):
        self.action_type = action_type
        self.rock_sample_loc = rock_sample_loc

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
    def all_actions(state, history: list, include_buy_information=False):
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

        buy_info_past_actions = [hist.rock_sample_loc for hist in history if isinstance(hist, Action) and hist.action_type == RobotActions.BUY_INFORMATION]
        for rock in state.rocks:
            if rock.loc == tuple(cur_agent_loc) and not rock.picked:
                all_actions.append(Action(action_type=RobotActions.COLLECT_ROCK, rock_sample_loc=rock.loc))

            if not rock.picked and rock.loc not in buy_info_past_actions:
                all_actions.append(Action(action_type=RobotActions.SAMPLE, rock_sample_loc=rock.loc))
                if include_buy_information:
                    all_actions.append(Action(action_type=RobotActions.BUY_INFORMATION, rock_sample_loc=rock.loc))

        return all_actions

    def __str__(self):
        return f"Action: {self.action_type}, {self.rock_sample_loc}"

    def db_str(self):
        return f"{self.action_type}, {self.rock_sample_loc}"

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


class State:
    ASSUMED_ROCK_REWARD = 15

    def __init__(self, cur_step, grid_size, sample_prob,agents, agent_locations, agent_selection, rocks: RockTile, gas_fee, sample_gas_fee, information_fee, start_pt, end_pt):
        self.cur_step: int = cur_step
        self.grid_size: Tuple[int, int] = grid_size
        self.sample_prob = sample_prob
        self.agents = agents
        self.agent_locations = agent_locations
        self.agent_selection = agent_selection
        self.rocks: list[RockTile] = rocks
        self.gas_fee = gas_fee
        self.sample_gas_fee = sample_gas_fee
        self.information_fee = information_fee
        self.start_pt = start_pt
        self.end_pt = end_pt
        self._rocks_map = None
        self._rock_rewards = None

    def rock_rewards(self):
        if self._rock_rewards is None:
            self._rock_rewards = np.array([r.reward for r in self.rocks])
        return self._rock_rewards

    @property
    def rocks_map(self) -> Dict[Tuple[int, int], RockTile]:
        if self._rocks_map is None:
            self._rocks_map = {r.loc: r for r in self.rocks}
        return self._rocks_map

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
            "sample_gas_fee": self.sample_gas_fee,
            "information_fee": self.information_fee,
            "start_pt": self.start_pt,
            "end_pt": self.end_pt,
        }



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

    @staticmethod
    @lru_cache()
    def get_all_possible_rock_beliefs() -> list[list[RockTile]]:
        items = [1, -1]
        all_possible_rock_beliefs = []
        rocks = config.get_rocks(cast=RockTile)
        for rock_beliefs_to_change in product(items, repeat=len(rocks)):
            possible_rock_belief = [RockTile(loc=tuple(r.loc), reward=rb*State.ASSUMED_ROCK_REWARD, picked=False, rock_number=r.rock_number) for
              rb, r in zip(rock_beliefs_to_change, rocks)]
            all_possible_rock_beliefs.append(possible_rock_belief)

        return  all_possible_rock_beliefs

    def get_all_possible_belief_states(self) -> tuple[List["State"], List[float]]:
        # OPTIMIZED: Cache and reuse state dict to avoid repeated dict() calls
        base_state_dict = self.dict()
        possible_states = []
        possible_states_probs = []
        
        for rock_belief in State.get_all_possible_rock_beliefs():
            # OPTIMIZED: Copy base dict instead of calling self.dict() each time
            s_dict = base_state_dict.copy()
            # OPTIMIZED: Vectorized picked status update
            for r, rb in zip(self.rocks, rock_belief):
                rb.picked = r.picked
            s_dict["rocks"] = rock_belief
            s = State(**s_dict)
            possible_states.append(s)
        return possible_states, possible_states_probs



    def get_states_probs_by_belief(self, belief_probs: Dict[tuple, Dict[SampleObservation, float]]):
        return np.prod([belief_probs[rt.loc][SampleObservation.GOOD_ROCK if rt.reward > 0 else SampleObservation.BAD_ROCK] for rt in self.rocks])

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
            "rocks": [RockTile.from_rock_tile(r) for r in self.rocks],
            "gas_fee": self.gas_fee,
            "sample_gas_fee": self.sample_gas_fee,
            "information_fee": self.information_fee,
            "start_pt": self.start_pt,
            "end_pt": self.end_pt,
        })

    def calc_sample_probs(self, rock_loc: Tuple[int, int]) -> (float, float):
        location = self.current_agent_location()
        # sensor quality
        # distance to rock
        distance_to_rock = np.linalg.norm(np.array(location) - np.array(rock_loc))
        # measurement error function
        sample_prob_with_distance = 1 / 2 * (1 + np.exp(-(distance_to_rock + 1 / 3) * np.log(2) / self.sample_prob))
        return sample_prob_with_distance, 1 - sample_prob_with_distance


