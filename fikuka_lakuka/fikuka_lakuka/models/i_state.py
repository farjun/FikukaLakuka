import random
from collections import defaultdict
from typing import Tuple, List

import gym
import numpy as np

from config import config
from fikuka_lakuka.fikuka_lakuka.models import ActionSpace
from fikuka_lakuka.fikuka_lakuka.models.action_space import Action, Observation, Actions
from pydantic import BaseModel

class RockTile(BaseModel):
    loc: Tuple[int,int]
    ui_loc: int
    reward: float
    picked: bool = False


class IState:
    ROBOT2 = -2
    ROBOT1 = -1
    EMPTY = 0
    START = 1
    END = 2
    ROCK = 3

    def __init__(self):
        self.grid_size: Tuple[int, int] = config.get_in_game_context("environment", "grid_size")
        self.gas_fee = config.get_in_game_context("environment", "gas_fee")
        self._cur_agent_idx = config.get_in_game_context("environment", "starting_agent")
        self.num_of_agents = len(config.get_in_game_context("playing_agents"))

        agents = config.get_in_game_context("playing_agents")
        rocks_reward_arr = config.get_in_game_context("environment", "rocks_reward")
        self.rocks_arr = [tuple(x) for x in config.get_in_game_context("environment", "rocks")]
        self.rocks: List[RockTile] = [RockTile(loc=loc,ui_loc=loc[0]*self.grid_size[0] + loc[1], reward=reward) for loc, reward in zip(self.rocks_arr, rocks_reward_arr)]
        self.rocks_arr_ui = [loc[0]*self.grid_size[0] + loc[1] for loc in self.rocks_arr]
        self.rocks_set = set(tuple(x) for x in self.rocks_arr)
        self.rocks_rewards = dict((tuple(loc), reward) for loc, reward in zip(self.rocks_arr, rocks_reward_arr))
        self.collected_rocks = [False]*len(self.rocks_arr)

        self.start_pt = config.get_in_game_context("environment", "start")
        self.end_pt = config.get_in_game_context("environment", "end")
        assert len(self.grid_size) == 2, "only 2d grid is supported.. common..."
        self._board = np.zeros(self.grid_size, dtype=int)
        self._agent_locations = [self.start_pt.copy() for agent in agents]

        self._board[self.start_pt[0], self.start_pt[1]] = IState.START
        self._board[self.end_pt[0], self.end_pt[1]] = IState.END
        for rock in self.rocks_set:
            self._board[rock[0], rock[1]] = IState.ROCK

    @property
    def cur_agent_idx(self):
        return self._cur_agent_idx

    @property
    def board(self):
        temp_board = self._board.copy()
        for i, agent_pos in enumerate(self._agent_locations):
            temp_board[agent_pos[0], agent_pos[1]] = -(i + 1)
        return temp_board

    def next_agent(self):
        self._cur_agent_idx = (self._cur_agent_idx + 1) % self.num_of_agents

    def sample_rock(self, rock_loc: Tuple[int, int]):
        return random.sample([Observation.BAD_ROCK, Observation.GOOD_ROCK],1)[0]

    def calc_good_sample_prob(self, rock_loc: Tuple[int, int], given_that_rock: Observation)->float:
        location = self.cur_agent_location()
        sample_prob = config.get_in_game_context("environment", "sample_prob")
        manhetten_dist = abs(location[0] - rock_loc[0]) + abs(location[1] - rock_loc[1])
        sample_prob_with_distance = 1 / manhetten_dist * sample_prob
        if given_that_rock == Observation.GOOD_ROCK:
            if self.rocks_rewards[rock_loc] > 0: # good rock
                return sample_prob_with_distance
            else: # bad rock
                return 1 - sample_prob_with_distance

        elif given_that_rock == Observation.BAD_ROCK:
            if self.rocks_rewards[rock_loc] > 0: # good rock
                return 1 - sample_prob_with_distance
            else: # bad rock
                return sample_prob_with_distance
    def update(self, agent: int, action: Action)->Tuple[float, bool, Observation]:
        if self._agent_locations[agent] == self.end_pt:
            return 10, True, Observation.NO_OBS

        if action.action_type == Actions.SAMPLE:
            return 0, False, self.sample_rock(action.rock_sample_loc)

        reward = -self.gas_fee
        # update location
        agent_pos = self._agent_locations[agent]
        board_x, board_y = self._board.shape
        if action.action_type == Actions.LEFT:
            agent_pos[1] = max([0, agent_pos[1] - 1])
        elif action.action_type == Actions.RIGHT:
            agent_pos[1] = min([board_x -1, agent_pos[1] + 1])
        elif action.action_type == Actions.UP:
            agent_pos[0] = max([0, agent_pos[0] - 1])
        elif action.action_type == Actions.DOWN:
            agent_pos[0] = min([board_y -1, agent_pos[0] + 1])

        self._agent_locations[agent] = agent_pos
        agent_pos = tuple(agent_pos)

        # remove rocks
        if agent_pos in self.rocks_set:
            self.rocks[self.rocks_arr.index(agent_pos)].picked = True

            self.collected_rocks[self.rocks_arr.index(agent_pos)] = True
            reward += self.rocks_rewards[agent_pos]
            self._board[agent_pos[0], agent_pos[1]] = IState.EMPTY
            self.rocks_set.remove(agent_pos)

        done = any([pos == self.end_pt for pos in self._agent_locations])

        return reward, done, Observation.NO_OBS

    def print(self):
        print(self.board)

    def get_agent_location(self, agent: int, as_ui_idx = False)->Tuple[int, int]:
        cur_pos = self._agent_locations[agent]
        if as_ui_idx:
            return cur_pos[0]*self.grid_size[0] + cur_pos[1]
        return cur_pos

    def cur_agent_location(self)->Tuple[int, int]:
        return self.get_agent_location(self._cur_agent_idx)

    def agent_locations(self)->Tuple[int, int]:
        return self._agent_locations

    def cur_agent_ui_location(self)->int:
        cur_pos = self.get_agent_location(self._cur_agent_idx)
        return cur_pos[0]*self.grid_size[0] + cur_pos[1]

    def get_end_pt(self, as_ui_idx = False)->Tuple[int, int]:
        if as_ui_idx:
            return self._as_ui_pt(self.end_pt)
        else:
            return self.end_pt

    def _as_ui_pt(self, pt: Tuple[int,int]):
        return pt[0] * self.grid_size[0] + pt[1]