import random
from collections import defaultdict
from typing import Tuple, List, Dict

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

    def is_good(self):
        return self.reward > 0


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
        self.rocks_by_loc: Dict[Tuple[int, int],RockTile] = dict((tuple(loc), rock) for loc, rock in zip(self.rocks_arr, self.rocks))

        self.rocks_set = set(tuple(x) for x in self.rocks_arr)
        self.rocks_rewards = dict((tuple(loc), reward) for loc, reward in zip(self.rocks_arr, rocks_reward_arr))
        self.collected_rocks = [False]*len(self.rocks_arr)

        self.start_pt = config.get_in_game_context("environment", "start")
        self.end_pt = config.get_in_game_context("environment", "end")
        self.sample_prob = config.get_in_game_context("environment", "sample_prob")

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

    def sample_rock(self, agent, rock_loc: Tuple[int, int]):
        agnet_location = self._agent_locations[agent]
        dist = abs(agnet_location[0] - rock_loc[0]) + abs(agnet_location[1] - rock_loc[1])
        p = self.calc_sample_prob(dist)
        rock = self.rocks_by_loc[rock_loc]
        if rock.is_good():
            good_rock_prob, bad_rock_prob = p, 1-p
        else:
            good_rock_prob, bad_rock_prob = 1-p, p

        return np.random.choice([Observation.BAD_ROCK, Observation.GOOD_ROCK],1,p=[bad_rock_prob, good_rock_prob])[0]

    def calc_good_sample_prob(self, rock_loc: Tuple[int, int], observation: Observation)->(float,float):
        location = self.cur_agent_location()
        # sensor quality
        # distance to rock
        distance_to_rock = np.linalg.norm(np.array(location) - np.array(rock_loc))
        # measurement error function
        sample_prob_with_distance = self.calc_sample_prob(distance_to_rock)
        if observation == Observation.GOOD_ROCK:
            return sample_prob_with_distance, 1-sample_prob_with_distance
        if observation == Observation.BAD_ROCK:
            return 1 - sample_prob_with_distance, sample_prob_with_distance

    def calc_sample_prob(self, distance_to_rock):
        distance_to_rock /= 3
        return 1/2 * (1 + np.exp(-distance_to_rock * np.log(2) / self.sample_prob))

    def update(self, agent: int, action: Action)->Tuple[float, bool, Observation]:
        if self._agent_locations[agent] == self.end_pt:
            return 10, True, Observation.NO_OBS

        if action.action_type == Actions.SAMPLE:
            return 0, False, self.sample_rock(agent, action.rock_sample_loc)

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

    def agent_locations(self, cur_agent_first = False)->Tuple[int, int]:
        if cur_agent_first:
            res = self._agent_locations.copy()
            cur_agnt_loc = res.pop(self.cur_agent_idx)
            return [cur_agnt_loc] + res
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