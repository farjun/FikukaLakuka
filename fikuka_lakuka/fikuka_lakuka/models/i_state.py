from collections import defaultdict
from typing import Tuple

import gym
import numpy as np

from config import config
from fikuka_lakuka.fikuka_lakuka.models import ActionSpace
from fikuka_lakuka.fikuka_lakuka.models.action_space import Action
from pydantic import BaseModel
class RockTile(BaseModel):
    loc: Tuple[int,int]
    ui_loc: int
    reward: float
    picked = False


class IState:
    ROBOT2 = -2
    ROBOT1 = -1
    EMPTY = 0
    START = 1
    END = 2
    ROCK = 3

    def __init__(self):
        self.grid_size = config.get_in_game_context("environment", "grid_size")
        self.gas_fee = config.get_in_game_context("environment", "gas_fee")
        self._cur_agent_idx = config.get_in_game_context("environment", "starting_agent")
        self.num_of_agents = len(config.get_in_game_context("playing_agents"))

        agents = config.get_in_game_context("playing_agents")
        rocks_reward_arr = config.get_in_game_context("environment", "rocks_reward")
        self.rocks_arr = [tuple(x) for x in config.get_in_game_context("environment", "rocks")]
        self.rocks = [RockTile(loc=loc,ui_loc=loc[0]*self.grid_size[0] + loc[1], reward=reward) for loc, reward in zip(self.rocks_arr, rocks_reward_arr)]
        self.rocks_arr_ui = [loc[0]*self.grid_size[0] + loc[1] for loc in self.rocks_arr]
        self.rocks_set = set(tuple(x) for x in self.rocks_arr)
        self.rocks_rewards = dict((tuple(loc), reward) for loc, reward in zip(self.rocks_arr, rocks_reward_arr))
        self.collected_rocks = defaultdict(list)

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

    def update(self, agent: int, action: int)->float:
        if self._agent_locations[agent] == self.end_pt:
            return 10, True

        if action >= Action.SAMPLE.value:
            return 0, False

        action = Action(action)
        reward = -self.gas_fee
        # update location
        agent_pos = self._agent_locations[agent]
        board_x, board_y = self._board.shape
        if action == Action.LEFT:
            agent_pos[1] = max([0, agent_pos[1] - 1])
        elif action == Action.RIGHT:
            agent_pos[1] = min([board_x -1, agent_pos[1] + 1])
        elif action == Action.UP:
            agent_pos[0] = max([0, agent_pos[0] - 1])
        elif action == Action.DOWN:
            agent_pos[0] = min([board_y -1, agent_pos[0] + 1])

        self._agent_locations[agent] = agent_pos
        agent_pos = tuple(agent_pos)

        # remove rocks
        if agent_pos in self.rocks_set:
            self.rocks[self.rocks_arr.index(agent_pos)].picked = True

            self.collected_rocks[agent].append(self.rocks_set.remove(agent_pos))
            reward += self.rocks_rewards[agent_pos]
            self._board[agent_pos[0], agent_pos[1]] = IState.EMPTY

        done = all([pos == self.end_pt for pos in self._agent_locations])

        self.next_agent()
        return reward, done

    def print(self):
        print(self.board)

    def get_agent_location(self, agent: int, as_ui_idx = False)->Tuple[int, int]:
        cur_pos = self._agent_locations[agent]
        if as_ui_idx:
            return cur_pos[0]*self.grid_size[0] + cur_pos[1]
        return cur_pos

    def cur_agent_location(self)->Tuple[int, int]:
        return self.get_agent_location(self._cur_agent_idx)

    def cur_agent_ui_location(self)->int:
        cur_pos = self.get_agent_location(self._cur_agent_idx)
        return cur_pos[0]*self.grid_size[0] + cur_pos[1]