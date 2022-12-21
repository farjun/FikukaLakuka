import gym
import numpy as np

from config import config

class IState:
    def __init__(self):
        self._action_space = gym.spaces.Discrete(6)
        self.grid_size = config.get_in_game_context("environment", "grid_size")
        assert len(self.grid_size) == 2, "only 2d grid is supported.. common..."
        self.board = gym.spaces.Box(low=self.grid_size[0], high=self.grid_size[1], shape=self.grid_size, dtype=np.int)
        rocks_arr = config.get_in_game_context("environment", "rocks")
        self.rocks = gym.spaces.Space((1, len(rocks_arr)))
        spaces = {"board": self.board, "rocks": self.rocks}


        self._space = gym.spaces.Dict(spaces)

    @property
    def space(self):
        return self._space

    def get_agent_state(self, board: np.array):
        pass

    def print(self):
        print(self.board)

