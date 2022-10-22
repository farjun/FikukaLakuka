import gym
import numpy as np

from config import config

class IState:
    def __init__(self):
        self._action_space = gym.spaces.Discrete(6)
        self.grid_size = config.get_in_game_context("environment", "grid_size")
        assert len(self.grid_size) == 2, "only 2d grid is supported.. common..."
        self.board = gym.spaces.Box(low=self.grid_size[0], high=self.grid_size[1], shape=self.grid_size,
                                dtype=np.int)

        spaces = {"board": self.board}

        self._space = gym.spaces.Dict(spaces)


    def get_agent_state(self, board: np.array):
        pass

    def print(self):
        print(self.board)

