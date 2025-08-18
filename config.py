import argparse
import os
import sys
from functools import lru_cache
from pathlib import Path

import confuse as confuse
import numpy as np


class Config(object):
    def __init__(self, path: str):
        self._config = confuse.Configuration('FikukaLakuka', __name__)
        self._config.set_file(path)
        self._setup_script_args()
        self.games_to_run = [str(name) for name in self._config["general"]["games_to_run"]]

        self.cur_game: str | None = None
        self.seed: int | None = None

    def set_game(self, game_name: str):
        self.cur_game = game_name

    def set_seed(self, seed: int):
        self.seed: int = seed
        np.random.seed(self.seed)

    def _setup_script_args(self):
        parser = argparse.ArgumentParser(
            prog='Robots',
            description='runs the robots space game',
            epilog='Robots game description can be found online')
        parser.add_argument("-g", "--game", type=str)
        self.args = parser.parse_args()

    @lru_cache(maxsize=1000)
    def get(self, *args, start_at=None):
        cur_pos = start_at if start_at is not None else self._config
        for path_key in args:
            cur_pos = cur_pos[path_key]
            if path_key == 'environment':
                cur_pos = self._config['environments'][cur_pos.get()]
        return cur_pos.get()

    def get_in_game_context(self, *args, game_name=None):
        game_conf = self._config["games"][game_name or self.cur_game]
        return self.get(*args, start_at=game_conf)

    def get_in_agent_context(self, *args):
        game_conf = self._config["agents"]
        return self.get(*args, start_at=game_conf)

    def get_rocks(self, game_name: str = None, cast=None):
        rocks_arr = self.get_in_game_context("environment", "rocks", game_name=game_name)
        rocks_reward_arr = self.get_in_game_context("environment", "rocks_reward", game_name= game_name)
        if cast is not None:
            return [cast(loc, reward) for loc, reward in zip(rocks_arr, rocks_reward_arr)]
        else:
            return rocks_arr, rocks_reward_arr


config = Config(os.getenv("config_path", Path(__file__).parent / "config.yaml"))
