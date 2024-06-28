import argparse
import os
import sys
from pathlib import Path

import confuse as confuse


class Config(object):
    def __init__(self, path: str):
        self._config = confuse.Configuration('FikukaLakuka', __name__)
        self._config.set_file(path)
        self._setup_script_args()
        self.cur_game = self.args.game or str(self._config["general"]["game_to_run"])
    def _setup_script_args(self):
        parser = argparse.ArgumentParser(
            prog='Robots',
            description='runs the robots space game',
            epilog='Robots game description can be found online')
        parser.add_argument("-g", "--game", type=str)
        self.args = parser.parse_args()

    def get(self, *args, start_at=None):
        cur_pos = start_at if start_at is not None else self._config
        for path_key in args:
            cur_pos = cur_pos[path_key]
        return cur_pos.get()

    def get_in_game_context(self, *args):
        game_conf = self._config["games"][self.cur_game]
        return self.get(*args, start_at=game_conf)

    def get_in_agent_context(self, *args):
        game_conf = self._config["agents"]
        return self.get(*args, start_at=game_conf)


config = Config(os.getenv("config_path", Path(__file__).parent / "config.yaml"))
