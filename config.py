import os
from pathlib import Path

import confuse as confuse


class Config(object):
    def __init__(self, path: str):
        self._config = confuse.Configuration('FikukaLakuka', __name__)
        self._config.set_file(path)

    def get(self, *args, start_at=None):
        cur_pos = start_at if start_at is not None else self._config
        for path_key in args:
            cur_pos = cur_pos[path_key]
        return cur_pos.get()

    def get_in_game_context(self, *args):
        cur_game = str(self._config["general"]["game_to_run"])
        game_conf = self._config["games"][cur_game]
        return self.get(*args, start_at=game_conf)

    def get_in_agent_context(self, *args):
        cur_game = str(self._config["general"]["game_to_run"])
        game_conf = self._config["agents"][cur_game]
        return self.get(*args, start_at=game_conf)

config = Config(os.getenv("env", Path(__file__).parent / "config.yaml"))
