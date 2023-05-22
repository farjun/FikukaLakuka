from gym import Space

from config import config


class ObservationSpace:
    def __init__(self):
        rocks_arr = config.get_in_game_context("environment", "rocks")
        self.space = Space((1, len(rocks_arr) + 4))
