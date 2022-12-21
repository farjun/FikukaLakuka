from typing import List

from fikuka_lakuka.fikuka_lakuka.models import Observation, Action


class History:
    def __init__(self, past_observations: List[Observation] = None, past_actions: List[Action] = None):
        self.past_actions = past_actions or list()
        self.past_observations = past_observations or list()

    def add_obs(self, obs: Observation):
        self.past_observations.append(obs)

    def add_action(self, action: Action):
        self.past_observations.append(action)