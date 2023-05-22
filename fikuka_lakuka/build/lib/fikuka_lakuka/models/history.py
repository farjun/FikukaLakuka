from typing import List

from fikuka_lakuka.fikuka_lakuka.models import ObservationSpace, ActionSpace


class History:
    def __init__(self, past_observations: List[int] = None, past_actions: List[ActionSpace] = None):
        self.past_actions = past_actions or list()
        self.past_observations = past_observations or list()

    def add_obs(self, obs: int):
        self.past_observations.append(obs)

    def add_action(self, action: ActionSpace):
        self.past_observations.append(action)