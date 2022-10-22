from typing import List

from fikuka_lakuka.models import Observation, Action


class History:
    def __init__(self, past_observations: List[Observation], past_actions: List[Action]):
        self.past_actions = past_actions
        self.past_observations = past_observations

