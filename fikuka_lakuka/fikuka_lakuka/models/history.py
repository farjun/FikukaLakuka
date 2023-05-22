from typing import List

import numpy as np
from pydantic import BaseModel
from fikuka_lakuka.fikuka_lakuka.models import ObservationSpace, ActionSpace
from fikuka_lakuka.fikuka_lakuka.models.action_space import Action, Observation

MAX_PLAYERS=4
class HistoryStep(BaseModel):
    action: Action
    observation: Observation
    reward: float
    players_pos:List[List[int]]

    def to_arr(self):
        return [self.action.action_type.value, self.observation.value, np.array(self.players_pos)]

    @staticmethod
    def from_arr(arr:List[int]):
        return HistoryStep(action=arr[0], observation=arr[1], reward=arr[2], players_pos=arr[3:])

class History:
    def __init__(self, past: List[HistoryStep] = None):
        self.past = past or list()

    def update(self, action:Action, observation:Observation, reward: float, agents_locations:List[int]):
        self.past.append(HistoryStep.from_arr([action, observation, reward] + agents_locations))

    def to_db_obj(self)->List[List[int]]:
        return [step.to_arr() for step in self.past]

