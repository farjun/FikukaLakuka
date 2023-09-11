from typing import List

import numpy as np
from pydantic import BaseModel
from fikuka_lakuka.fikuka_lakuka.models import ObservationSpace, ActionSpace
from fikuka_lakuka.fikuka_lakuka.models.action_space import Action, Observation

MAX_PLAYERS=4
class HistoryStep(BaseModel):
    cur_agent: int
    action: Action
    observation: Observation
    reward: float
    players_pos:List[List[int]]
    agent_beliefs: List[float]

    def to_arr(self):
        return [self.cur_agent, self.action.action_type.name, self.observation.name, str(self.players_pos), np.array(self.agent_beliefs)]

    @staticmethod
    def from_arr(arr:List[int]):
        return HistoryStep(cur_agent=arr[0],action=arr[1], observation=arr[2], reward=arr[3], players_pos=arr[4], agent_beliefs=arr[5])

class History:
    def __init__(self, past: List[HistoryStep] = None):
        self.past = past or list()

    def update(self, cur_agent: int, action:Action, observation:Observation, reward: float, agents_locations:List[int], agent_beliefs:List[float]):
        self.past.append(HistoryStep.from_arr([cur_agent, action, observation, reward, agents_locations, agent_beliefs]))

    def to_db_obj(self)->List[List[int]]:
        return [step.to_arr() for step in self.past]

    def cur_step(self):
        return len(self.past)


