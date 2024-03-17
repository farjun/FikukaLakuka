from typing import List, Optional
from pydantic import BaseModel
from Multi_Agent_Robot.multi_agent_robot.env.types import Action,SampleObservation

MAX_PLAYERS = 4


class HistoryStep(BaseModel):
    cur_agent: int
    action: Optional[Action] = None
    observation: Optional[SampleObservation] = None
    reward: float = 0
    players_pos: List[List[int]]
    agent_beliefs: List[str]

    def to_arr(self):
        action = self.action.action_type.name if self.action is not None else ""
        action_rock_sample_loc = str(self.action.rock_sample_loc) if self.action is not None else ""
        observation_name = self.observation.name if self.observation is not None else ""
        return [self.cur_agent, action, action_rock_sample_loc, observation_name, str(self.players_pos), str(self.agent_beliefs)]

    @staticmethod
    def from_arr(arr: List[int]):
        return HistoryStep(cur_agent=arr[0], action=arr[1], observation=arr[2], reward=arr[3], players_pos=arr[4], agent_beliefs=arr[5])


class History:
    def __init__(self, past: List[HistoryStep] = None):
        self.past = past or list()

    def update(self, cur_agent: int, action: Action, observation: SampleObservation, reward: float, agents_locations: List[List[int]],
               agent_beliefs: List[float]):
        self.past.append(HistoryStep.from_arr([cur_agent, action, observation, reward, agents_locations, agent_beliefs]))

    def to_db_obj(self) -> List[List[int]]:
        return [step.to_arr() for step in self.past]

    def cur_step(self):
        return len(self.past)
