from typing import List, Optional
from pydantic import BaseModel
from Multi_Agent_Robot.multi_agent_robot.env.types import Action, SampleObservation, OracleActions, State

MAX_PLAYERS = 4


class HistoryStep(BaseModel):
    cur_step: Optional[int] = None
    agent_selection: int
    reward: float = 0
    agent_locations: List[List[int]]
    agent_beliefs: Optional[List[str]] = None
    agent_tree: Optional[str] = None
    agent_belief_states: Optional[str] = None
    agent_belief_states_probs: Optional[str] = None
    action: Optional[Action] = None
    observation: Optional[SampleObservation] = None
    oracle_action: Optional[Action] = None
    oracle_beliefs: Optional[List[str]] = None
    state: Optional[State] = None

    def __repr__(self):
        return f"Step: {self.reward}, {self.agent_locations}, {self.action}, {self.observation}, {self.oracle_action}"

    class Config:
        arbitrary_types_allowed = True

    def to_arr(self):
        action = self.action.db_str() if self.action is not None else ""
        oracle_action = self.oracle_action.db_str() if self.oracle_action is not None else ""
        observation_name = self.observation.name if self.observation is not None else ""
        return [self.cur_step,
                str(self.agent_locations[self.agent_selection]),
                action,
                self.reward,
                observation_name,
                str(self.agent_locations),
                str(self.agent_beliefs),
                str(self.agent_tree),
                self.agent_belief_states,
                self.agent_belief_states_probs,
                oracle_action,
                str(self.oracle_beliefs)]


class History:
    TABLE_COLUMNS = (
        "step",
        "cur_agent_location",
        "action",
        "reward",
        "observation",
        "agents_locations",
        "agent_rock_beliefs",
        "agent_tree",
        "agent_belief_states",
        "agent_belief_states_probs",
        "oracle_action",
        "oracle_beliefs"
    )

    def __init__(self, past: List[HistoryStep] = None):
        self.past = past or list()

    def add_step(self, state: State, **kwargs):
        self.past.append(HistoryStep(**state.dict(), **kwargs))

    def to_db_obj(self) -> List[List[int]]:
        return [step.to_arr() for step in self.past]

    def get_last_step_db_obj(self)->list:
        return self.past[-1].to_arr()
