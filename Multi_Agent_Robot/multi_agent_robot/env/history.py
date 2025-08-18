from typing import List, Optional
from pydantic import BaseModel
from Multi_Agent_Robot.multi_agent_robot.env.types import Action, SampleObservation, OracleActions, State, RobotActions

MAX_PLAYERS = 4


class HistoryStep(BaseModel):
    cur_step: Optional[int] = None
    agent_selection: int
    reward: float = 0
    agent_locations: List[List[int]]
    agent_beliefs: Optional[List[str]] = None
    agent_q_values: Optional[list[tuple]] = None
    agent_belief_states: Optional[str] = None
    agent_belief_states_probs: Optional[str] = None
    action: Optional[Action] = None
    observation: Optional[SampleObservation] = None
    oracle_action: Optional[Action] = None
    oracle_beliefs: Optional[List[str]] = None
    oracle_beliefs_dist: Optional[List[str]] = None
    state: Optional[State] = None
    oracle_last_action_generated_q_values : Optional[list] = []
    oracle_belief_scores : Optional[list] = []
    oracle_simulation_beliefs : Optional[list] = []
    cur_simulated_changed_rock : Optional[tuple] = None
    unchanged_rock_beliefs : Optional[list] = None

    def __repr__(self):
        return f"Step: {self.cur_step} {self.reward}, {self.agent_locations}, {self.action}, {self.observation}, {self.oracle_action}"

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
                str(self.agent_q_values),
                self.agent_belief_states,
                self.agent_belief_states_probs,
                oracle_action,
                str(self.oracle_beliefs),
                str(self.oracle_beliefs_dist),
                str(self.oracle_last_action_generated_q_values),
                str(self.oracle_belief_scores),
                str(self.oracle_simulation_beliefs),
                str(self.cur_simulated_changed_rock),
                str(self.unchanged_rock_beliefs),
                ]


class History:
    TABLE_COLUMNS = (
        "step",
        "cur_agent_location",
        "action",
        "reward",
        "observation",
        "agents_locations",
        "agent_rock_beliefs",
        "agent_q_values",
        "agent_belief_states",
        "agent_belief_states_probs",
        "oracle_action",
        "oracle_beliefs",
        "oracle_beliefs_dist",
        "oracle_last_action_generated_q_values",
        "oracle_belief_scores",
        "oracle_simulation_beliefs",
        "cur_simulated_changed_rock",
        "unchanged_rock_beliefs"

    )

    def __init__(self, past: List[HistoryStep] = None, states: List[State] = None):
        self.past = past or list()
        self.states = states or list()

    def add_step(self, state: State, **kwargs):
        self.past.append(HistoryStep(**state.dict(), **kwargs))
        self.states.append(state)

    def to_db_obj(self) -> List[List[int]]:
        return [step.to_arr() for step in self.past]

    def get_last_step_db_obj(self)->list:
        return self.past[-1].to_arr()

    def get_actions(self, filter_action_type = None):
        all_actions = [h_step.action for h_step in self.past]
        if filter_action_type is not None:
            all_actions = [a for a in all_actions if a.action_type == filter_action_type]
        return all_actions

    def get_q_values(self, filter_action_type = None):
        all_agent_q_values = [h_step.agent_q_values for h_step in self.past]
        if filter_action_type is not None:
            filtered_q_values = []
            for q_values in all_agent_q_values:
                q_values = [v for v in q_values if v[1].action.action_type == filter_action_type]
                filtered_q_values.append(q_values)
            return filtered_q_values

        return all_agent_q_values
