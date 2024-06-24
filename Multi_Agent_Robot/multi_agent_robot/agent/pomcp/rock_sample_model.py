from typing import Tuple

from Multi_Agent_Robot.multi_agent_robot.data.api import DataApi
from Multi_Agent_Robot.multi_agent_robot.env.multi_agent_robot import MultiAgentRobotEnv
from Multi_Agent_Robot.multi_agent_robot.env.types import State, Action, SampleObservation
from config import config
from .util import draw_arg


class RockSampleModel(object):
    def __init__(self):
        """
        Expected attributes in env:
            model_spec
            discount
            costs
            values
            states
            actions
            observations
            T
            Z
            R
        """
        self.costs = {}
        self.discount_reward = config.get_in_game_context("environment", "discount_reward")
        self.actions = Action.all_actions()
        self.observations = None
        self.max_depth = None

    def num_states(self, state:State)->int:
        return state.num_of_possible_states()

    @property
    def num_actions(self):
        return len(self.actions)

    def gen_particles(self, state: State, n, prob=None):
        states = state.get_all_possible_belief_states()
        if prob is None:
            # by default use uniform distribution for particles generation
            prob = [1 / len(states)] * len(states)
        return [hash(states[draw_arg(prob)]) for i in range(n)]

    def get_legal_actions(self, state):
        """
        Simplest situation is every action is legal, but the actual model class
        may handle it differently according to the specific knowledge domain
        :param state:
        :return: actions selectable at the given state
        """
        return self.actions

    def cost_function(self, action):
        if not self.costs:
            return 0
        return self.costs[self.actions.index(action)]

    def simulate_action(self, state: State, ai: Action = None):
        """
        Query the resultant new state, observation and rewards, if action ai is taken from state si

        si: current state
        ai: action taken at the current state
        return: next state, observation and reward
        """
        env = MultiAgentRobotEnv(state.agents)
        observation, reward, done, truncated, info = env.step(action=ai, skip_agent_update=True)
        return hash(env.state), observation, reward, 0

    def take_action(self, action):
        """
        Accepts an action and changes the underlying environment state

        action: action to take
        return: next state, observation and reward
        """
        state, observation, reward = self.simulate_agent_turn(self.curr_state, action)
        self.curr_state = state

        return state, observation, reward

    def simulate_agent_turn(self, state: State, action: Action) -> Tuple[State, SampleObservation, float]:
        inner_env = MultiAgentRobotEnv(state.agents)
        data_api = DataApi(force_recreate=False, schema=f"pomcp_simulations_{state.cur_step}")
        observation, reward, done, truncated, state = inner_env.step(action)
        if done:
            data_api.write_history(inner_env.history)

        return state, observation, reward

