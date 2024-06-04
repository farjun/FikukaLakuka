import random
from typing import List, Tuple, Dict

import numpy as np
from dijkstar import Graph, find_path
from dijkstar.algorithm import PathInfo

from Multi_Agent_Robot.multi_agent_robot.agent.base import Agent
from Multi_Agent_Robot.multi_agent_robot.data.api import DataApi
from Multi_Agent_Robot.multi_agent_robot.env.history import History
from Multi_Agent_Robot.multi_agent_robot.env.multi_agent_robot import MultiAgentRobotEnv, State, run_one_episode
from Multi_Agent_Robot.multi_agent_robot.env.types import SampleObservation, Action, RobotActions, OracleActions
from config import config


def norm_mat(x: np.ndarray) -> np.ndarray:
    np_min = np.min(x)
    norm_factor = np.abs(np_min)
    x = x + norm_factor
    x[-1, :] = 0
    return x, norm_factor


class OracleAgent(Agent):
    """
    1. Keep track of personal beliefs about the other agents beliefs
    2. Update beliefs based on the other agents' actions
    3. Use the updated beliefs to make decisions, calculate approximate value of information and decide whether to send a message to the
        other agents
    4. Implement the act and update methods
    5. Implement the get_rock_beliefs method
    6. Implement the calc_good_sample_prob method
    7. Implement the update method
    """

    INTERVEEN_THRESHOLD = 3

    def __init__(self, config_params: dict):
        super().__init__()
        self.config_params = config_params
        self.data_api = DataApi(force_recreate=True)
        self.gas_fee = config.get_in_game_context("environment", "gas_fee")
        self.in_a_simulation = False
        rocks = config.get_in_game_context("environment", "rocks")
        rocks_reward = config.get_in_game_context("environment", "rocks_reward")
        self.rock_probs = dict(
            (tuple(x), {SampleObservation.GOOD_ROCK: 0.5, SampleObservation.BAD_ROCK: 0.5}) for x in rocks)
        self.agents_rock_probs = [dict((tuple(x),
                                            {SampleObservation.GOOD_ROCK: 0.5, SampleObservation.BAD_ROCK: 0.5}) for x
                                           in rocks)
                                  ]

        self.sample_count = dict((tuple(x), 0) for x in rocks)
        self.real_rock_probs = dict((tuple(rock_loc), {SampleObservation.GOOD_ROCK: 1 if reward > 0 else 0,
                                                       SampleObservation.BAD_ROCK: 1 if reward <= 0 else 0}) for
                                    rock_loc, reward in zip(rocks, rocks_reward))

    def get_oracles_beliefs_as_db_repr(self, state) -> np.ndarray:
        oracle_beliefs = list()
        for agents_rock_probs in self.agents_rock_probs:
            cur_oracle_beliefs_on_agents_rock_probs = list()
            for rock in state.rocks:
                rock_beliefs = agents_rock_probs[rock.loc]
                cur_oracle_beliefs_on_agents_rock_probs.append(
                    f"{rock.loc}:{rock_beliefs[SampleObservation.GOOD_ROCK]}")

            oracle_beliefs.append(f"{cur_oracle_beliefs_on_agents_rock_probs}")
        return oracle_beliefs

    def oracle_act(self, state, last_action: Action, observation:SampleObservation, history: History):
        """
        the oracle preforms the following:
        1. if not sample -
                simulate some belief vectors
                run a simulation of the robot and see what results in the action given
                set the best belief as the given belief
            if sample -
                we can verify our belief using conclusion as to why the robot sent a sample

        """
        if self.in_a_simulation:
            return Action(action_type=OracleActions.DONT_SEND_DATA)

        if all(state.collected_rocks()):
            return Action(action_type=OracleActions.DONT_SEND_DATA)

        if last_action.action_type in [RobotActions.SAMPLE]:
            self.update_agents_rock_probs_on_agent_sample(last_action.rock_sample_loc, state)
            generated_rock_probs, changed_rock_locs = self.generate_possible_rock_probs()
            sum_rewards = self.simulate_agent_run(state, generated_rock_probs,
                                                  changed_rock_locs)
            optimal_reward = sum_rewards.pop(0)
            max_sum_i, max_sum = max(enumerate(sum_rewards), key=lambda x: x[1])
            if max_sum >= OracleAgent.INTERVEEN_THRESHOLD and changed_rock_locs[max_sum_i] is not None:
                rock_loc = changed_rock_locs[max_sum_i]
                self.update_agents_rock_probs_on_send_data(rock_loc)
                return Action(action_type=OracleActions.SEND_GOOD_ROCK, rock_sample_loc=rock_loc)

        return Action(action_type=OracleActions.DONT_SEND_DATA)

    def generate_possible_rock_probs(self) -> Tuple[List[dict], List]:
        interveen_rock_probs = [self.real_rock_probs.copy()]
        rocks_locs = [None]
        choice = np.random.choice(self.agents_rock_probs, size=min(20, len(self.agents_rock_probs)))
        for chosen_belief in choice:
            for rock_loc in chosen_belief:
                interveened_agents_rock_probs = chosen_belief.copy()
                interveened_agents_rock_probs[rock_loc] = self.real_rock_probs[rock_loc]

                interveen_rock_probs.append(interveened_agents_rock_probs)
                rocks_locs.append(rock_loc)

        return interveen_rock_probs, rocks_locs

    def enter_inner_simulation_mode(self, beliefs: Dict[tuple, Dict[SampleObservation, float]]):
        self.in_a_simulation = True
        self._real_agents_beliefs = self.rock_probs
        self.rock_probs = beliefs

        self._real_agents_sample_counts = self.sample_count.copy()

    def exit_inner_simulation_mode(self):
        self.in_a_simulation = False
        self.rock_probs = self._real_agents_beliefs
        self.sample_count = self._real_agents_sample_counts

    def simulate_agent_run(self, state: State, rock_probs: List[dict], changed_rock_locs: List[tuple]) -> List[float]:
        sum_rewards = list()
        for rock_prob, changed_rock_loc in zip(rock_probs, changed_rock_locs):
            self.enter_inner_simulation_mode(rock_prob.copy())
            inner_env = MultiAgentRobotEnv(state.agents)
            formatted_changed_rock_loc = f"{changed_rock_loc[0]}_{changed_rock_loc[1]}" if changed_rock_loc is not None else "None"
            sum_rewards.append(run_one_episode(inner_env,
                                               schema_name=f"oracle_simulations_{state.cur_step}_{formatted_changed_rock_loc}"))
            self.exit_inner_simulation_mode()

        return sum_rewards

    def update_agents_rock_probs_on_send_data(self, rock_loc):
        #update agent's probs
        self.rock_probs[rock_loc] = self.real_rock_probs[rock_loc]
        # update self probs
        for possible_belief in self.agents_rock_probs:
            possible_belief[rock_loc] = self.real_rock_probs[rock_loc]

    def update_agents_rock_probs_on_agent_sample(self, rock_sample_loc:tuple[int,int], state):
        new_agents_rock_probs = list()
        for possible_belief in self.agents_rock_probs:
            bad_rock_prob, good_rock_prob = self.get_bu_rock_probs(rock_sample_loc, possible_belief[rock_sample_loc], SampleObservation.BAD_ROCK, state)
            if_agent_got_bad_rock = {SampleObservation.GOOD_ROCK : good_rock_prob, SampleObservation.BAD_ROCK : bad_rock_prob}
            bad_rock_prob, good_rock_prob = self.get_bu_rock_probs(rock_sample_loc, possible_belief[rock_sample_loc], SampleObservation.GOOD_ROCK, state)
            if_agent_got_good_rock = {SampleObservation.GOOD_ROCK : good_rock_prob, SampleObservation.BAD_ROCK : bad_rock_prob}

            bad_rock_belief, good_rock_belief = possible_belief.copy(), possible_belief.copy()
            bad_rock_belief[rock_sample_loc] = if_agent_got_bad_rock
            good_rock_belief[rock_sample_loc] = if_agent_got_good_rock
            new_agents_rock_probs.extend([bad_rock_belief, good_rock_belief])

        self.agents_rock_probs = new_agents_rock_probs

# TODO better planning course - somthing the includes the sample prob
