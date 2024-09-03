import time
from typing import List, Tuple, Dict

import numpy as np
from pydantic.v1.schema import schema

from Multi_Agent_Robot.multi_agent_robot.agent.base import Agent
from Multi_Agent_Robot.multi_agent_robot.data.api import DataApi
from Multi_Agent_Robot.multi_agent_robot.env.history import History
from Multi_Agent_Robot.multi_agent_robot.env.multi_agent_robot import MultiAgentRobotEnv, State
from Multi_Agent_Robot.multi_agent_robot.env.types import SampleObservation, Action, RobotActions, OracleActions
from config import config
from tqdm import tqdm


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

    INTERVEEN_THRESHOLD = 10
    MAX_ALLOWED_AGENT_DISTS = 4

    def __init__(self, config_params: dict):
        super().__init__()
        self.config_params = config_params
        self.data_api = DataApi(db_name=f'oracle_db_{config.get("general", "game_to_run")}', force_recreate=True)
        self.gas_fee = config.get_in_game_context("environment", "gas_fee")
        self.in_a_simulation = False
        rocks = config.get_in_game_context("environment", "rocks")
        rocks_reward = config.get_in_game_context("environment", "rocks_reward")
        self.rock_probs = dict((tuple(x), {SampleObservation.GOOD_ROCK: 0.5, SampleObservation.BAD_ROCK: 0.5}) for x in rocks)
        self.agent_beliefs_elements = [
            dict((tuple(x), {SampleObservation.GOOD_ROCK: 0.5, SampleObservation.BAD_ROCK: 0.5}) for x in rocks)]
        self.agent_beliefs_dist = [1]

        self.sample_count = dict((tuple(x), 0) for x in rocks)
        self.real_rock_probs = dict((tuple(rock_loc), {SampleObservation.GOOD_ROCK: 1 if reward > 0 else 0,
                                                       SampleObservation.BAD_ROCK: 1 if reward <= 0 else 0}) for
                                    rock_loc, reward in zip(rocks, rocks_reward))

    def get_oracles_beliefs_as_db_repr(self, state) -> np.ndarray:
        oracle_beliefs = list()
        for agents_rock_probs in self.agent_beliefs_elements:
            cur_oracle_beliefs_on_agents_rock_probs = list()
            for rock in state.rocks:
                rock_beliefs = agents_rock_probs[rock.loc]
                cur_oracle_beliefs_on_agents_rock_probs.append(
                    f"{rock.loc}:{rock_beliefs[SampleObservation.GOOD_ROCK]}")

            oracle_beliefs.append(f"{cur_oracle_beliefs_on_agents_rock_probs}")
        return oracle_beliefs

    def oracle_act(self, state, last_action: Action, observation: SampleObservation, history: History):
        """
        the oracle preforms the following:
        1. if not sample -
                simulate some belief vectors
                run a simulation of the robot and see what results in the action given
                set the best belief as the given belief
            if sample -
                we can verify our belief using conclusion as to why the robot sent a sample
        """
        if True:
            return Action(action_type=OracleActions.DONT_SEND_DATA)

        if self.in_a_simulation or all(state.collected_rocks()):
            return Action(action_type=OracleActions.DONT_SEND_DATA)

        if last_action.action_type in [RobotActions.BUY_INFORMATION]:
            self.update_agents_rock_probs_on_send_data(last_action.rock_sample_loc)

        if last_action.action_type in [RobotActions.SAMPLE]:
            self.update_agents_rock_probs_on_agent_sample(last_action.rock_sample_loc, state)

        no_interveen_rock_probs, interveen_rock_probs, changed_rock_locs = self.generate_possible_rock_probs()
        no_interveen_sum_rewards, no_interveen_histories = self.simulate_agent_run(state, no_interveen_rock_probs,  history, f"oracle_simulation_no_help_step_{state.cur_step}")

        if len(self.agent_beliefs_elements) > self.MAX_ALLOWED_AGENT_DISTS:
            simulated_actions = [simulated_history.past[0].action for simulated_history in no_interveen_histories]
            matching_simulated_actions = [last_action == simulated_action for simulated_action in simulated_actions]
            if any(matching_simulated_actions):
                matching_action_likelihood = 1/len(state.rocks)
                for i, likelyhood in enumerate(matching_simulated_actions):
                    self.agent_beliefs_dist[i] *= 1+matching_action_likelihood if likelyhood else matching_action_likelihood
                probs_sum = sum(self.agent_beliefs_dist)
                self.agent_beliefs_dist = [prob/probs_sum for prob in self.agent_beliefs_dist]
                self.agent_beliefs_elements = sorted(self.agent_beliefs_elements, key=lambda x: self.agent_beliefs_dist[self.agent_beliefs_elements.index(x)], reverse=True)
                self.agent_beliefs_dist = sorted(self.agent_beliefs_dist)

            self.agent_beliefs_elements = self.agent_beliefs_elements[:self.MAX_ALLOWED_AGENT_DISTS]
            self.agent_beliefs_dist = self.agent_beliefs_dist[:self.MAX_ALLOWED_AGENT_DISTS]

        # optimal_reward = self.simulate_agent_run(state, [self.real_rock_probs.copy()], history)[0][0]
        # todo if buy info in histroy's next step check if it is worth it
        # todo worth it = does it need the info? based on the reward
        interveen_sum_rewards, interveen_histories = self.simulate_agent_run(state, interveen_rock_probs, history, f"oracle_simulation_help_step_{state.cur_step}")
        max_send_data_reward_i, max_send_data_reward = max(enumerate(interveen_sum_rewards), key=lambda x: x[1])

        max_no_send_data_reward = max(no_interveen_sum_rewards)
        if (max_send_data_reward > max_no_send_data_reward and
                max_send_data_reward - max_no_send_data_reward >= OracleAgent.INTERVEEN_THRESHOLD):
            max_reward_hist = interveen_histories[max_send_data_reward_i]
            if RobotActions.BUY_INFORMATION in [hs.action.action_type for hs in max_reward_hist.past]:
                rock_loc = changed_rock_locs[max_send_data_reward_i]
                self.update_agents_rock_probs_on_send_data(rock_loc)
                return Action(action_type=OracleActions.SEND_GOOD_ROCK, rock_sample_loc=rock_loc)

        return Action(action_type=OracleActions.DONT_SEND_DATA)

    def generate_possible_rock_probs(self) -> Tuple[List[dict], List[dict], List]:
        interveen_rock_probs = []
        no_interveen_rock_probs = []
        rocks_locs = []
        for chosen_belief in self.agent_beliefs_elements:
            no_interveen_rock_probs.append(chosen_belief.copy())
            for rock_loc in chosen_belief:
                interveened_agents_rock_probs = chosen_belief.copy()
                interveened_agents_rock_probs[rock_loc] = self.real_rock_probs[rock_loc]
                interveen_rock_probs.append(interveened_agents_rock_probs)
                rocks_locs.append(rock_loc)

        num_of_interveen_simulations = min(4, len(interveen_rock_probs))
        interveen_rock_probs = np.random.choice(interveen_rock_probs, size=num_of_interveen_simulations)
        return no_interveen_rock_probs, interveen_rock_probs, rocks_locs

    def enter_inner_simulation_mode(self, beliefs: Dict[tuple, Dict[SampleObservation, float]], backup_data=True):
        self.in_a_simulation = True
        self._real_agents_beliefs = self.rock_probs
        self.rock_probs = beliefs

    def exit_inner_simulation_mode(self, restore_data=True):
        self.in_a_simulation = False
        self.rock_probs = self._real_agents_beliefs

    def simulate_agent_run(self, state: State, rock_probs: List[dict], history: History, hist_table_name: str, steps=7) -> Tuple[
        List[float], List[History]]:
        sum_rewards = list()
        histories = list()
        env = MultiAgentRobotEnv(state.agents)
        for i, rock_prob in tqdm(enumerate(rock_probs),total=len(rock_probs), desc=f"Oracle is Simulating agent runs at step {state.cur_step}, writing to table {hist_table_name} for {steps} steps"):
            simulation_history = History()
            self.enter_inner_simulation_mode(rock_prob, backup_data=True)
            cur_state = state.deep_copy()
            total_reward = 0
            for _ in range(steps):
                action = self.act(cur_state, history)
                observation, reward, done, cur_state = env.transotion_state(cur_state, action)
                self.update(cur_state, reward, action, observation, env.history)
                history_data = self.get_history_data(cur_state, env.history)
                simulation_history.add_step(cur_state, action=action, observation=observation, reward=reward, **history_data)
                total_reward += reward
            histories.append(simulation_history)
            self.data_api.create_run_history_table(hist_table_name)
            self.data_api.write_history(simulation_history, schema= hist_table_name)
            sum_rewards.append(total_reward)
            self.exit_inner_simulation_mode(restore_data = True)
        return sum_rewards, histories

    def update_agents_rock_probs_on_send_data(self, rock_loc):
        # update agent's probs
        self.rock_probs[rock_loc] = self.real_rock_probs[rock_loc]
        # update self probs
        for possible_belief in self.agent_beliefs_elements:
            possible_belief[rock_loc] = self.real_rock_probs[rock_loc]

    def update_agents_rock_probs_on_agent_sample(self, rock_sample_loc: tuple[int, int], state):
        new_agent_beliefs_dist = list()
        new_agent_beliefs_elements = list()
        for belief_prob, possible_belief in zip(self.agent_beliefs_dist, self.agent_beliefs_elements):
            bad_rock_prob, good_rock_prob = self.get_bu_rock_probs(rock_sample_loc, possible_belief[rock_sample_loc],
                                                                   SampleObservation.BAD_ROCK, state)
            if_agent_got_bad_rock = {SampleObservation.GOOD_ROCK: good_rock_prob,
                                     SampleObservation.BAD_ROCK: bad_rock_prob}
            bad_rock_prob, good_rock_prob = self.get_bu_rock_probs(rock_sample_loc, possible_belief[rock_sample_loc],
                                                                   SampleObservation.GOOD_ROCK, state)
            if_agent_got_good_rock = {SampleObservation.GOOD_ROCK: good_rock_prob,
                                      SampleObservation.BAD_ROCK: bad_rock_prob}

            bad_rock_belief, good_rock_belief = possible_belief.copy(), possible_belief.copy()
            bad_rock_belief[rock_sample_loc] = if_agent_got_bad_rock
            good_rock_belief[rock_sample_loc] = if_agent_got_good_rock
            new_agent_beliefs_elements.extend([bad_rock_belief, good_rock_belief])
            new_agent_beliefs_dist.extend([0.5 * belief_prob, 0.5 * belief_prob])

        self.agent_beliefs_elements = new_agent_beliefs_elements
        self.agent_beliefs_dist = new_agent_beliefs_dist

