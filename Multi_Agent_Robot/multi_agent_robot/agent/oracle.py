from collections import Counter, defaultdict
from typing import List, Tuple, Dict

import numpy as np
from scipy.stats import rankdata

from Multi_Agent_Robot.multi_agent_robot.agent.base import Agent
from Multi_Agent_Robot.multi_agent_robot.data.api import DataApi
from Multi_Agent_Robot.multi_agent_robot.env.agent_action_space import AgentActionSpace
from Multi_Agent_Robot.multi_agent_robot.env.history import History
from Multi_Agent_Robot.multi_agent_robot.env.multi_agent_robot import MultiAgentRobotEnv, State
from Multi_Agent_Robot.multi_agent_robot.env.types import SampleObservation, Action, RobotActions, OracleActions
from config import config
from tqdm.auto import tqdm as auto_tqdm


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

    INTERVEEN_THRESHOLD = 7
    MAX_ALLOWED_AGENT_DISTS = 8
    HELPS_PER_BELIEF = 2

    def __init__(self, config_params: dict):
        super().__init__()
        self.config_params = config_params
        self.oracle_max_simulation_depth = config_params.get("oracle_max_simulation_depth", 4)
        self._data_api = None
        self.gas_fee = config.get_in_game_context("environment", "gas_fee")
        self.in_a_simulation = False
        rocks = config.get_in_game_context("environment", "rocks")
        self.enable_oracle = config_params.get("enable_oracle", True)
        rocks_reward = config.get_in_game_context("environment", "rocks_reward")
        self.real_rock_probs = dict((tuple(rock_loc), {SampleObservation.GOOD_ROCK: 1 if reward > 0 else 0,
                                                       SampleObservation.BAD_ROCK: 1 if reward <= 0 else 0}) for
                                    rock_loc, reward in zip(rocks, rocks_reward))

        self.allow_real_rock_probs = self.config_params.get("allow_real_rock_probs")
        if self.allow_real_rock_probs:
            self.rock_probs = self.real_rock_probs.copy()
            self.agent_beliefs_elements = [self.real_rock_probs.copy()]
        else:
            self.rock_probs = dict(
                (tuple(x), {SampleObservation.GOOD_ROCK: 0.5, SampleObservation.BAD_ROCK: 0.5}) for x in rocks)
            self.agent_beliefs_elements = [
                dict((tuple(x), {SampleObservation.GOOD_ROCK: 0.5, SampleObservation.BAD_ROCK: 0.5}) for x in rocks)]


        self.prev_agent_beliefs_elements = None
        self.agent_beliefs_dist = [1]

        self.sample_count = dict((tuple(x), 0) for x in rocks)

        # for history tracking
        self.history_information = dict()
        self._last_action_generated_q_values = []
        self.agent_beliefs_scores = []
        self.simulation_information = []

    def add_to_history(self, k, v):
        self.history_information[k] = v


    @property
    def data_api(self):
        if self._data_api is None:
            self._data_api = DataApi()
        return self._data_api


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
        if self.in_a_simulation:
            if last_action.action_type in [RobotActions.BUY_INFORMATION]:
                self.update_agents_rock_probs_on_send_data(last_action.rock_sample_loc)
                action_type = OracleActions.SEND_GOOD_ROCK if state.rocks_map[last_action.rock_sample_loc].reward > 0 else OracleActions.SEND_BAD_ROCK
                return Action(action_type=action_type, rock_sample_loc=last_action.rock_sample_loc)
            else:
                return Action(action_type=OracleActions.DONT_SEND_DATA)

        if not self.enable_oracle or all(state.collected_rocks()) or state.current_agent_location() == state.end_pt:
            return Action(action_type=OracleActions.DONT_SEND_DATA)

        if last_action.action_type in [RobotActions.SAMPLE]:
            self.update_agents_rock_probs_on_agent_sample(last_action.rock_sample_loc, state, last_action, history)

        no_interveen_rock_probs, interveen_rock_probs, changed_rock_locs = self.generate_possible_rock_probs(state, intervenes_per_belief=self.HELPS_PER_BELIEF)
        self.simulation_information = no_interveen_rock_probs
        agent_actions = []
        for i in range(2):
            no_interveen_sum_rewards, no_interveen_histories = self.simulate_agent_run(state,
                                                                                       no_interveen_rock_probs.copy(),
                                                                                       history,
                                                                                       f"oracle_simulation_no_help_step_{state.cur_step}",
                                                                                       unchanged_rock_beliefs=no_interveen_rock_probs)

            for hist in no_interveen_histories:
                simulation_q_values = hist.get_q_values()
                agent_actions.append([history_step.action for history_step in hist.past])

        # Step 1: Accumulate weighted info requests
        info_request_weights = defaultdict(float)

        for belief_index, belief_score in enumerate(self.agent_beliefs_scores):
            belief_actions = agent_actions[belief_index]

            for action in belief_actions:
                if action.action_type == RobotActions.BUY_INFORMATION:
                    # Weighted vote: add belief score to that rock's "request count"
                    info_request_weights[action.rock_sample_loc] += belief_score
                    break  # Only consider the first BUY_INFORMATION per belief (optional choice)

        # Step 2: Choose whether to send info and which rock to send
        if not info_request_weights:
            print("Oracle action: DONT_SEND_DATA ")
            return Action(action_type=OracleActions.DONT_SEND_DATA)
        else: # Choose the rock with the highest total weighted score
            rock_to_send_loc = max(info_request_weights, key=info_request_weights.get)
            rock_to_send = state.rocks_map[rock_to_send_loc]
            print("Oracle action: Sending info on rock {rock_to_send.rock_loc} ")
            return Action(action_type=OracleActions.SEND_GOOD_ROCK, rock_sample_loc=rock_to_send.rock_loc)



    def generate_possible_rock_probs(self, state, intervenes_per_belief = 1) -> Tuple[List[dict], List[dict], List]:
        interveen_rock_probs = []
        no_interveen_rock_probs = []
        rocks_locs = []
        rock_distances = self.get_rock_distances(state)
        dist_rocks = sorted(list(zip(rock_distances, state.rocks)), key=lambda x: x[0])

        for chosen_belief in self.agent_beliefs_elements[:OracleAgent.MAX_ALLOWED_AGENT_DISTS]:
            no_interveen_rock_probs.append(chosen_belief.copy())
            intervenes_added_to_belief = 0
            for rock_distance, rock in dist_rocks:
                if rock.picked:
                    continue
                if intervenes_added_to_belief >= intervenes_per_belief:
                    break
                intervenes_added_to_belief += 1

                interveened_agents_rock_probs = chosen_belief.copy()
                interveened_agents_rock_probs[rock.loc] = self.real_rock_probs[rock.loc]
                interveen_rock_probs.append(interveened_agents_rock_probs)
                rocks_locs.append(rock.loc)

        return no_interveen_rock_probs, interveen_rock_probs, rocks_locs

    def enter_inner_simulation_mode(self, beliefs: Dict[tuple, Dict[SampleObservation, float]], backup_data=False):
        self.in_a_simulation = True
        self._real_agents_beliefs = self.rock_probs
        self.rock_probs = {k:v.copy() for k,v in beliefs.items()}

    def exit_inner_simulation_mode(self, restore_data=False):
        self.in_a_simulation = False
        self.rock_probs = self._real_agents_beliefs

    def generate_q_values_of_beliefs(self, state: State, possible_beliefs: list, history: History, num_of_generations=1):
        all_q_values = list()
        for belief in possible_beliefs:
            self.enter_inner_simulation_mode(belief, backup_data=True)
            generated_q_values = list()
            actions = list()
            for i in range(num_of_generations):
                action = self.act(state, history)
                actions.append(action)
                generated_q_values.append(self.get_q_values())

            all_q_values.append(generated_q_values)
            self.exit_inner_simulation_mode(restore_data=True)

        return all_q_values

    def simulate_agent_run(self, state: State, rock_probs: list[dict], history: History, hist_table_name: str, changed_rock_locs = None, unchanged_rock_beliefs = None) -> Tuple[
        List[float], List[History]]:
        sum_rewards = list()
        histories = list()
        env = MultiAgentRobotEnv(state.agents)
        for i, rock_prob in auto_tqdm(enumerate(rock_probs), total=len(rock_probs), desc=f"Oracle is Simulating agent runs: step={state.cur_step}, writing to table {hist_table_name} for {self.oracle_max_simulation_depth} steps"):
            self.add_to_history("cur_simulated_changed_rock", changed_rock_locs[i] if changed_rock_locs is not None else None)
            self.add_to_history("unchanged_rock_beliefs", self.get_beliefs_as_db_repr(state, rock_prob) if unchanged_rock_beliefs is not None else None)
            simulation_history = History()
            self.enter_inner_simulation_mode(rock_prob, backup_data=True)
            cur_state = state.deep_copy()
            total_reward = 0
            for i in auto_tqdm(range(self.oracle_max_simulation_depth), total=self.oracle_max_simulation_depth,  position=0, leave=True):
                action = self.act(cur_state, history)
                observation, reward, done, cur_state = env.transotion_state(cur_state, action)
                self.update(cur_state, reward, action, observation, env.history)
                history_data = self.get_history_data(cur_state, env.history)
                simulation_history.add_step(cur_state, action=action, observation=observation, reward=reward, **history_data)
                total_reward += reward
                if done:
                    break

            histories.append(simulation_history)
            self.data_api.create_run_history_table(hist_table_name)
            self.data_api.write_history(simulation_history, schema= hist_table_name)
            sum_rewards.append(total_reward)
            self.exit_inner_simulation_mode(restore_data=True)

        return sum_rewards, histories

    def update_agents_rock_probs_on_send_data(self, rock_loc):
        # update agent's probs
        self.rock_probs[rock_loc] = self.real_rock_probs[rock_loc]
        # update self probs
        for possible_belief in self.agent_beliefs_elements:
            possible_belief[rock_loc] = self.real_rock_probs[rock_loc]


    def update_agents_rock_probs_on_agent_sample(self, rock_sample_loc: tuple[int, int], state, last_action, history):
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
            good_sample_prob, bad_sample_prob = state.calc_sample_probs(last_action.rock_sample_loc)
            new_agent_beliefs_dist.extend([belief_prob * bad_sample_prob, belief_prob * good_sample_prob])

        if len(history.past) >= 2:
            self._last_action_generated_q_values = self.generate_q_values_of_beliefs(history.states[-1],
                                                                               self.agent_beliefs_elements,
                                                                               history, num_of_generations=5)
            agent_beliefs_scores = self.get_q_values_scores_on_last_action(last_action, self._last_action_generated_q_values)
            normalized_agent_beliefs_scores = np.asarray(agent_beliefs_scores) / sum(agent_beliefs_scores)
            spredead_normalized_agent_beliefs_scores = np.repeat(normalized_agent_beliefs_scores, 2)
            new_agent_beliefs_dist = spredead_normalized_agent_beliefs_scores * new_agent_beliefs_dist
            new_agent_beliefs_dist /= np.sum(new_agent_beliefs_dist)


        agent_beliefs_ranked = sorted(zip(new_agent_beliefs_dist, new_agent_beliefs_elements), key=lambda x: x[0])
        self.prev_agent_beliefs_elements = self.agent_beliefs_elements
        self.agent_beliefs_scores,  self.agent_beliefs_elements = zip(*agent_beliefs_ranked)
        self.agent_beliefs_dist = new_agent_beliefs_dist
        self.agent_beliefs_scores, self.agent_beliefs_elements, self.agent_beliefs_dist = self.agent_beliefs_scores[:16], self.agent_beliefs_elements[:16], self.agent_beliefs_dist[:16]


    def get_q_values_scores_on_last_action(self, last_action, last_action_generated_q_values):
        actions = [a.action for a in last_action_generated_q_values[0][0]]
        last_action_index = actions.index(last_action)
        belief_scores = list()
        for belief_samples in last_action_generated_q_values:
            last_action_q_values = list()
            for sample in belief_samples:
                probs = [a.value for a in sample]
                ranks = rankdata(probs)
                last_action_q_values.append(ranks[last_action_index])
            belief_scores.append(sum(last_action_q_values) / len(last_action_q_values))
        return belief_scores

    def get_q_values(self):
        raise NotImplementedError

    def get_history_data(self, state:State, history:History)->dict:
        return {
            "oracle_beliefs": self.get_oracles_beliefs_as_db_repr(state),
            "oracle_beliefs_dist": [str(i) for i in self.agent_beliefs_dist],
            "oracle_last_action_generated_q_values" : self._last_action_generated_q_values,
            "oracle_simulation_beliefs": self.simulation_information,
            **self.history_information
        }

# change oracle beliefs to dist - implemented
# take prior into consideration - implemented
# run with and without oracle and compare preformence
# check if matching oracles buy information and rock loc will work - implemented
# add rock help loc to tags

