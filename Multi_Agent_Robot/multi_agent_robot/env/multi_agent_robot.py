from time import sleep
from typing import Tuple, List, Dict

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv

from Multi_Agent_Robot.multi_agent_robot.agent.base import Agent
from Multi_Agent_Robot.multi_agent_robot.data.api import DataApi
from Multi_Agent_Robot.multi_agent_robot.env.history import History
from Multi_Agent_Robot.multi_agent_robot.env.types import RockTile, SampleObservation, RobotActions, Action, \
    State
from Multi_Agent_Robot.multi_agent_robot.ui.gui import RockGui
from config import config


class MultiAgentRobotEnv(AECEnv):
    MAX_STEPS = 100
    metadata = {
        "name": "multi_agent_robot_v0",
    }
    REAL_ROCK_PROBS_MAP = None

    def __init__(self, agents: List[Agent]):
        super().__init__()
        # Get all current game configurations
        self.history = History()
        self.agents: list[Agent] = agents
        self.grid_size: Tuple[int, int] = config.get_in_game_context("environment", "grid_size")
        self.rocks_arr: List[Tuple[int, int]] = [tuple(x) for x in config.get_in_game_context("environment", "rocks")]
        self.rocks_reward_arr: List[int] = config.get_in_game_context("environment", "rocks_reward")
        self.start_pt: List[int] = config.get_in_game_context("environment", "start")
        self.end_pt: List[int] = config.get_in_game_context("environment", "end")
        self.sample_prob: float = config.get_in_game_context("environment", "sample_prob")
        # Derive constants from the configurations
        self.agent_types: List[str] = ["oracle" if agent == "oracle" else "robot" for agent in self.agents]
        # Create a dictionary of rocks and their rewards and whether they have been collected or not
        self.rocks_arr = [RockTile(loc=loc, reward=reward) for loc, reward in
                          zip(self.rocks_arr, self.rocks_reward_arr)]
        self.rocks_map: Dict[Tuple[int, int], RockTile] = {tuple(rt.loc): rt for rt in self.rocks_arr}
        MultiAgentRobotEnv.REAL_ROCK_PROBS_MAP = self.rocks_map


        # Define the observation space as a dictionary of spaces for each agent, containing the board as seen by the agent and the agent's

        # Define the board as a 2D array of zeros
        agent_locations = [self.start_pt.copy() for _ in range(self.num_agents)]

        # Set board values for start, end and rocks

        self.last_preformed_action = None
        # Set the current state
        self.state = State(
            cur_step=0,
            agent_selection=config.get_in_game_context("environment", "starting_agent"),
            grid_size=self.grid_size,
            sample_prob=self.sample_prob,
            agents=self.agents,
            agent_locations=agent_locations,
            rocks=self.rocks_arr,
            gas_fee=config.get_in_game_context("environment", "gas_fee"),
            sample_gas_fee=config.get_in_game_context("environment", "sample_gas_fee"),
            information_fee=config.get_in_game_context("environment", "information_fee"),
            start_pt=self.start_pt,
            end_pt=self.end_pt
        )
        self.history.add_step(self.state.deep_copy())

        # Set the GUI
        self._gui = None

    @property
    def agent_selection(self) -> int:
        return self.state.agent_selection

    def sample(self):
        agent = self.agents[self.agent_selection]
        action = agent.act(self.state, self.history)
        return action.space

    def reset(self, **kwargs):
        """
        Resets the environment to its initial state. Uses all the configurations to set the environment to its initial state.
        :return:
        """
        # Reset the board as a 2D array of zeros
        self.state.agent_locations = [self.start_pt.copy() for _ in range(self.num_agents)]

        # Reset the rocks
        for rock in self.rocks_arr:
            rock.picked = False

        # Set the GUI
        self._gui = None

    @property
    def gui(self):
        if self._gui is None:
            self._gui = RockGui(self.state)
        return self._gui

    @staticmethod
    def transotion_state(state: State, action: Action) -> tuple:
        observation, reward, done = SampleObservation.NO_OBS, 0, False
        if action.action_type == RobotActions.SAMPLE:
            observation = MultiAgentRobotEnv.sample_rock(state, action.rock_sample_loc)
            reward -= state.sample_gas_fee

        elif action.action_type == RobotActions.BUY_INFORMATION:
            observation = SampleObservation.GOOD_ROCK if MultiAgentRobotEnv.REAL_ROCK_PROBS_MAP[action.rock_sample_loc].reward > 0 else SampleObservation.BAD_ROCK
            reward -= state.information_fee

        elif action.action_type == RobotActions.COLLECT_ROCK:
            agent_pos = state.agent_locations[state.agent_selection]
            rock_reward, observation = MultiAgentRobotEnv.remove_rock(state, tuple(agent_pos))
            reward += rock_reward

        else:  # Action is a movement action
            reward -= state.gas_fee
            # Update location
            agent_pos = state.agent_locations[state.agent_selection]
            board_x, board_y = state.grid_size
            new_agent_pos = MultiAgentRobotEnv.move_robot(action, agent_pos, board_x, board_y)
            state.agent_locations[state.agent_selection] = new_agent_pos

        if state.agent_locations[state.agent_selection] == state.end_pt:
            done = True
            reward += 10

        state.agent_selection = (state.agent_selection + 1) % len(state.agents)
        state.cur_step += 1
        return observation, reward, done, state

    def step(self, action: Action) -> tuple:
        self.last_preformed_action = action
        observation, reward, done, self.state = self.transotion_state(self.state, action)
        truncated = False
        return observation, reward, done, truncated, self.state

    def run_one_turn(self):
        agent = self.agents[self.agent_selection]
        action = agent.act(self.state.deep_copy(), self.history)
        observation, reward, done, truncated, self.state = self.step(action)
        self.render(**agent.get_render_data())
        oracle_action = agent.update(self.state, reward, action, observation, self.history)
        history_data = agent.get_history_data(self.state, self.history)
        self.history.add_step(
            self.state.deep_copy(),
            action=action,
            observation=observation,
            reward=reward,
            oracle_action=oracle_action,
            **history_data,
        )
        return observation, reward, done, truncated, {}

    @staticmethod
    def remove_rock(state, rock_pos)->tuple[int, SampleObservation]:
        reward = 0
        observation = SampleObservation.NO_OBS
        if rock_pos in state.rocks_map.keys() and not state.rocks_map[rock_pos].picked:
            rock = state.rocks_map[rock_pos]
            rock.picked = True
            reward = rock.reward
            observation = SampleObservation.GOOD_ROCK if rock.is_good() else SampleObservation.BAD_ROCK

        return reward, observation

    @staticmethod
    def move_robot(action, agent_pos: tuple, board_x, board_y):
        agent_pos = list(agent_pos)
        if action.action_type == RobotActions.LEFT:
            agent_pos[1] = max([0, agent_pos[1] - 1])
        elif action.action_type == RobotActions.RIGHT:
            agent_pos[1] = min([board_x - 1, agent_pos[1] + 1])
        elif action.action_type == RobotActions.UP:
            agent_pos[0] = max([0, agent_pos[0] - 1])
        elif action.action_type == RobotActions.DOWN:
            agent_pos[0] = min([board_y - 1, agent_pos[0] + 1])
        return agent_pos

    def render(self, mode='not', close=False, **kwargs):
        if close:
            return
        msg = f"step={self.state.cur_step} {repr(self.last_preformed_action.ui_repr())}"
        if mode == "human":
            self.gui.render(self.state, msg=msg, **kwargs)
        else:
            print(msg)

    def observation_space(self, agent):
        return spaces.Dict(
            {
                "board_observation": spaces.Space(
                    shape=(self.grid_size[0], self.grid_size[1]), dtype=np.int8
                ),
                "belief_vec": spaces.Box(
                    low=0, high=1, shape=(len(self.rocks_arr), 1), dtype=np.int8,
                ),
            }
        )

    @staticmethod
    def sample_rock(state: State, rock_loc: Tuple[int, int]) -> SampleObservation:
        good_sample_prob, bad_sample_prob = state.calc_sample_probs(rock_loc)
        p = [good_sample_prob, bad_sample_prob] if state.rocks_map[rock_loc].is_good() else [bad_sample_prob, good_sample_prob]
        sample = np.random.choice([SampleObservation.GOOD_ROCK.value, SampleObservation.BAD_ROCK.value], 1, p=p)
        return SampleObservation(sample[0])



def run_one_episode(env, verbose=False, use_sleep=False, force_recreate_tables=False, schema_name="env",
                    skip_reset=False, max_steps=None):
    data_api = DataApi(force_recreate=force_recreate_tables)
    if not skip_reset:
        env.reset()

    total_reward = 0
    done = False
    for i in range(max_steps or env.MAX_STEPS):
        if done:
            break
        observation, reward, done, truncated, info = env.run_one_turn()
        total_reward += reward
        data_api.write_history_step(env.history.get_last_step_db_obj())
        env.render()
        if use_sleep:
            sleep(0.05)

        if done:
            break

    return total_reward
