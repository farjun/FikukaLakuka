from time import sleep

from pettingzoo import AECEnv
from gymnasium import spaces
from pydantic import BaseModel

from Multi_Agent_Robot.multi_agent_robot.agent.base import Agent
from Multi_Agent_Robot.multi_agent_robot.data.api import DataApi
from Multi_Agent_Robot.multi_agent_robot.env.agent_action_space import AgentActionSpace
import numpy as np

from Multi_Agent_Robot.multi_agent_robot.env.history import History
from Multi_Agent_Robot.multi_agent_robot.env.types import RockTile, CellType, SampleObservation, RobotActions, Action, \
    OracleActions, State
from Multi_Agent_Robot.multi_agent_robot.ui.gui import RockGui
from config import config
from typing import Tuple, List, Dict


class MultiAgentRobotEnv(AECEnv):
    MAX_STEPS = 200
    metadata = {
        "name": "multi_agent_robot_v0",
    }

    def __init__(self, agents: List[Agent]):
        super().__init__()
        # Get all current game configurations
        self.history = History()
        self.agents: List = agents
        self.grid_size: Tuple[int, int] = config.get_in_game_context("environment", "grid_size")
        self.rocks_arr: List[Tuple[int, int]] = [tuple(x) for x in config.get_in_game_context("environment", "rocks")]
        self.rocks_reward_arr: List[int] = config.get_in_game_context("environment", "rocks_reward")
        self.start_pt: List[int] = config.get_in_game_context("environment", "start")
        self.end_pt: List[int] = config.get_in_game_context("environment", "end")
        self.sample_prob: float = config.get_in_game_context("environment", "sample_prob")
        self.gas_fee: float = config.get_in_game_context("environment", "gas_fee")
        self.sample_gas_fee: float = config.get_in_game_context("environment", "sample_gas_fee")

        agent_selection: int = config.get_in_game_context("environment", "starting_agent")

        # Derive constants from the configurations
        self.agent_types: List[str] = ["oracle" if agent == "oracle" else "robot" for agent in self.agents]
        self.n_rocks: int = len(self.rocks_arr)
        # Create a dictionary of rocks and their rewards and whether they have been collected or not
        self.rocks_arr = [RockTile(loc=loc, reward=reward) for loc, reward in
                          zip(self.rocks_arr, self.rocks_reward_arr)]
        self.rocks_map: Dict[Tuple[int, int], RockTile] = {tuple(rt.loc): rt for rt in self.rocks_arr}

        # Define the observation space as a dictionary of spaces for each agent, containing the board as seen by the agent and the agent's
        # belief vector on the rocks in the environment
        self.observation_spaces = self._convert_to_dict(
            [
                spaces.Dict(
                    {
                        "board_observation": spaces.Space(
                            shape=(self.grid_size[0], self.grid_size[1]), dtype=np.int8
                        ),
                        "belief_vec": spaces.Box(
                            low=0, high=1, shape=(self.n_rocks, 1), dtype=np.int8,
                        ),
                    }
                )
                for _ in range(self.num_agents)
            ]
        )
        # Define the action space as a dictionary of spaces for each agent, containing the action space for the agent
        self.action_spaces = self._convert_to_dict(
            [
                AgentActionSpace(agent_type, self.n_rocks) for agent_type in self.agent_types
            ]
        )

        # Define the board as a 2D array of zeros
        self._board = np.zeros(self.grid_size, dtype=int)
        agent_locations = [self.start_pt.copy() for _ in range(self.num_agents)]

        # Set board values for start, end and rocks
        self._board[self.start_pt[0], self.start_pt[1]] = CellType.START.value
        self._board[self.end_pt[0], self.end_pt[1]] = CellType.END.value
        for rock in self.rocks_arr:
            self._board[rock.loc[0], rock.loc[1]] = CellType.ROCK.value

        # Set the current state
        self.state = State(
            cur_step = 0,
            board=self._board,
            agent_selection=agent_selection,
            grid_size=self.grid_size,
            sample_prob=self.sample_prob,
            agents=self.agents,
            agent_locations=agent_locations,
            current_agent_location=agent_locations[agent_selection],
            rocks_reward=self.rocks_reward_arr,
            rocks=self.rocks_arr,
            rocks_map=self.rocks_map,
            collected_rocks=self.collected_rocks,
            gas_fee=self.gas_fee,
            start_pt=self.start_pt,
            end_pt=self.end_pt
        )
        # Set the GUI
        self._gui = None

    @property
    def collected_rocks(self):
        return [rock.picked for rock in self.rocks_map.values()]

    @property
    def agent_selection(self)->int:
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
        self.state.board = np.zeros(self.grid_size, dtype=int)
        self.state.agent_locations = [self.start_pt.copy() for _ in range(self.num_agents)]

        # Set board values for start, end and rocks
        self.state.board[self.start_pt[0], self.start_pt[1]] = CellType.START.value
        self.state.board[self.end_pt[0], self.end_pt[1]] = CellType.END.value

        # Reset the rocks
        for rock in self.rocks_arr:
            rock.picked = False
            self.state.board[rock.loc[0], rock.loc[1]] = CellType.ROCK.value

        # Set the GUI
        self._gui = None

    @property
    def gui(self):
        if self._gui is None:
            self._gui = RockGui(self.state)
        return self._gui

    def step(self, action: Action = None, skip_agent_update=False)->tuple:
        agent = self.agents[self.agent_selection]
        action = action or agent.act(self.state, self.history)
        observation, reward, done = SampleObservation.NO_OBS, 0, False

        if action.action_type == RobotActions.SAMPLE:
            observation = self.sample_rock(self.agent_selection, action.rock_sample_loc)
            reward -= self.sample_gas_fee

        else:  # Action is a movement action
            reward -= self.gas_fee
            # Update location
            agent_pos = self.state.agent_locations[self.agent_selection]
            board_x, board_y = self.state.board.shape
            new_agent_pos = self.move_robot(action, agent_pos, board_x, board_y)
            self.state.board[agent_pos[0], agent_pos[1]] = CellType.EMPTY.value
            self.state.board[new_agent_pos[0], new_agent_pos[1]] = CellType.ROBOT1.value
            self.state.agent_locations[self.agent_selection] = new_agent_pos
            reward += self.remove_rock(agent_pos, tuple(new_agent_pos))

        if self.state.agent_locations[self.state.agent_selection] == self.end_pt:
            done = True
            reward += 10

        # Update belief vector with respect to each agent
        if not skip_agent_update:
            agent_beliefs, oracles_beliefs, oracle_action = agent.update(self.state, reward, action, observation, self.history)
            self.history.update(
                cur_agent = self.agent_selection,
                action=action,
                observation=observation,
                reward=reward,
                players_pos=self.state.agent_locations,
                agent_beliefs=agent_beliefs,
                oracle_action=oracle_action,
                oracle_beliefs=oracles_beliefs,
                state=self.state
            )

        truncated = False
        self.state.cur_step +=1
        return observation, reward, done, truncated, self.state

    def remove_rock(self, agent_pos, new_agent_pos):
        reward = 0
        if new_agent_pos in self.state.rocks_map.keys() and not self.state.rocks_map[new_agent_pos].picked:
            rock = self.state.rocks_map[new_agent_pos]
            rock.picked = True
            reward = rock.reward
            self.state.board[agent_pos[0], agent_pos[1]] = CellType.EMPTY.value

        return reward

    @staticmethod
    def move_robot(action, agent_pos:tuple, board_x, board_y):
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

    def render(self, mode='not', close=False):
        if close:
            return
        if mode == "human":
            self.gui.render(self.state)
        else:
            print(self._board)

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def sample_rock(self, agent, rock_loc: Tuple[int, int]) -> SampleObservation:
        agent_location = self.state.agent_locations[agent]
        dist = np.linalg.norm(np.array(agent_location) - np.array(rock_loc), ord=1)
        p = self.calc_sample_prob(dist)
        rock = self.rocks_map[rock_loc]
        if rock.is_good():
            good_rock_prob, bad_rock_prob = p, 1 - p
        else:
            good_rock_prob, bad_rock_prob = 1 - p, p
        sample = np.random.choice([SampleObservation.BAD_ROCK.value, SampleObservation.GOOD_ROCK.value], 1,
                                  p=[bad_rock_prob, good_rock_prob])
        return SampleObservation(sample[0])

    def calc_sample_prob(self, distance_to_rock):
        distance_to_rock /= 3
        return 1 / 2 * (1 + np.exp(-distance_to_rock * np.log(2) / self.sample_prob))

    def _int_to_name(self, ind):
        return self.agents[ind]

    def _name_to_int(self, name):
        return self.agents.index(name)

    def _convert_to_dict(self, list_of_list):
        return dict(zip(self.agents, list_of_list))



def run_one_episode(env, verbose=False, use_sleep=False, force_recreate_tables=False, schema_name="env", skip_reset=False, max_steps=None):
    data_api = DataApi(force_recreate=force_recreate_tables, schema=schema_name)
    if not skip_reset:
        env.reset()

    total_reward = 0

    for i in range(max_steps or env.MAX_STEPS):
        done = False
        for _ in env.agent_iter():

            observation, reward, done, truncated, info = env.step()
            total_reward += reward
            if verbose:
                env.render(mode="human")

            if done:
                data_api.write_history(env.history)

                if verbose:
                    print("done @ step {}".format(i))

                break
            if use_sleep:
                sleep(0.05)
        if done:
            break

    if verbose:
        print("cumulative reward", total_reward)

    return total_reward
