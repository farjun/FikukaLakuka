from pettingzoo import AECEnv
from gymnasium import spaces

from Multi_Agent_Robot.multi_agent_robot.agent import init_agent, BayesianBeliefAgent
from Multi_Agent_Robot.multi_agent_robot.env.agent_action_space import AgentActionSpace
import numpy as np

from Multi_Agent_Robot.multi_agent_robot.env.history import History
from Multi_Agent_Robot.multi_agent_robot.env.types import RockTile, CellType, SampleObservation, RobotActions, Action, OracleActions
from Multi_Agent_Robot.multi_agent_robot.ui.gui import RockGui
from config import config
from typing import Tuple, List, Dict


class MultiAgentRobotEnv(AECEnv):
    MAX_STEPS = 1000
    metadata = {
        "name": "multi_agent_robot_v0",
    }

    def __init__(self):
        super().__init__()
        # Get all current game configurations
        self.history = History()
        self.agents: List = [init_agent(agent_id) for agent_id in config.get_in_game_context("playing_agents")]
        self.grid_size: Tuple[int, int] = config.get_in_game_context("environment", "grid_size")
        self.rocks_arr: List[Tuple[int, int]] = [tuple(x) for x in config.get_in_game_context("environment", "rocks")]
        self.rocks_reward_arr: List[int] = config.get_in_game_context("environment", "rocks_reward")
        self.start_pt: List[int] = config.get_in_game_context("environment", "start")
        self.end_pt: List[int] = config.get_in_game_context("environment", "end")
        self.sample_prob: float = config.get_in_game_context("environment", "sample_prob")
        self.gas_fee: float = config.get_in_game_context("environment", "gas_fee")
        self.agent_selection: int = config.get_in_game_context("environment", "starting_agent")

        # Derive constants from the configurations
        self.agent_types: List[str] = ["oracle" if agent == "oracle" else "robot" for agent in self.agents]
        self.n_rocks: int = len(self.rocks_arr)
        # Create a dictionary of rocks and their rewards and whether they have been collected or not
        self.rock_dict: Dict[Tuple[int, int], RockTile] = {tuple(loc): RockTile(loc=loc, reward=reward) for loc, reward in
                                                           zip(self.rocks_arr, self.rocks_reward_arr)}

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
        self._agent_locations = [self.start_pt.copy() for _ in range(self.num_agents)]

        # Set board values for start, end and rocks
        self._board[self.start_pt[0], self.start_pt[1]] = CellType.START.value
        self._board[self.end_pt[0], self.end_pt[1]] = CellType.END.value
        for rock in self.rocks_arr:
            self._board[rock[0], rock[1]] = CellType.ROCK.value

        # Set the current state
        self.state = {
            "board": self._board,
            "grid_size": self.grid_size,
            "sample_prob": self.sample_prob,
            "agents": self.agents,
            "agent_locations": self._agent_locations,
            "agent_selection": self.agent_selection,
            "current_agent_location": self._agent_locations[self.agent_selection],
            "rocks_reward": self.rocks_reward_arr,
            "rocks_dict": self.rock_dict,
            "collected_rocks": self.collected_rocks,
            "gas_fee": self.gas_fee,
            "start_pt": self.start_pt,
            "end_pt": self.end_pt,
        }
        # Set the GUI
        self.gui = RockGui(self.state)

    @property
    def collected_rocks(self):
        return [rock.picked for rock in self.rock_dict.values()]

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
        self._board = np.zeros(self.grid_size, dtype=int)
        self._agent_locations = [self.start_pt.copy() for _ in range(self.num_agents)]

        # Set board values for start, end and rocks
        self._board[self.start_pt[0], self.start_pt[1]] = CellType.START.value
        self._board[self.end_pt[0], self.end_pt[1]] = CellType.END.value

        # Reset the rocks
        for rock in self.rocks_arr:
            self.rock_dict[rock].picked = False
            self._board[rock[0], rock[1]] = CellType.ROCK.value

        # Reset the current state
        self.update_state()
        # Set the GUI
        self.gui = RockGui(self.state)

    def step(self, action: Action = None):
        """
        Used to step the environment forward by one step. It takes an action for the current agent and should be used within a loop to
        where we loop through all the agents in the environment (Using AECIter).
        """
        agent_type = self.agent_types[self.agent_selection]
        if agent_type == "oracle":
            return self.oracle_step(self.agent_selection)
        elif agent_type == "robot":
            return self.robot_step(self.agent_selection)
        raise ValueError("Invalid agent type")

    def robot_step(self, agent_id: int):
        agent = self.agents[agent_id]
        action = agent.act(self.state, self.history)
        observation, reward, done, robot_observation = None, 0, False, SampleObservation.NO_OBS

        if action.action_type == RobotActions.SAMPLE:
            robot_observation = self.sample_rock(agent_id, action.rock_sample_loc)
        else:  # Action is a movement action
            reward -= self.gas_fee
            # Update location
            agent_pos = self._agent_locations[agent_id]
            board_x, board_y = self._board.shape
            new_agent_pos = self.move_robot(action, agent_pos, board_x, board_y)
            self._board[agent_pos[0], agent_pos[1]] = CellType.EMPTY.value
            self._board[new_agent_pos[0], new_agent_pos[1]] = CellType.ROBOT1.value
            self._agent_locations[agent_id] = new_agent_pos
            reward += self.remove_rock(agent_pos, tuple(new_agent_pos))

        if self._agent_locations[agent_id] == self.end_pt:
            done = True
            reward += 10
        # Update belief vector with respect to each agent
        agent_beliefs = agent.update(self.state, reward, action, observation, self.history)
        self.history.update(agent_id, action, robot_observation, reward, self._agent_locations, agent_beliefs.copy())

        observation = self.state["board"], agent_beliefs
        self.update_state()
        truncated = False
        return observation, reward, done, truncated, self.state

    def remove_rock(self, agent_pos, new_agent_pos):
        reward = 0
        if new_agent_pos in self.rock_dict.keys() and not self.rock_dict[new_agent_pos].picked:
            rock = self.rock_dict[new_agent_pos]
            rock.picked = True
            reward = rock.reward
            self._board[agent_pos[0], agent_pos[1]] = CellType.EMPTY.value
        return reward

    def oracle_step(self, agent_id: int):
        information_cost = config.get_in_game_context("environment", "information_cost")
        agent = self.agents[agent_id]
        action = agent.act(self.state, self.history)
        observation, reward, done = None, 0, False
        # TODO maybe if we want we can add a parameter to the oracle to specify which robot to update
        if action.action_type == OracleActions.SEND_GOOD_ROCK:
            reward -= information_cost
            # Update robot beliefs
            for robot in self.agents:
                if isinstance(robot, BayesianBeliefAgent):
                    robot.update_beliefs(action.rock_sample_loc, True)
        elif action.action_type == OracleActions.SEND_BAD_ROCK:
            reward -= information_cost
            # Update robot beliefs
            for robot in self.agents:
                if isinstance(robot, BayesianBeliefAgent):
                    robot.update_beliefs(action.rock_sample_loc, False)
        elif action.action_type == OracleActions.DONT_SEND_DATA:
            pass

        agent_beliefs = agent.update(self.state, reward, action, observation, self.history)
        # TODO maybe define this as somthing more meaningful
        oracle_observation = SampleObservation.NO_OBS
        self.history.update(agent_id, action, oracle_observation, reward, self._agent_locations, agent_beliefs.copy())

        observation = self.state["board"], agent_beliefs
        self.update_state()
        truncated = False
        return observation, reward, done, truncated, self.state

    @staticmethod
    def move_robot(action, agent_pos, board_x, board_y):
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
        agent_location = self._agent_locations[agent]
        dist = np.linalg.norm(np.array(agent_location) - np.array(rock_loc), ord=1)
        p = self.calc_sample_prob(dist)
        rock = self.rock_dict[rock_loc]
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

    def update_state(self):
        self.state["board"] = self._board
        self.state["agent_locations"] = self._agent_locations
        self.state["agent_selection"] = self.agent_selection
        self.state["current_agent_location"] = self._agent_locations[self.agent_selection]
        self.state["rocks_dict"] = self.rock_dict
        self.state["collected_rocks"] = self.collected_rocks
