import gym

from config import config
from fikuka_lakuka.fikuka_lakuka.models.i_state import IState
from fikuka_lakuka.fikuka_lakuka.models.action_space import ActionSpace
from fikuka_lakuka.fikuka_lakuka.models.history import History
from fikuka_lakuka.fikuka_lakuka.models.agent import init_agent
from fikuka_lakuka.fikuka_lakuka.models.agent.base import Agent
from fikuka_lakuka.fikuka_lakuka.models.observation_space import ObservationSpace
from fikuka_lakuka.fikuka_lakuka.ui.robots import RobotsUI


class RobotsEnv_v0(gym.Env):
    MAX_STEPS = 1000

    def __init__(self):
        super()
        self.state = IState()
        self.action = ActionSpace()
        self._observation_space = ObservationSpace()
        self.history = History()
        self._ui_renderer = None

        self.observation_space = self._observation_space.space
        self.agents = [init_agent(agent_id) for agent_id in config.get_in_game_context("playing_agents")]
        self.action_space = self.action

    @property
    def cur_agent(self) -> Agent:
        return self.agents[self.state.cur_agent_idx]

    @property
    def ui_renderer(self):
        if not self._ui_renderer:
            self._ui_renderer = RobotsUI()
        return self._ui_renderer

    def step(self, action):
        assert action in self.action_space
        action = self.cur_agent.act(self.state, self.history)
        self.history.add_action(action)
        reward, done = self.state.update(self.state.cur_agent_idx, action)
        return self.state.board, reward, done, False, {"info": "some info"}

    def reset(self, **kwargs):
        self.state = IState()
        return self.state.board,  {"info": "some info"}

    def sample(self):
        action = self.cur_agent.act(self.history)
        return action.space

    def render(self, mode='not', close=False):
        if mode=="human":
            self.ui_renderer.show(self.state.board)
        else:
            self.state.print()
