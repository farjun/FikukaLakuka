import gym
import numpy as np
from gym import spaces

from config import config
from fikuka_lakuka.models.i_state import IState


class RobotsEnv(gym.Env):

    def __init__(self):
        super()
        self.state = IState()

    def step(self, action):
        assert action in self.action_space

    def reset(self):
        self.state = IState()

    def render(self, mode='human', close=False):
        self.state.print()

