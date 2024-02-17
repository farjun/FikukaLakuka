from itertools import cycle

from pynput import keyboard
from pynput.keyboard import Key

from fikuka_lakuka.fikuka_lakuka.models import History, ActionSpace
from fikuka_lakuka.fikuka_lakuka.models.action_space import Action, Actions
from fikuka_lakuka.fikuka_lakuka.models.agent.base import Agent
from fikuka_lakuka.fikuka_lakuka.models.i_state import IState


class ConstAgent(Agent):

    def __init__(self, config_params: dict):
        self.config_params = config_params
        self.action_space = ActionSpace()
        self.actions_iter = iter([Actions.DOWN] * 10)


    def act(self, state: IState, history: History) -> Action:
        try:
            return Action(action_type=next(self.actions_iter))
        except StopIteration:
            return Action(action_type=Actions.SAMPLE, rock_sample_loc = state.rocks[0].loc)

