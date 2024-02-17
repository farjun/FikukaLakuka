from pynput import keyboard
from pynput.keyboard import Key

from fikuka_lakuka.fikuka_lakuka.models import History, ActionSpace
from fikuka_lakuka.fikuka_lakuka.models.action_space import Action, Actions
from fikuka_lakuka.fikuka_lakuka.models.agent.base import Agent
from fikuka_lakuka.fikuka_lakuka.models.i_state import IState
import curses
import os
os.environ['TERM'] = 'xterm'
import keyboard


class ManualAgent(Agent):

    def __init__(self, config_params: dict):
        self.config_params = config_params
        self.action_space = ActionSpace()


    def act(self, state: IState, history: History) -> Action:
        arrow_key = self.wait_for_arrow_key()
        print(f"You pressed: {arrow_key}")
        return Actions.DOWN

    def wait_for_arrow_key(self):
        print("Press an arrow key to continue...")
        while True:
            event = keyboard.read_event()
            if event.event_type == keyboard.KEY_DOWN and event.name in ['up', 'down', 'left', 'right']:
                return event.name


