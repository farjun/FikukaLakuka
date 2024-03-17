import keyboard

from Multi_Agent_Robot.multi_agent_robot.agent.base import Agent
from Multi_Agent_Robot.multi_agent_robot.env.agent_action_space import AgentActionSpace
from Multi_Agent_Robot.multi_agent_robot.env.history import History
from Multi_Agent_Robot.multi_agent_robot.env.types import Action, RobotActions
from config import config

# TODO Problem with running this agent on mac since keyboard requires root access

class ManualAgent(Agent):

    def __init__(self, config_params: dict):
        self.config_params = config_params
        rocks = config.get_in_game_context("environment", "rocks")
        self.action_space = AgentActionSpace(agent_type="robot", n_rocks=len(rocks))

    def act(self, state, history: History) -> Action:
        arrow_key = self.wait_for_arrow_key()
        print(f"You pressed: {arrow_key}")
        action = Action(action_type=RobotActions[arrow_key.upper()])
        return action

    def wait_for_arrow_key(self):
        print("Press an arrow key to continue...")
        while True:
            event = keyboard.read_event()
            if event.event_type == keyboard.KEY_DOWN and event.name in ['up', 'down', 'left', 'right']:
                return event.name
