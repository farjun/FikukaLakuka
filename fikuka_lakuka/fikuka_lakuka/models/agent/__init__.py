from config import config
from fikuka_lakuka.fikuka_lakuka.models.agent.random import RandomAgent


def init_agent(agent_id :str):
    return {
        "random": RandomAgent
    }[agent_id](config.get_in_agent_context(agent_id))