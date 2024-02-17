from config import config
from fikuka_lakuka.fikuka_lakuka.models.agent.random import RandomAgent
from fikuka_lakuka.fikuka_lakuka.models.agent.algo import AlgoAgent
from fikuka_lakuka.fikuka_lakuka.models.agent.baysian_update import BaysianBeliefAgent
from fikuka_lakuka.fikuka_lakuka.models.agent.manual import ManualAgent
from fikuka_lakuka.fikuka_lakuka.models.agent.const import ConstAgent


def init_agent(agent_id :str):
    return {
        "random": RandomAgent,
        "manual": ManualAgent,
        "const": ConstAgent,
        "algo": AlgoAgent,
        "bbu": BaysianBeliefAgent,

    }[agent_id](config.get_in_agent_context(agent_id))