from Multi_Agent_Robot.multi_agent_robot.agent.algo import AlgoAgent
from Multi_Agent_Robot.multi_agent_robot.agent.baysian_update import BayesianBeliefAgent
from Multi_Agent_Robot.multi_agent_robot.agent.const import ConstAgent
from Multi_Agent_Robot.multi_agent_robot.agent.manual import ManualAgent
from Multi_Agent_Robot.multi_agent_robot.agent.random import RandomAgent
from config import config


def init_agent(agent_id: str):
    return {
        "random": RandomAgent,
        "manual": ManualAgent,
        "const": ConstAgent,
        "algo": AlgoAgent,
        "bbu": BayesianBeliefAgent,

    }[agent_id](config.get_in_agent_context(agent_id))
