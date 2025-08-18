from Multi_Agent_Robot.multi_agent_robot.agent.algo import AlgoAgent
from Multi_Agent_Robot.multi_agent_robot.agent.baysian_update import BayesianBeliefAgent
from Multi_Agent_Robot.multi_agent_robot.agent.const import ConstAgent
from Multi_Agent_Robot.multi_agent_robot.agent.random import RandomAgent
from Multi_Agent_Robot.multi_agent_robot.agent.pomcp import POMCPAgent
from config import config


def init_agent(agent_id: str):
    model_name = config.get_in_agent_context(agent_id, "model_name")
    return {
        "random": RandomAgent,
        "const": ConstAgent,
        "algo": AlgoAgent,
        "bbu": BayesianBeliefAgent,
        "pomcp": POMCPAgent,

    }[model_name](config.get_in_agent_context(agent_id))
