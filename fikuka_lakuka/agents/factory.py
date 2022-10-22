from typing import List

from stable_baselines3 import PPO

from config import config

STABLEINES_AGENTS = {
    "PPOtest": PPO
}


class AgentFactory:
    def __init__(self):
        self.stablines_agents = STABLEINES_AGENTS

    def make_stablines_agent(self, agent_name: str, env, **kwargs):
        stablines_agent_class = self.stablines_agents[agent_name]
        return stablines_agent_class(config.get_in_agent_context(agent_name, "name"), env, **kwargs)

    def make_agent(self, agent_name: str, env, **kwargs):
        if agent_name in STABLEINES_AGENTS:
            return self.make_stablines_agent(agent_name, env, **kwargs)

    def make_agents(self, agent_names: List[str], env, **kwargs) -> dict:
        agent_map = dict()
        for agent_name in agent_names:
            agent_map[agent_name] = self.make_agent(agent_name, env, **kwargs)
        return agent_map
