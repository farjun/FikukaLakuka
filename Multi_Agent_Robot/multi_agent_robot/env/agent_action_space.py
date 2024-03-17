from gym.spaces import Discrete


class AgentActionSpace(Discrete):
    """
    Custom action space for the agents in the environment.
    There is a different action space for each agent, depending on the type of agent.
    For Robots:
    - 4 directions (UP, DOWN, LEFT, RIGHT)
    - Sample each rock (n_rocks)
    For Oracle:
    - Do nothing (Don't interact with the environment)
    - Send information about a rock (n_rocks)
    """

    def __init__(self, agent_type, n_rocks):
        self.agent_type = agent_type
        if agent_type == "robot":
            self.num_of_actions = 4 + n_rocks  # 4 directions + sample each rock
        elif agent_type == "oracle":
            self.num_of_actions = 1 + n_rocks  # do nothing + send information about a rock
        super().__init__(self.num_of_actions)
