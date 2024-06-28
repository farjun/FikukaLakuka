#!/usr/bin/env python
from Multi_Agent_Robot.multi_agent_robot.agent import init_agent
from Multi_Agent_Robot.multi_agent_robot.env.multi_agent_robot import MultiAgentRobotEnv, run_one_episode
from config import config


def main():
    # first, create the custom environment and run it for one episode
    agents = [init_agent(agent_id) for agent_id in config.get_in_game_context("playing_agents")]
    env = MultiAgentRobotEnv(agents)

    # next, calculate a baseline of rewards based on random actions
    # (no policy)
    history = []

    sum_reward = run_one_episode(env, verbose=True, use_sleep=False, force_recreate_tables=True)
    history.append(sum_reward)

    avg_sum_reward = sum(history) / len(history)
    print("\nbaseline cumulative reward: {:6.2}".format(avg_sum_reward))
    # graph_baysian_agents_beliefs()


if __name__ == "__main__":
    main()
    # run_state_clustering()
