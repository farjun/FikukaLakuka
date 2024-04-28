#!/usr/bin/env python
from time import sleep

from Multi_Agent_Robot.multi_agent_robot.data.api import DataApi
from Multi_Agent_Robot.multi_agent_robot.data.data_utils import graph_baysian_agents_beliefs
from Multi_Agent_Robot.multi_agent_robot.env.multi_agent_robot import MultiAgentRobotEnv


def run_one_episode(env, verbose=False):
    data_api = DataApi(force_recreate=True)
    env.reset()
    total_reward = 0

    for i in range(env.MAX_STEPS):
        done = False
        for _ in env.agent_iter():

            observation, reward, done, truncated, info = env.step()
            total_reward += reward
            if verbose:
                env.render(mode="human")

            if done:
                data_api.write_history(env.history)
                if verbose:
                    print("done @ step {}".format(i))

                break
            sleep(0.05)
        if done:
            break

    if verbose:
        print("cumulative reward", total_reward)

    return total_reward


def main():
    # first, create the custom environment and run it for one episode
    env = MultiAgentRobotEnv()

    # next, calculate a baseline of rewards based on random actions
    # (no policy)
    history = []

    sum_reward = run_one_episode(env, verbose=True)
    history.append(sum_reward)

    avg_sum_reward = sum(history) / len(history)
    print("\nbaseline cumulative reward: {:6.2}".format(avg_sum_reward))
    graph_baysian_agents_beliefs()


if __name__ == "__main__":
    main()
    # run_state_clustering()
