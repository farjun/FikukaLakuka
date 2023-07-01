#!/usr/bin/env python
# encoding: utf-8
from time import sleep

import gym
import fikuka_lakuka
from fikuka_lakuka.fikuka_lakuka.data.plotting import plot_3d_data
from fikuka_lakuka.fikuka_lakuka.data.data_utils import DataUtils
from fikuka_lakuka.fikuka_lakuka.data.api import DataApi
from fikuka_lakuka.fikuka_lakuka.gym_envs.robots.env import RobotsEnv_v0


def run_one_episode(env: RobotsEnv_v0, verbose=False):
    data_api = DataApi(force_recreate=True)
    env.reset()
    reward = 0

    for i in range(env.MAX_STEPS):
        action = env.action_space.sample()

        if verbose:
            print("action:", action)

        observation, reward, done, info = env.step(action)

        if verbose:
            env.render()

        if done:
            data_api.write_history(env.history)
            if verbose:
                print("done @ step {}".format(i))

            break
        sleep(0.2)


    if verbose:
        print("cumulative reward", reward)

    return reward


def main():
    # first, create the custom environment and run it for one episode
    env = gym.make("robots-v0")
    sum_reward = run_one_episode(env, verbose=True)

    # next, calculate a baseline of rewards based on random actions
    # (no policy)
    history = []

    sum_reward = run_one_episode(env, verbose=True)
    history.append(sum_reward)

    avg_sum_reward = sum(history) / len(history)
    print("\nbaseline cumulative reward: {:6.2}".format(avg_sum_reward))

def run_state_clustering():
    data_api = DataApi()
    du = DataUtils(data_api)
    clustered_states = du.cluster_states("bbu")
    plot_3d_data(clustered_states, "baysian update states")

if __name__ == "__main__":
    main()
    # run_state_clustering()#