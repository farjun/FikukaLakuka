#!/usr/bin/env python
from Multi_Agent_Robot.multi_agent_robot.agent import init_agent
from Multi_Agent_Robot.multi_agent_robot.data.api import get_game_db_path
from Multi_Agent_Robot.multi_agent_robot.env.multi_agent_robot import MultiAgentRobotEnv, run_one_episode
from Multi_Agent_Robot.multi_agent_robot.env.types import State
from config import config

def reset_caches():
    State.get_all_possible_rock_beliefs.cache_clear()

def main():
    # first, create the custom environment and run it for one episode
    for game_name in config.games_to_run:
        config.set_game(game_name)
        for seed in config.get_in_game_context("seeds"):
            if not config.get("general", "override_games") and get_game_db_path(game_name, seed).exists():
                continue

            config.set_seed(seed)
            reset_caches()
            print("#"*10 + f"  Starting Game: {game_name}  " + "#"*10)
            agents = [init_agent(agent_id) for agent_id in config.get_in_game_context("playing_agents")]
            env = MultiAgentRobotEnv(agents)

            history = []
            sum_reward = run_one_episode(env, verbose=True, use_sleep=False, force_recreate_tables=True)
            history.append(sum_reward)

            avg_sum_reward = sum(history) / len(history)
            print("\nbaseline cumulative reward: {}".format(avg_sum_reward))


if __name__ == "__main__":
    main()
    # run_state_clustering()
