import gym
from ray.tune.registry import register_env

from fikuka_lakuka.agents import AgentFactory
from config import config
from ray.rllib.algorithms.ppo import PPO

def play_round(env, agents, state):
    agent_names = config.get_in_game_context("agents")
    agents_rewards = dict(zip(agent_names, [0]*len(agent_names)))

    for agent_name in agent_names:
        agent_class = agents[agent_name]
        action, _states = agent_class.predict(state)
        state, rewards, done, info = env.step(action)
        agents_rewards[agent_name] += rewards

        env.render()


        if done == 1:
            # report at the end of each episode
            print("cumulative reward", agents_rewards)
            state = env.reset()
            return state, rewards

def main():
    gym.envs.register(
        **config.get_in_game_context("gym_params")
    )
    env = gym.make(config.get_in_game_context("gym_params", "id"))
    agent_factory = AgentFactory()
    agents = agent_factory.make_agents(config.get_in_game_context("agents"), env=env, verbose=True)
    n_turns =10
    obs, info = env.reset(seed=config.get_in_game_context("seed"), return_info=True)

    for i in range(n_turns):
        play_round(env, agents, obs)


def run_atari_example():
    config = {
        # Environment (RLlib understands openAI gym registered strings).
        "env": "Taxi-v3",
        # Use 2 environment workers (aka "rollout workers") that parallelly
        # collect samples from their own environment clone(s).
        "num_workers": 2,
        # Change this to "framework: torch", if you are using PyTorch.
        # Also, use "framework: tf2" for tf2.x eager execution.
        "framework": "tf",
        # Tweak the default model provided automatically by RLlib,
        # given the environment's observation- and action spaces.
        "model": {
            "fcnet_hiddens": [64, 64],
            "fcnet_activation": "relu",
        },
        # Set up a separate evaluation worker set for the
        # `algo.evaluate()` call after training (see below).
        "evaluation_num_workers": 1,
        # Only for evaluation runs, render the env.
        "evaluation_config": {
            "render_env": True,
        },
    }

    # Create our RLlib Trainer.
    algo = PPO(config=config)

    # Run it for n training iterations. A training iteration includes
    # parallel sample collection by the environment workers as well as
    # loss calculation on the collected batch and a model update.
    for _ in range(3):
        print(algo.train())

    # Evaluate the trained Trainer (and render each timestep to the shell's
    # output).
    algo.evaluate()

if __name__ == '__main__':
    print("STARTED RUNNING")
    # main()
    run_atari_example()