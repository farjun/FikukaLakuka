from gym.envs.registration import register
register(
    id="robots-v0",
    entry_point="fikuka_lakuka.fikuka_lakuka.gym_envs.robots.env:RobotsEnv_v0",
    max_episode_steps=300,
)
