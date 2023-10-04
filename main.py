import airline
import environment

import os
import ray
from ray import air, tune
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env import PettingZooEnv
from ray.tune.registry import register_env


def test_env():
    # print("Hello World")


    # To interact with your custom AEC environment, use the following code:

    # import aec_rps

    env = environment.env(airline.CONFIG, render_mode="human")
    env.reset(seed=42)

    # env = aec_rps.env(render_mode="human")
    # env.reset(seed=42)

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        else:
            # this is where you would insert your policy
            action = env.action_space(agent).sample()
            # print(action)
            # print(type(action))
        env.step(action)
    env.close()

def train():
    ray.init()

    cfg = airline.CONFIG
    def env_creator(cfg):
        # env = environment.env(cfg, render_mode="human")
        env = environment.env(cfg, render_mode=None)
        env.reset(seed=42)
        return PettingZooEnv(env)
    env_name = "airline"
    register_env(env_name, env_creator)
    
    env = env_creator(cfg)
    obs_space = env.observation_space
    act_space = env.action_space
    policies = {agent_id:(None, obs_space, act_space, {"gamma": 0.99}) for agent_id in env.get_agent_ids()}
    
    config = (
        PPOConfig()
        .environment(
            env=env_name, 
            env_config=cfg,
            disable_env_checking=True
            )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id),
            )
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
        .debugging(
            log_level="ERROR"
            # log_level="DEBUG"
        ) 
        .framework(framework="torch")
    )

    tune.run(
        "PPO",
        name="PPO",
        stop={"timesteps_total": 10000},
        checkpoint_freq=10,
        config=config.to_dict(),
        local_dir = os.getcwd()+"/ray_results/"+env_name,
    )

    
if __name__=="__main__":
    # test_env()
    # raise NotImplementedError()
    # https://github.com/ray-project/ray/blob/master/rllib/examples/multi_agent_independent_learning.py
    # https://github.com/ray-project/ray/blob/master/rllib/examples/custom_env.py
    # https://github.com/Farama-Foundation/PettingZoo/blob/master/tutorials/Ray/rllib_leduc_holdem.py
    train()