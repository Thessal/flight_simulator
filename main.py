import airline
import environment

import os
import ray
from ray import air, tune
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ddpg import DDPGConfig
from ray.rllib.algorithms.dqn import DQNConfig
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

    # https://github.com/ray-project/ray/blob/master/rllib/examples/multi_agent_independent_learning.py
    # https://github.com/ray-project/ray/blob/master/rllib/examples/custom_env.py
    # https://github.com/Farama-Foundation/PettingZoo/blob/master/tutorials/Ray/rllib_leduc_holdem.py


    ray.init()

    cfg = airline.CONFIG
    cfg["num_iters"] = 1000
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
    policies = {agent_id:(None, obs_space, act_space, {}) for agent_id in env.get_agent_ids()}
    
    config = (
        DQNConfig()
        # PPOConfig()
        .environment(
            env=env_name, 
            env_config=cfg,
            # disable_env_checking=True
            )
        # .training(gamma=0.9, lr=0.01, kl_coeff=0.3)
        # .rollouts(num_rollout_workers=4) 
        .rollouts(num_rollout_workers=2, rollout_fragment_length=len(policies)*2)
        # .rollouts(num_rollout_workers=1, rollout_fragment_length=5)
        # .rollouts(num_rollout_workers=2, rollout_fragment_length='auto')
        # .training(train_batch_size=2, gamma=0.5, lr=0.1,)
        # .training(train_batch_size=2, gamma=0.5, lr=0.1, kl_coeff=0.3   )
        # .rollouts(num_rollout_workers=1, rollout_fragment_length=30)
        # .training(train_batch_size=200, gamma=0.9, lr=0.01, kl_coeff=0.3)
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
        # .exploration(
        #     exploration_config={
        #         # The Exploration class to use.
        #         "type": "EpsilonGreedy",
        #         # Config for the Exploration class' constructor:
        #         "initial_epsilon": 0.1,
        #         "final_epsilon": 0.0,
        #         "epsilon_timesteps": 100000,  # Timesteps over which to anneal epsilon.
        #     }
        # )
    )
    # print(config.to_dict())
    # raise NotImplementedError()

    tune.run(
        # "PPO",
        # name="PPO",
        "DQN",
        name="DQN",
        stop={"timesteps_total": cfg["num_iters"]*10},
        checkpoint_freq=10,
        config=config.to_dict(),
        local_dir = os.getcwd()+"/ray_results/"+env_name,
    )

    
if __name__=="__main__":
    # test_env()
    # raise NotImplementedError()
    train()

