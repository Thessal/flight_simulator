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
import multiprocessing

def test_env():
    env = environment.env(airline.CONFIG.copy(), render_mode="human")
    env.reset(seed=42)

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

    cfg = airline.CONFIG.copy()
    sim_days = 30
    cfg["num_iters"] = sim_days * 24 * 60 // cfg["timestep"]
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
    
    n_workers = int(0.8*multiprocessing.cpu_count())
    config = (
        DQNConfig()
        .environment(
            env=env_name, 
            env_config=cfg,
            )
        # .training(gamma=0.9, lr=0.01)
        .training(train_batch_size=n_workers*len(policies))
        .rollouts(num_rollout_workers=n_workers, rollout_fragment_length=len(policies))
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
        "DQN",
        name="DQN",
        stop={"episodes_total": 12 * 10}, #10y
        checkpoint_freq=1000,
        checkpoint_at_end=True,
        config=config.to_dict(),
        local_dir = os.getcwd()+"/ray_results/"+env_name,
    )

    import bot
    bot.send_telegram("Finished training")

    
if __name__=="__main__":
    train()

