from evaluate_policy import evaluate_policy
import gymnasium as gym
import numpy as np

default_config = {}

def train_sb3(config):
    import os, random
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from stable_baselines3 import SAC as SB3_SAC
    from stable_baselines3.sac import MlpPolicy
    import torch
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    env = gym.make(config["environment_name"])
    env.reset(seed=config["seed"])
    def policy_factory(obs_dim, action_dim, lr_schedule, **kwargs):
        return MlpPolicy(obs_dim, action_dim, lr_schedule, net_arch=[config["hidden_dim"], config["hidden_dim"]])
    model = SB3_SAC(policy_factory, env, learning_rate=1e-3, batch_size=100, buffer_size=config["n_steps"])
    returns = []
    for evaluation_step_i in range(config["n_steps"] // config["evaluation_interval"]):
        model.learn(total_timesteps=config["evaluation_interval"], reset_num_timesteps=False)
        def policy(observation):
            return model.predict(observation, deterministic=True)[0]
        returns.append(evaluate_policy(policy, config))
    return returns