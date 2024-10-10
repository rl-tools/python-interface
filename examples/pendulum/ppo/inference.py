import gymnasium as gym
from gymnasium.wrappers import RescaleAction
from rltools import load_checkpoint_from_path
import math

policy = load_checkpoint_from_path("pendulum_ppo_checkpoint.h", force_recompile=True)

env = gym.make("Pendulum-v1", render_mode="human")
env = RescaleAction(env, -1, 1) # wlog actions are normalized to [-1, 1] in RLtools


while True:
    observation, _ = env.reset()
    finished = False
    while not finished:
        env.render()
        action = policy.evaluate(observation)
        observation, reward, terminated, truncated, _ = env.step(action)
        finished = terminated or truncated