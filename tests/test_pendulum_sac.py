from rltools import SAC
import gymnasium as gym
from gymnasium.wrappers import RescaleAction

def test_pendulum_sac():
    seed = 0xf00d
    def env_factory():
        env = gym.make("Pendulum-v1")
        env = RescaleAction(env, -1, 1) # wlog actions are normalized to [-1, 1] in RLtools
        
        env.reset(seed=seed)
        return env

    sac = SAC(env_factory, interface_name="test_pendulum_sac")
    state = sac.State(seed)

    finished = False
    while not finished:
        finished = state.step()