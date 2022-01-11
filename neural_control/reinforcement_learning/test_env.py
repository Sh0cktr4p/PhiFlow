import numpy as np
from gym import Env
from reinforcement_learning.envs.two_way_coupling_env import TwoWayCouplingConfigEnv
from reinforcement_learning.envs.skip_stack_wrapper import SkipStackWrapper
from reinforcement_learning.envs.test_torch_env import TwoWayCouplingConfigTorchEnv
from reinforcement_learning.envs.numpy_wrapper import NumpyWrapper
from reinforcement_learning.envs.seed_on_reset_wrapper import SeedOnResetWrapper

def test_env(env: Env):
    obs = env.reset()
    episode = 0

    while episode < 1:
        act = np.zeros(env.action_space.shape, dtype=np.float32)
        act[-1] = 0.5
        obs, rew, done, info = env.step(act)
        #print(obs[:])
        env.render('file')
        if done:
            obs = env.reset()
            print("simulated episode %i" % episode)
            episode += 1

if __name__ == '__main__':
    config_path = "/home/felix/Code/HiWi/Brener/PhiFlow/neural_control/inputs.json"

    env_a = TwoWayCouplingConfigEnv(config_path)
    env_a = SeedOnResetWrapper(env_a)

    env_b = TwoWayCouplingConfigTorchEnv(config_path)
    env_b = NumpyWrapper(env_b)
    env_b = SeedOnResetWrapper(env_b)

    obs_b = env_b.reset()
    obs_a = env_a.reset()

    act = np.ones((2,), dtype=np.float32)
    obs_a, rew_a, _, _ = env_a.step(act)
    obs_b, rew_b, _, _ = env_b.step(act)

    print(np.sum((obs_a - obs_b) ** 2))
    print(np.sum((rew_a - rew_b) ** 2))
    print(obs_a.shape)
    print(obs_b.shape)
