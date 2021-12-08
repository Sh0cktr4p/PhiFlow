from reinforcement_learning.envs.skip_stack_wrapper import BrenerStacker
from stable_baselines3.common.envs.identity_env import IdentityEnv
from gym import Env
from gym.spaces import Box
import numpy as np

class TestEnv(Env):
    def __init__(self, dim: int, ep_len: int=10):
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(dim,))
        self.action_space = Box(low=-np.inf, high=np.inf, shape=(1,))
        assert ep_len > 0
        self.ep_len = ep_len
        self.step_idx = 0
    
    def reset(self):
        self.step_idx = 0
        return np.zeros(self.observation_space.shape)

    def step(self, action: np.ndarray):
        assert isinstance(action, np.ndarray)
        assert action.shape == self.action_space.shape
        done = self.step_idx == self.ep_len
        self.step_idx += 1
        return np.full(self.observation_space.shape, -action[0]), 0, done, {}

if __name__ == '__main__':
    env = TestEnv(10)
    env = BrenerStacker(env, 2, 3, 2, True)

    obs = env.reset()
    print('Obs: ', obs)
    print('')
    done = False
    act = np.array([1])
    print(env.action_space.shape)
    while not done:
        print('Act: ', act)
        obs, rew, done, _ = env.step(act)
        act += 1
        print('Obs: ', obs)
        print('')
    
    print('done')