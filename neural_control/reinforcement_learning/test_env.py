import numpy as np
from gym import Env
from reinforcement_learning.envs.two_way_coupling_env import TwoWayCouplingConfigEnv
from reinforcement_learning.envs.skip_stack_wrapper import SkipStackWrapper

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

    env = TwoWayCouplingConfigEnv(config_path)
    env = SkipStackWrapper(env, skip=8, stack=4)

    test_env(env)
