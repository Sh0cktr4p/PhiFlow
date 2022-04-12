from gym import Wrapper
import numpy as np

class SkipEnv(Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip
        print("Using skip wrapper")

    def step(self, action):
        """Repeat action, average reward."""
        total_reward = np.array(0.0)
        obs = None
        done = None
        acc_info = {}
        steps_collected = 0
        for _ in range(self._skip):
            steps_collected += 1
            obs, reward, done, info = self.env.step(action)
            acc_info.update(info)
            total_reward += reward
            if done:
                break

        return obs, total_reward / steps_collected, done, acc_info

    def reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        obs = self.env.reset()
        return obs
