from collections import deque
from typing import Any, Dict, Tuple

import numpy as np
from gym import Env, Wrapper
from gym.spaces import Box


class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]


class FrameStack(Wrapper):
    def __init__(self, env: Env, k: int):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(shp[:-1] + (shp[-1] * k,)), dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action: np.ndarray):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class BrenerStacker(Wrapper):
    def __init__(self, env: Env, n_present_features: int, n_past_features: int, past_window: int, append_past_actions: bool):
        super().__init__(env)
        self.frames = deque([], maxlen=past_window + 1)
        self.past_actions = deque([], maxlen=past_window)

        self.n_present_features = n_present_features
        self.n_past_obs_features = n_past_features
        self.past_window = past_window
        self.append_past_actions = append_past_actions
        
        self.n_past_features = self.n_past_obs_features
        self.n_action_features = np.prod(env.action_space.shape)
        if self.append_past_actions:
            self.n_past_features += self.n_action_features

        assert isinstance(env.observation_space, Box)
        obs_dim = self.n_present_features + self.n_past_features * self.past_window
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,))

    def reset(self) -> np.ndarray:
        obs: np.ndarray = self.env.reset()
        for _ in range(self.past_window):
            self.frames.append(np.zeros(obs.shape))
            self.past_actions.append(np.zeros((self.n_action_features)))
        self.frames.append(obs.copy())
        return self._get_obs()

    def step(self, act: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, Dict[str, Any]]:
        obs, rew, done, info = self.env.step(act)
        self.frames.append(obs.copy())
        self.past_actions.append(act.copy())
        return self._get_obs(), rew, done, info

    def _get_obs(self):
        assert len(self.frames) == self.past_window + 1
        assert len(self.past_actions) == self.past_window

        def collect_obs_features(frame_index: int, feature_count) -> np.ndarray:
            return self.frames[frame_index].reshape(-1)[:feature_count]

        def collect_past_features(frame_index: int) -> np.ndarray:
            obs_features = collect_obs_features(frame_index, self.n_past_obs_features)
            act_features = self.past_actions[frame_index]
            if self.append_past_actions:
                return np.concatenate([obs_features, act_features])
            return obs_features

        return np.concatenate([
            *[collect_past_features(i) for i in range(self.past_window)],
            collect_obs_features(-1, self.n_present_features),
        ])


class MaxAndSkipEnv(Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        acc_info = {}
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            acc_info.update(info)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        frames = np.stack(self._obs_buffer)
        #max_frame = np.max(frames, axis=0)
        min_val = frames.min(axis=0)
        max_val = frames.max(axis=0)
        abs_max_frame = np.where(-min_val > max_val, min_val, max_val)

        return abs_max_frame, total_reward, done, acc_info

    def reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class SkipStackWrapper(Wrapper):
    def __init__(self, env: Env, skip: int=4, stack: int=4):
        env = MaxAndSkipEnv(env, skip=skip)
        env = FrameStack(env, k=stack)

        super().__init__(env)
