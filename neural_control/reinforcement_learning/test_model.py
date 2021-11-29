from gym import Env
from reinforcement_learning.envs.skip_stack_wrapper import SkipStackWrapper
from reinforcement_learning.envs.seed_on_reset_wrapper import SeedOnResetWrapper
from reinforcement_learning.envs.two_way_coupling_env import TwoWayCouplingConfigEnv
from stable_baselines3.sac import SAC
import os

def evaluate_model(model: SAC, env: Env):
    print(f"writing output to {env.sim_export_path}")

    episode = 0

    while episode < 20:
        step = 0
        cum_rew = 0
        done = False
        obs = env.reset()

        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, rew, done, _ = env.step(act)
            cum_rew += rew
            env.render('file')
            step += 1
        
        print("simulated episode %i. Cumulative reward: %f" % (episode, cum_rew))
        episode += 1


if __name__ == '__main__':
    config_path = os.path.join(os.path.dirname(__file__), os.pardir, 'inputs.json')
    model_path = os.path.join(os.path.dirname(__file__), os.pardir, 'storage', 'networks', '64_32_3e-4_bs128_angvelpen_rewnorm_small_obs_seeded_2')

    print(f"loading model from {model_path}")

    env = TwoWayCouplingConfigEnv(config_path)
    env = SkipStackWrapper(env, skip=8, stack=4)
    env = SeedOnResetWrapper(env, 1000)
    model = SAC.load(model_path)

    evaluate_model(model, env)
