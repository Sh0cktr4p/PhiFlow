from gym import Env
from reinforcement_learning.envs.skip_stack_wrapper import SkipStackWrapper
from reinforcement_learning.envs.two_way_coupling_env import TwoWayCouplingConfigEnv
from stable_baselines3.sac import SAC

def evaluate_model(model: SAC, env: Env):
    obs = env.reset()
    episode = 0

    while episode < 10:
        act, _ = model.predict(obs, deterministic=True)
        obs, rew, done, info = env.step(act)
        env.render('file')
        if done:
            obs = env.reset()
            print("simulated episode %i" % episode)
            episode += 1

if __name__ == '__main__':
    config_path = "/home/felix/Code/HiWi/Brener/PhiFlow/neural_control/inputs.json"
    model_path = "/home/felix/Code/HiWi/Brener/PhiFlow/neural_control/storage/networks/64_64_64_3e-4_2grst_bs128"

    env = TwoWayCouplingConfigEnv(config_path)
    env = SkipStackWrapper(env, skip=8, stack=4)
    model = SAC.load(model_path)

    evaluate_model(model, env)
