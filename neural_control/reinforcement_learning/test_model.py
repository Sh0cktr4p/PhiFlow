from gym import Env
from reinforcement_learning.envs.skip_stack_wrapper import SkipStackWrapper
from reinforcement_learning.envs.two_way_coupling_env import TwoWayCouplingConfigEnv
from stable_baselines3.sac import SAC

def evaluate_model(model: SAC, env: Env):
    print(f"writing output to {env.sim_export_path}")

    obs = env.reset()
    episode = 0
    cum_rew = 0

    while episode < 20:
        act, _ = model.predict(obs, deterministic=True)
        obs, rew, done, info = env.step(act)
        cum_rew += rew
        env.render('file')
        if done:
            obs = env.reset()
            print("simulated episode %i. Cumulative reward: %f" % (episode, cum_rew))
            cum_rew = 0
            episode += 1

if __name__ == '__main__':
    config_path = "/home/felix/Code/HiWi/Brener/PhiFlow/neural_control/inputs.json"
    model_path = "/home/felix/Code/HiWi/Brener/PhiFlow/neural_control/storage/networks/128_128_128_3e-4_2grst_bs128_angvelpen_rewnorm_3"

    print(f"loading model from {model_path}")

    env = TwoWayCouplingConfigEnv(config_path)
    env = SkipStackWrapper(env, skip=8, stack=4)
    model = SAC.load(model_path)
    env.seed(0)

    evaluate_model(model, env)
