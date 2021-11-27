import os
from InputsManager import InputsManager, RLInputsManager
from misc_funcs import extract_inputs
import torch
import numpy as np

from stable_baselines3.sac import SAC
from reinforcement_learning.extract_model import load_sac_torch_module, SACActorModule
from reinforcement_learning.envs.two_way_coupling_env import TwoWayCouplingConfigEnv

if __name__ == '__main__':
    agent_path = '/home/felix/Code/HiWi/Brener/PhiFlow/neural_control/storage/networks/simple_env_norewnorm_noskipstack.zip'
    test_module_path = '/home/felix/Documents/GuidedResearch/rl/trained_model_0000.pth'
    ref_module_path = '/home/felix/Code/Test/PythonStuff/noskipstack.pkl'

    agent = SAC.load(agent_path)
    test_module = load_sac_torch_module(test_module_path).cpu()
    ref_module = load_sac_torch_module(ref_module_path).cpu()

    env = TwoWayCouplingConfigEnv('neural_control/inputs.json')

    obs = env.reset()
    obs_tensor = torch.tensor(obs.reshape(1, *obs.shape))
    agent_prediction, _ = agent.predict(obs, deterministic=True)
    test_module_prediction = test_module(obs_tensor).detach().numpy()
    ref_module_prediction = ref_module(obs_tensor).detach().numpy()
    print('Agent prediction: ', agent_prediction)
    print('Breners model prediction: ', test_module_prediction)
    print('My model prediction: ', ref_module_prediction)

    inp = InputsManager('/home/felix/Documents/GuidedResearch/rl/inputs.json')
    rl_inp = RLInputsManager(inp.past_window, inp.n_past_features, inp.rl['n_snapshots_per_window'], 'cpu')

    def obs_to_inp(o):
        nn_inputs, _ = extract_inputs(inp.nn_vars, env.sim, env.probes, env.pos_objective, env.ang_objective, env.ref_vars, env.translation_only)
        rl_inp.add_snapshot(nn_inputs.cpu().view(1, -1))
        return rl_inp.values.view(1, -1)

    for policy_function in [
        lambda o: agent.predict(o, deterministic=True)[0],
        lambda o: test_module(obs_to_inp(o)).detach().numpy().squeeze(0),
        lambda o: ref_module(torch.tensor(o).unsqueeze(0)).detach().numpy().squeeze(0),
    ]:
        done = False
        obs = env.reset()
        a = np.zeros((3,), dtype=np.float32)
        while not done:
            obs, _, done, _ = env.step(a)
            env.render('file')
            a = policy_function(obs)
