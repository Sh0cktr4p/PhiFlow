import os
import shutil
from argparse import ArgumentParser

from gym import Env
from reinforcement_learning.extract_model import store_sac_actor_as_torch_module
from stable_baselines3.sac import SAC
from stable_baselines3.common.callbacks import CallbackList

from reinforcement_learning.envs.two_way_coupling_env import TwoWayCouplingConfigEnv
from reinforcement_learning.envs.stack_observations_wrapper import StackObservations
from reinforcement_learning.envs.seed_on_reset_wrapper import SeedOnResetWrapper
from reinforcement_learning.callbacks import EveryNTimestepsPlusStartFinishFunctionCallback
from InputsManager import InputsManager


CONFIG_FILENAME = 'inputs.json'
AGENT_FILENAME = 'agent.zip'
TORCH_MODEL_FILENAME_TEMPLATE = 'trained_model_%04i.pth'


def get_env(config_path: str) -> Env:
    env = TwoWayCouplingConfigEnv(config_path)
    env = StackObservations(env, n_present_features=4, n_past_features=4, past_window=2, append_past_actions=True)
    env = SeedOnResetWrapper(env)
    
    print('Observation space shape: %s' % str(env.observation_space.shape))
    return env


def create_model_folder(name: str, storage_base_path: str, config_path: str):
    path_to_model_folder = os.path.join(storage_base_path, name)
    assert os.path.exists(config_path)
    assert os.path.exists(storage_base_path)
    assert not os.path.exists(path_to_model_folder)

    os.mkdir(path_to_model_folder)
    shutil.copyfile(config_path, os.path.join(path_to_model_folder, CONFIG_FILENAME))

    return path_to_model_folder    

def train_model(path_to_model_folder: str, log_path):
    name = os.path.basename(path_to_model_folder)
    config_path = os.path.join(path_to_model_folder, CONFIG_FILENAME)
    agent_path = os.path.join(path_to_model_folder, AGENT_FILENAME)
    torch_module_path = os.path.join(path_to_model_folder, TORCH_MODEL_FILENAME_TEMPLATE % 0)

    assert os.path.exists(path_to_model_folder)
    assert os.path.exists(config_path)
    assert not os.path.exists(agent_path) # Training continuation currently not supported

    inp = InputsManager(config_path)
    env = get_env(config_path)
    agent = SAC('MlpPolicy', env, tensorboard_log=log_path, verbose=1, **inp.rl['training_params'])
    
    def store_fn(_):
        print('Storing agent...')
        agent.save(agent_path)
        store_sac_actor_as_torch_module(agent_path, torch_module_path)

    callback = CallbackList([
        EveryNTimestepsPlusStartFinishFunctionCallback(inp.rl['model_export_stride'], store_fn)
    ])

    agent.learn(total_timesteps=inp.rl['n_iterations'], callback=callback, tb_log_name=name)


if __name__ == '__main__':
    base_directory = os.path.join(os.path.dirname(__file__), os.pardir)
    base_storage_directory = os.path.join(base_directory, 'storage')
    default_config_path = os.path.join(base_directory, 'inputs.json')
    default_model_storage_path = os.path.join(base_storage_directory, 'networks')
    default_log_storage_path = os.path.join(base_storage_directory, 'tensorboard')

    parser = ArgumentParser()
    parser.add_argument('-n', '--name', dest='name', type=str, help='model name, no storing if not specified')
    parser.add_argument('-c', '--config', dest='config', type=str, default=default_config_path, help='path to config file')
    parser.add_argument('-p', '--path', dest='path', type=str, default=default_model_storage_path, help='path to model storage folder')
    parser.add_argument('-l', '--log', dest='log', type=str, default=default_log_storage_path, help='path to tensorboard logs')

    args = parser.parse_args()

    if args.name:
        path_to_model_folder = create_model_folder(args.name, args.path, args.config)
        train_model(path_to_model_folder, args.log)
    else:
        print('\033[31mNo name specified, training in sandbox mode (no storing)\033[0m')
        inp = InputsManager(args.config)
        env = get_env(args.config)
        agent = SAC('MlpPolicy', env, verbose=1, **inp.rl['training_params'])
        agent.learn(total_timesteps=inp.rl['n_iterations'])
