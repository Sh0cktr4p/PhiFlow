import shutil
import time
from typing import Any, Dict, List, Tuple
import os

from matplotlib.pyplot import contour
from modules import ResBlocksFeaturesExtractor

import torch
import numpy as np
from gym import Env
from gym.spaces import Box
from stable_baselines3.sac import SAC
from stable_baselines3.ppo import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from phi import math

from InputsManager import InputsManager
from misc.TwoWayCouplingSimulation import TwoWayCouplingSimulation
from network.misc_funcs import calculate_loss, extract_inputs, Probes, prepare_export_folder, rotate
from reinforcement_learning.envs.rew_norm_wrapper import RewNormWrapper
from reinforcement_learning.envs.skip_stack_wrapper import SkipStackWrapper, FrameStack
from reinforcement_learning.callbacks import EveryNRolloutsPlusStartFinishFunctionCallback


def profile(fn):
    def wrapper(*args, **kwargs):
        before = time.time()
        result = fn(*args, **kwargs)
        print("%s ran for %f seconds" % (str(fn), time.time() - before))
        return result
    
    return wrapper


class TwoWayCouplingEnv(Env):
    def __init__(
        self, 
        device: str,
        n_steps: int,
        dt: float,
        domain_size: Tuple[int, int],
        re: float,
        obs_width: int, 
        obs_height: int, 
        obs_xy: Tuple[int, int], 
        obs_mass: float,
        obs_inertia: float,
        translation_only: bool,
        sponge_intensity: float,
        sponge_size: List[int],
        inflow_velocity: float,
        probes_offset: float, 
        probes_size: float,
        probes_n_rows: int,
        probes_n_columns: int,
        past_window: int,
        n_past_features: int,
        sim_import_path: str,
        sim_export_path: str,
        export_vars: List[str],
        export_stride: int,
        ref_vars: dict,
    ):
        self.device = device
        self.sim = TwoWayCouplingSimulation(device, translation_only)
        self.sim.set_initial_conditions(obs_width, obs_height, path=sim_import_path)
        self.dt = dt
        self.domain_size = domain_size
        self.re = re
        self.obs_mass = obs_mass
        self.obs_inertia = obs_inertia
        self.translation_only = translation_only
        self.sponge_intensity = sponge_intensity
        self.sponge_size = sponge_size
        self.ref_vars = ref_vars
        self.inflow_velocity = inflow_velocity

        self.probes = Probes(
            obs_width / 2 + probes_offset,
            obs_height / 2 + probes_offset,
            probes_size,
            probes_n_rows,
            probes_n_columns,
            obs_xy
        )

        self.n_steps = n_steps
        self.step_idx = 0
        self.epis_idx = 0

        self.pos_objective = None
        self.ang_objective = None
        self.forces = None
        self.torque = None
        self.pos_error = None
        self.rew = None
        self.rew_baseline = np.array(0)

        self.past_window = past_window
        self.n_past_features = n_past_features

        self.sim_export_path = sim_export_path
        self.export_vars = export_vars
        self.export_stride = export_stride
        self.export_folder_created = False

        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()

    def reset(self) -> np.ndarray:
        self.step_idx = 0
        self.sim.setup_world(
            self.re, 
            self.domain_size, 
            self.dt, 
            self.obs_mass, 
            self.obs_inertia, 
            self.inflow_velocity, 
            self.sponge_intensity,
            self.sponge_size
        )
        self.pos_objective = (torch.rand(2) * torch.tensor([40, 20]) + torch.tensor([40, 20])).cuda()
        self.ang_objective = (torch.rand(1) * 2 * math.PI - math.PI).cuda()
        obs, loss = self._extract_inputs()
        self.rew_baseline = self._get_rew(loss, 0, False)
        print("pos objective: %s" % str(self.pos_objective))
        return obs
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, Dict[str, Any]]:
        self.step_idx += 1
        forces, torque = self._split_action_to_force_torque(action)
        if self.translation_only:
            torque *= 0
        self.forces = forces
        self.torque = torque
        global_forces = self._to_global(forces)
        self.sim.apply_forces(global_forces * self.ref_vars['force'], torque * self.ref_vars['torque'])
        self.sim.advect()
        self._make_incompressible()
        self.probes.update_transform(self.sim.obstacle.geometry.center.numpy(), -(self.sim.obstacle.geometry.angle.numpy() - math.PI / 2.0))
        self.sim.calculate_fluid_forces()

        obs, loss = self._extract_inputs()
        done = self._obstacle_leaving_domain() or self.step_idx == self.n_steps
        self.rew = self._get_rew(loss, self.rew_baseline, done)
        info = {}

        if done:
            self.epis_idx += 1

        return obs, self.rew, done, info

    def render(self, mode: str) -> None:
        if not self.export_folder_created:
            self.epis_idx = 0       # Reset episode index for interactive data reader to work properly
            shutil.rmtree(f"{self.sim_export_path}/tensorboard", ignore_errors=True)
            prepare_export_folder(self.sim_export_path, self.step_idx)
            self.export_folder_created = True
        
        probes_points = self.probes.get_points_as_tensor()
        self.sim.probes_points = probes_points.native().detach()
        self.sim.probes_vx = self.sim.velocity.x.sample_at(probes_points).native().detach()
        self.sim.probes_vy = self.sim.velocity.y.sample_at(probes_points).native().detach()
        self.sim.control_force_x, self.sim.control_force_y = self.forces.detach().clone() * self.ref_vars['force']
        self.sim.control_torque = self.torque.detach().clone() * self.ref_vars['torque']
        self.sim.reference_x = self.pos_objective[0].detach().clone()
        self.sim.reference_y = self.pos_objective[1].detach().clone()
        self.sim.reference_angle = self.ang_objective.detach().clone()
        self.sim.error_x = self.pos_error[0]
        self.sim.error_y = self.pos_error[1]
        self.sim.reward = self.rew
        
        self.sim.export_data(
            self.sim_export_path, 
            self.epis_idx, 
            self.step_idx // self.export_stride, 
            self.export_vars, 
            (self.epis_idx==0 and self.step_idx == 0)
        )

    def close(self) -> None:
        pass

    def seed(self, seed=0) -> None:
        print("this is seed, yo")
        torch.manual_seed(seed)

    def _get_action_space(self) -> Box:
        return Box(-1, 1, shape=(3,), dtype=np.float32)

    def _get_observation_space(self) -> Box:
        return Box(-np.inf, np.inf, shape=self.reset().shape, dtype=np.float32)

    def _extract_inputs(self) -> Tuple[np.ndarray, np.ndarray]:
        obs, loss = extract_inputs(self.sim, self.probes, self.pos_objective, self.ang_objective, self.ref_vars, self.translation_only)
        return obs.cpu().numpy().reshape(-1), loss.cpu().numpy().reshape(-1)

    def _get_obs(self) -> np.ndarray:
        return self._extract_inputs()[0]

    def _get_rew(self, loss_inputs: np.ndarray, baseline: np.ndarray, done: bool) -> np.ndarray:
        self.pos_error = loss_inputs[0:2]

        pos_rew = -1 * np.sum(self.pos_error ** 2)

        rew = np.array(pos_rew) - baseline
        if baseline != 0:
            rew = rew / np.abs(baseline)
        
        #print(np.sqrt(np.sum(self.pos_error ** 2)))

        if np.sum(self.pos_error ** 2) < self.ref_vars['destination_zone_size'] ** 2:
            rew += 9

        if not self.translation_only:
            self.ang_error = loss_inputs[4:5]
            # TODO calculate angular reward and add to output

        return rew
        #return pos_rew * 30 + vel_rew if done else vel_rew

    def _make_incompressible(self) -> None:
        converged = False

        while not converged:
            try:
                self.sim.make_incompressible()
                converged = True
            except AssertionError:
                print('Assertion error in make_incompressible, probably non-converging pressure solver')

    def _obstacle_leaving_domain(self) -> bool:
        obstacle_center = self.sim.obstacle.geometry.center
        return math.any(obstacle_center > self.domain_size) or math.any(obstacle_center < (0, 0))

    def _split_action_to_force_torque(self, action: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        control_effort = torch.tensor(action).to(self.device)
        control_effort = torch.clamp(control_effort, -1, 1)
        return control_effort[:2], control_effort[-1:]

    def _to_global(self, force: torch.Tensor) -> torch.Tensor:
        current_obs_angle = -(self.sim.obstacle.geometry.angle - math.PI / 2.0).native()
        return rotate(force, current_obs_angle)


class TwoWayCouplingConfigEnv(TwoWayCouplingEnv):
    def __init__(self, config_path):
        config = InputsManager(config_path)
        config.calculate_properties()

        obs_width = config.simulation['obs_width']
        obs_mass = config.simulation['obs_mass']
        obs_inertia = config.simulation['obs_inertia']
        inflow_velocity = config.simulation['inflow_velocity']
        domain_size = config.simulation['domain_size']

        ref_vars = dict(
            length=obs_width,
            angle=math.PI,
            velocity=inflow_velocity,
            ang_velocity=inflow_velocity / obs_width,
            force=obs_mass * config.max_acc,
            torque=obs_inertia * config.max_ang_acc,
            time=obs_width / inflow_velocity,
            destination_zone_size=domain_size - config.online['destination_margins'] * 2,
        )

        super().__init__(
            device=config.device,
            n_steps=config.online['n_steps'],
            dt = config.simulation['dt'],
            domain_size=domain_size,
            obs_width=obs_width,
            obs_height=config.simulation['obs_height'],
            obs_xy=config.simulation['obs_xy'],
            obs_mass=obs_mass,
            obs_inertia=obs_inertia,
            translation_only=config.translation_only,
            inflow_velocity=inflow_velocity,
            probes_offset=config.probes_offset,
            probes_size=config.probes_size,
            probes_n_rows=config.probes_n_rows,
            probes_n_columns=config.probes_n_columns,
            past_window=config.past_window,
            n_past_features=config.n_past_features,
            sim_import_path=config.online['simulation_path'],
            sim_export_path=config.export_path,
            export_vars=config.export_vars,
            export_stride=config.export_stride,
            ref_vars=ref_vars,
        )

def get_env(skip: int=8, stack: int=4) -> Env:
    env = TwoWayCouplingConfigEnv("/home/felix/Code/HiWi/Brener/PhiFlow/neural_control/inputs.json")
    env = SkipStackWrapper(env, skip=skip, stack=stack)
    env.seed(0)
    return env

def train_model(name: str, log_dir: str, n_timesteps: int, **agent_kwargs) -> SAC:
    model_path = os.path.join("/home/felix/Code/HiWi/Brener/PhiFlow/neural_control/storage/networks", name)
    tb_log_path = os.path.join("/home/felix/Code/HiWi/Brener/PhiFlow/neural_control/storage/tensorboard", log_dir)

    env = get_env()

    
    if os.path.exists(model_path):
        print('model path exists, loading model')
        model = SAC.load(model_path)
    else:
        print('creating new model...')
        model = SAC('MlpPolicy', env, tensorboard_log=tb_log_path, verbose=1, **agent_kwargs)

    def store_fn(_):
        print(f"Storing model to {model_path}...")
        model.save(model_path)
        print("Stored model.")

    model.learn(total_timesteps=n_timesteps, callback=EveryNRolloutsPlusStartFinishFunctionCallback(1000, store_fn), tb_log_name=name)


if __name__ == '__main__':
    import phi.torch.flow as phiflow
    phiflow.TORCH_BACKEND.set_default_device("GPU")
    #train_model('64_64_64_3e-4', 'hparams_tuning', 15000, batch_size=64, learning_starts=32, learning_rate=3e-4, policy_kwargs=dict(net_arch=[64, 64, 64]))
    #train_model('64_64_64_5e-4', 'hparams_tuning', 15000, batch_size=64, learning_starts=32, learning_rate=5e-4, policy_kwargs=dict(net_arch=[64, 64, 64]))
    #train_model('64_64_64_64_3e-4', 'hparams_tuning', 15000, batch_size=64, learning_starts=32, learning_rate=3e-4, policy_kwargs=dict(net_arch=[64, 64, 64, 64]))
    train_model('64_64_64_64_5e-4_2grst_bs128', 'hparams_tuning', 15000, batch_size=128, learning_starts=32, learning_rate=5e-4, gradient_steps=2, policy_kwargs=dict(net_arch=[64, 64, 64, 64]))
    train_model('64_64_64_64_2e-4_2grst_bs128', 'hparams_tuning', 15000, batch_size=128, learning_starts=32, learning_rate=2e-4, gradient_steps=2, policy_kwargs=dict(net_arch=[64, 64, 64, 64]))
    train_model('128_128_128_3e-4_2grst_bs128', 'hparams_tuning', 15000, batch_size=128, learning_starts=32, learning_rate=3e-4, gradient_steps=2, policy_kwargs=dict(net_arch=[128, 128, 128]))
    train_model('64_64_64_3e-4_2grst_bs128', 'hparams_tuning', 50000, batch_size=128, learning_starts=32, learning_rate=3e-4, gradient_steps=2, policy_kwargs=dict(net_arch=[64, 64, 64]))
    #train_model('128_128_3e-4', 'hparams_tuning', 15000, batch_size=64, learning_starts=32, learning_rate=3e-4, policy_kwargs=dict(net_arch=[128, 128]))
    #train_model('128_128_5e-4', 'hparams_tuning', 15000, batch_size=64, learning_starts=32, learning_rate=5e-4, policy_kwargs=dict(net_arch=[128, 128]))
    #train_model('128_128_128_3e-4', 'hparams_tuning', 15000, batch_size=64, learning_starts=32, learning_rate=3e-4, policy_kwargs=dict(net_arch=[128, 128, 128]))
    #train_model('128_128_128_5e-4', 'hparams_tuning', 15000, batch_size=64, learning_starts=32, learning_rate=5e-4, policy_kwargs=dict(net_arch=[128, 128, 128]))
    #train_model('256_256', 'hparams_tuning', 15000, batch_size=64, learning_starts=32, learning_rate=3e-4, policy_kwargs=dict(net_arch=[256, 256]))
    policy_kwargs = dict(
        features_extractor_class=ResBlocksFeaturesExtractor,
        features_extractor_kwargs=dict(
            features_dim=64,
            n_blocks=3,
        ),
        net_arch=[64, 64],
    )

    train_model('3resbl64_64_64_5e-4_2grst_bs128', 'hparams_tuning', 30000, batch_size=128, learning_starts=32, learning_rate=5e-4, gradient_steps=2, policy_kwargs=policy_kwargs)


