from typing import Any, Dict, List, Optional, Tuple
import torch
import numpy as np
from misc.TwoWayCouplingSimulation import TwoWayCouplingSimulation
from misc_funcs import extract_inputs, Probes, prepare_export_folder, rotate
from gym import Env
from gym.spaces import Box
from phi import math
import os
from InputsManager import InputsManager

class TwoWayCouplingTorchEnv(Env):
    def __init__(
        self, 
        device: str,
        n_steps: int,
        dt: float,
        domain_size: Tuple[int, int],
        destination_margins: Tuple[int, int],
        re: float,
        obs_type: str,
        obs_width: float, 
        obs_height: float, 
        obs_xy: Tuple[int, int], 
        obs_mass: float,
        obs_inertia: float,
        translation_only: bool,
        sponge_intensity: float,
        sponge_size: List[int],
        inflow_on: bool,
        inflow_velocity: float,
        probes_offset: float, 
        probes_size: float,
        probes_n_rows: int,
        probes_n_columns: int,
        sim_import_path: str,
        sim_export_path: str,
        export_vars: List[str],
        export_stride: int,
        input_vars: list,
        ref_vars: dict,
    ):
        self.sim = TwoWayCouplingSimulation(device, translation_only)
        print(f"Sim import path: {sim_import_path}")
        self.sim.set_initial_conditions(
            obs_type=obs_type,
            obs_w=obs_width, 
            obs_h=obs_height, 
            path=sim_import_path
        )
        self.dt = dt
        self.destination_margins = destination_margins
        self.domain_size = domain_size
        self.re = re
        self.obs_mass = obs_mass
        self.obs_inertia = obs_inertia
        self.translation_only = translation_only
        self.sponge_intensity = sponge_intensity
        self.sponge_size = sponge_size
        self.input_vars = input_vars
        self.ref_vars = ref_vars
        self.inflow_on = inflow_on
        self.inflow_velocity = inflow_velocity

        self.probes = Probes(
            width_inner=obs_width / 2 + probes_offset,
            height_inner=obs_height / 2 + probes_offset,
            size=probes_size,
            n_rows=probes_n_rows,
            n_columns=probes_n_columns,
            center=obs_xy,
        )

        self.n_steps = n_steps
        self.step_idx = 0
        self.epis_idx = -1

        self.pos_objective = None
        self.ang_objective = None
        self.forces = None
        self.torque = None
        self.pos_error = None
        self.ang_error = None
        self.rew = None
        self.rew_baseline = torch.tensor(0)

        self.sim_export_path = sim_export_path
        self.export_vars = export_vars
        self.export_stride = export_stride
        self.export_folder_created = False

        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()

    @property
    def _obstacle_angle(self) -> torch.Tensor:
        return self.sim.obstacle.geometry.angle.native() - math.PI / 2.0


    def reset(self)-> torch.Tensor:
        self.step_idx = 0
        self.epis_idx += 1
        self.sim.setup_world(
            re=self.re, 
            domain_size=self.domain_size, 
            dt=self.dt, 
            obs_mass=self.obs_mass, 
            obs_inertia=self.obs_inertia, 
            reference_velocity=self.inflow_velocity, 
            sponge_intensity=self.sponge_intensity,
            sponge_size=self.sponge_size,
            inflow_on=self.inflow_on,
        )
        self.pos_objective, self.ang_objective = self._generate_objectives()
        obs, loss = self._extract_inputs()
        rew_baseline = self._get_rew(loss, False)
        #self.rew_baseline 
        return obs

    def step(self, action: torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor, bool, Dict[str, Any]]:
        self.step_idx += 1
        self.forces, self.torque = self._split_action_to_force_torque(action)
        self.forces = self._to_global(self.forces)

        self.sim.apply_forces(self.forces * self.ref_vars['force'], self.torque * self.ref_vars['torque'])
        self.sim.advect()
        converged = self.sim.make_incompressible()
        self.probes.update_transform(self.sim.obstacle.geometry.center.native(), -1 * self._obstacle_angle)
        self.sim.calculate_fluid_forces()
        obs, loss = self._extract_inputs()
        done = self._obstacle_leaving_domain() or self.step_idx == self.n_steps or not converged

        if torch.isnan(torch.sum(obs)):
            print('NaN value in observation!')
            obs[torch.isnan(obs)] = 0
            done = True

        if not self.translation_only and torch.abs(self.sim.obstacle.angular_velocity.numpy()) > self.ref_vars['max_ang_vel']:
            print('Hit maximum angular velocity, ending trajectory')
            done = True

        self.rew = self._get_rew(loss, done, self.rew_baseline)
        info = {}

        return obs, self.rew, done, info

    def render(self):
        pass

    def _get_action_space(self) -> Box:
        dim = 2
        if not self.translation_only:
            dim += 1
        return Box(-1, 1, shape=(dim,), dtype=np.float32)

    def _get_observation_space(self) -> Box:
        shape = self.reset().shape
        self.epis_idx -= 1  # Account for reset function call
        return Box(-np.inf, np.inf, shape=shape, dtype=np.float32)

    def _generate_objectives(self) -> Tuple[torch.Tensor, torch.Tensor]:
        pos_objective_min = torch.tensor(self.destination_margins)
        pos_objective_max = torch.tensor(self.domain_size) - torch.tensor(self.destination_margins)
        pos_objective = pos_objective_min + (torch.rand(2) * (pos_objective_max - pos_objective_min))
        ang_objective = torch.rand(1) * 2 * math.PI - math.PI
        return pos_objective.to(self.sim.device), ang_objective.to(self.sim.device)

    def _extract_inputs(self) -> Tuple[torch.Tensor, dict]:
        obs, loss = extract_inputs(self.input_vars, self.sim, self.probes, self.pos_objective, self.ang_objective, self.ref_vars, self.translation_only)
        return obs, loss

    def _get_rew(self, loss_inputs: dict, done: bool, baseline: Optional[torch.Tensor]=None) -> torch.Tensor:
        self.pos_error = torch.stack([loss_inputs[key] for key in ['error_x', 'error_y']])

        pos_rew = -1 * torch.sum(self.pos_error ** 2)

        ang_vel_rew = -10 * (self.sim.obstacle.angular_velocity.native() / self.ref_vars['max_ang_vel']) ** 2

        rew = pos_rew + ang_vel_rew
        if baseline:
            rew = (rew - baseline) / torch.abs(baseline)

        if torch.sum(self.pos_error ** 2) < 0.15 ** 2:
            rew += 9

        rew = torch.clamp(rew, min=-30)

        if not self.translation_only:
            pass
            # TODO calculate angular reward and add to output

        return rew

    def _obstacle_leaving_domain(self) -> bool:
        obstacle_center = self.sim.obstacle.geometry.center
        return math.any(obstacle_center > self.domain_size) or math.any(obstacle_center < (0, 0))


    def _split_action_to_force_torque(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        control_effort = action.to(self.sim.device)
        force = control_effort[:2]
        torque = torch.tensor([0]).to(self.sim.device) if self.translation_only else control_effort[-1:]
        return force, torque

    def _to_global(self, force: torch.Tensor) -> torch.Tensor:
        return rotate(force, -1 * self._obstacle_angle)


class TwoWayCouplingConfigTorchEnv(TwoWayCouplingTorchEnv):
    def __init__(self, config_path):
        simulation_storage_path = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'storage', 'simulation_data')
        config = InputsManager(config_path)
        config.calculate_properties()

        device = config.device
        max_acc = config.max_acc
        max_vel = config.max_vel
        max_ang_acc = config.max_ang_acc
        max_ang_vel = config.max_ang_vel
        translation_only = config.translation_only
        probes_offset = config.probes_offset
        probes_size = config.probes_size
        probes_n_rows = config.probes_n_rows
        probes_n_columns = config.probes_n_columns
        export_vars = config.export_vars
        export_stride = config.export_stride

        n_steps = config.online['n_timesteps']
        destination_margins = config.online['destinations_margins']
        sim_import_path = os.path.join(simulation_storage_path, config.online['simulation_path'])
        sim_export_path = os.path.join(simulation_storage_path, config.online['export_path'])

        dt = config.simulation['dt']
        domain_size = config.simulation['domain_size']
        re = config.simulation['re']
        obs_type = config.simulation['obs_type']
        obs_width = config.simulation['obs_width']
        obs_height = config.simulation['obs_height']
        obs_xy = config.simulation['obs_xy']
        obs_mass = config.simulation['obs_mass']
        obs_inertia = config.simulation['obs_inertia']
        sponge_intensity = config.simulation['sponge_intensity']
        sponge_size = config.simulation['sponge_size']
        inflow_on = config.simulation['inflow_on']
        inflow_velocity = config.simulation['reference_velocity']

        input_vars = config.nn_vars

        ref_vars = dict(
            length=obs_width,
            angle=math.PI,
            velocity=inflow_velocity,
            ang_velocity=inflow_velocity / obs_width,
            force=obs_mass * max_acc,
            torque=obs_inertia * max_ang_acc,
            time=obs_width / inflow_velocity,
            destination_zone_size=domain_size - destination_margins * 2,
            max_vel=max_vel,
            max_ang_vel=max_ang_vel,
        )

        print("Ref vars: %s" % ref_vars)

        super().__init__(
            device=device,
            n_steps=n_steps,
            dt=dt,
            domain_size=domain_size,
            destination_margins=destination_margins,
            re=re,
            obs_type=obs_type,
            obs_width=obs_width,
            obs_height=obs_height,
            obs_xy=obs_xy,
            obs_mass=obs_mass,
            obs_inertia=obs_inertia,
            translation_only=translation_only,
            sponge_intensity=sponge_intensity,
            sponge_size=sponge_size,
            inflow_on=inflow_on,
            inflow_velocity=inflow_velocity,
            probes_offset=probes_offset,
            probes_size=probes_size,
            probes_n_rows=probes_n_rows,
            probes_n_columns=probes_n_columns,
            sim_import_path=sim_import_path,
            sim_export_path=sim_export_path,
            export_vars=export_vars,
            export_stride=export_stride,
            input_vars=input_vars,
            ref_vars=ref_vars,
        )

if __name__ == '__main__':
    env = TwoWayCouplingConfigTorchEnv('neural_control/inputs.json')

    obs0 = env.reset()
    print(obs0)
    act1 = torch.tensor([1, 1], dtype=torch.float32, requires_grad=True)
    obs1, rew1, done, _ = env.step(act1)
    print(obs1)
    print(rew1)
    act2 = torch.tensor([1, 1], dtype=torch.float32, requires_grad=True)
    obs2, rew2, done, _ = env.step(act2)
    print(obs2)
    print(rew2)

    rew2.backward()
    print("gradient: " + str(act1.grad))

    '''
    env = TorchTestEnv(1, 1000)

    obss = []
    rews = []
    obss.append(env.reset())

    done = False

    acts = [torch.tensor(i, dtype=torch.float32, requires_grad=True) for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300]]

    i = 0

    while not done:
        #custom_input = int(input())
        list_input = acts[i]
        obs, rew, done, _ = env.step(list_input)
        env.render()
        obss.append(obs)
        rews.append(rew)
        i += 1
    
    print(obss)
    print(rews)

    print(acts[0].grad())
    rews[-1].backward()
    print(acts[0].grad())
    '''