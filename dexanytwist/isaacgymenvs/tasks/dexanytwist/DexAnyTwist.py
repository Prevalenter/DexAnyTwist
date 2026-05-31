# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import json

import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi

from isaacgymenvs.utils.torch_jit_utils import scale, unscale, quat_mul, quat_conjugate, \
    to_torch, get_axis_params, torch_rand_float, tensor_clamp, quat_apply
from isaacgymenvs.tasks.base.vec_task import VecTask
from isaacgymenvs.utils.rna_util import RandomNetworkAdversary

from gym import spaces

from isaacgymenvs.tasks.dexanytwist.config_load import get_twist_config


class DexAnyTwist(VecTask):
    OBS_DIMS = {
        "full_no_vel": 200,
        "full": 72,
        "full_state": 88,
    }
    NUM_STATES = 350
    NUM_ACTIONS = 40
    NUM_DOF_VALVE = 3
    UP_AXIS = "z"
    DEFAULT_SHADOW_HAND_DOF_POS = [0.0, 1.57, 0.0, 0.6, 0.0, -0.3, 0.4, -0.5, 0.7, -1.2]
    FINGERTIP_FORCE_MASK = [0, 1, 2, 6, 7, 8, 12, 13, 14]
    METRIC_SAVE_DIR = "experiments/DexTwistAnything/data"
    METRIC_SAVE_INTERVAL = 50
    METRIC_EXIT_STEPS = 3000

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self._load_twist_dataset()
        self.random_first_epoch_train = self.cfg['env']['first_epoch_train']
        self._load_hand_config()
        self._init_metric_buffers(rl_device)
        self._read_task_config()
        self._configure_observation_space()

        super().__init__(
            config=self.cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )

        self._configure_episode_timing()
        self._configure_viewer_camera()
        self._acquire_sim_tensors()
        self._init_runtime_buffers()

    def _load_twist_dataset(self):
        self.with_points_cloud = self.cfg["env"]["with_points_cloud"]

        self.twist_all_config, self.num_twist_all_cls, self.twist_all_cls_list = get_twist_config(
            dataset='twist_all',
            with_points_cloud=self.with_points_cloud,
        )

        if self.cfg['dataset_name'] == 'dataset_all':
            dataset_filename = os.path.join(os.path.dirname(__file__), "config/twist_anything/split/twist_cls_list.txt")
        elif self.cfg['dataset_name'] == 'train_all':
            dataset_filename = os.path.join(os.path.dirname(__file__), "config/twist_anything/split/twist_train_list.txt")
        else:
            raise ValueError(f"dataset_name {self.cfg['dataset_name']} is not supported")

        with open(dataset_filename, 'r') as f:
            self.twist_list_used = f.read().splitlines()
        
        self.twist_config = {}
        self.twist_object_used_index_list = []

        for cls_key in self.twist_all_config.keys():
            if cls_key in self.twist_list_used:
                self.twist_config[cls_key] = self.twist_all_config[cls_key]
                self.twist_object_used_index_list.append(self.twist_all_cls_list.index(cls_key))

        self.twist_cls_list = list(self.twist_config.keys())
        self.num_twist_cls = len(self.twist_cls_list)
        
        print('self.num_twist_cls: ', self.num_twist_cls)

    def _load_hand_config(self):
        hand_cfg_path = os.path.join(os.path.dirname(__file__), f"config/hand_config/{self.cfg['env']['hand_cfg']}")
        with open(hand_cfg_path, 'r') as f:
            self.hand_cfg = json.load(f)

    def _init_metric_buffers(self, rl_device):
        self.cons_successes_cls_list = torch.zeros(self.num_twist_cls, device=rl_device)
        self.cons_successes_cls_list_traj = []
        self.csc_list = []
        self.done_list = []
        self.reset_goal_list = []

    def _read_task_config(self):
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]
        
        self.calculate_cls_metric = self.cfg['env']['calculate_cls_metric']
        
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.action_delta_penalty_scale = self.cfg['env']['actionDeltaPenaltyScale']
        self.joint_velocity_penalty_scale = self.cfg['env']['jointVelocityPenaltyScale']

        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]
        self.success_tolerance = self.cfg["env"]["successTolerance"]
        self.reach_goal_bonus = self.cfg["env"]["reachGoalBonus"]
        self.fall_dist = self.cfg["env"]["fallDistance"]
        self.fall_penalty = self.cfg["env"]["fallPenalty"]
        self.rot_eps = self.cfg["env"]["rotEps"]

        self.vel_obs_scale = 0.2
        self.force_torque_obs_scale = 10.0

        self.reset_position_noise = self.cfg["env"]["resetPositionNoise"]
        self.reset_dof_pos_noise = self.cfg["env"]["resetDofPosRandomInterval"]
        self.reset_dof_vel_noise = self.cfg["env"]["resetDofVelRandomInterval"]

        self.force_scale = self.cfg["env"].get("forceScale", 0.0)
        self.force_prob_range = self.cfg["env"].get("forceProbRange", [0.001, 0.1])
        self.force_decay = self.cfg["env"].get("forceDecay", 0.99)
        self.force_decay_interval = self.cfg["env"].get("forceDecayInterval", 0.08)

        self.shadow_hand_dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        self.print_success_stat = self.cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = self.cfg["env"].get("averFactor", 0.1)

        self.ignore_z = False

        self.obs_type = self.cfg["env"]["observationType"]
        self.enable_rna = "random_network_adversary" in self.cfg["env"] and self.cfg["env"]["random_network_adversary"]["enable"]

        if self.enable_rna:
            if "prob" in self.cfg["env"]["random_network_adversary"]:
                self.action_perturb_prob = self.cfg["env"]["random_network_adversary"]["prob"]
            
            self.random_adversary_weight_sample_freq = self.cfg["env"]["random_network_adversary"]["weight_sample_freq"]

        if not (self.obs_type in ["full_no_vel", "full", "full_state"]):
            raise Exception(
                "Unknown type of observations!\nobservationType should be one of: [openai, full_no_vel, full, full_state]")

        print("Obs type:", self.obs_type)
        
        self.num_dof_valve = self.NUM_DOF_VALVE
        self.num_obs_dict = dict(self.OBS_DIMS)
        
        self.fingertips = self.hand_cfg['finger_tips_name']
        self.num_fingertips = len(self.fingertips)

        self.up_axis = self.UP_AXIS

        self.asymmetric_obs = self.cfg["env"]["asymmetric_observations"]
        
        if self.asymmetric_obs:
            self.asym_state_fingertip_state = self.cfg["env"]["asym_state_fingertip_state"]
            self.asym_state_dr_params = self.cfg["env"]["asym_state_dr_params"]

    def _configure_observation_space(self):
        self.obs_space = spaces.Box(
            np.ones(self.num_obs_dict[self.obs_type]) * -np.inf,
            np.ones(self.num_obs_dict[self.obs_type]) * np.inf,
        )
        self.state_space = spaces.Box(
            np.ones(self.NUM_STATES) * -np.inf,
            np.ones(self.NUM_STATES) * np.inf,
        )

        self.cfg["env"]["numObservations"] = self.num_obs_dict[self.obs_type]
        self.cfg["env"]["numStates"] = self.NUM_STATES
        self.cfg["env"]["numActions"] = self.NUM_ACTIONS

    def _configure_episode_timing(self):
        self.dt = self.sim_params.dt
        control_freq_inv = self.cfg["env"].get("controlFrequencyInv", 1)
        print('self.sim_params.dt: ', self.sim_params.dt, control_freq_inv)

        if self.reset_time > 0.0:
            self.max_episode_length = int(round(self.reset_time/(control_freq_inv * self.dt)))
            print("Reset time: ", self.reset_time)
            print("New episode length: ", self.max_episode_length)

    def _configure_viewer_camera(self):
        if self.viewer is not None:
            cam_pos = gymapi.Vec3(0.5, -0.5, 0.9)
            cam_target = gymapi.Vec3(0.0, -0.3, 0.5)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def _acquire_sim_tensors(self):
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        print(rigid_body_tensor.shape)
        
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)

        if self.obs_type == "full_state" or self.asymmetric_obs:
            sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
            self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, self.num_fingertips * 6)

            dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
            self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_shadow_hand_dofs+self.num_dof_valve)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.shadow_hand_default_dof_pos = torch.tensor(
            self.DEFAULT_SHADOW_HAND_DOF_POS,
            dtype=torch.float,
            device=self.device,
        )
        
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.shadow_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_shadow_hand_dofs]
        self.shadow_hand_dof_pos = self.shadow_hand_dof_state[..., 0]
        self.shadow_hand_dof_vel = self.shadow_hand_dof_state[..., 1]
        
        self.net_contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)
        
        self.valve_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_shadow_hand_dofs:]

        print('self.shadow_hand_dof_pos: ', self.shadow_hand_dof_pos.shape)
        print('self.shadow_hand_dof_vel: ', self.shadow_hand_dof_vel.shape)

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        print("Num dofs: ", self.num_dofs)

    def _init_runtime_buffers(self):
        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        self.global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32, device=self.device).view(self.num_envs, -1)

        self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        self.av_factor = to_torch(self.av_factor, dtype=torch.float, device=self.device)

        self.total_successes = 0
        self.total_resets = 0

        self.force_decay = to_torch(self.force_decay, dtype=torch.float, device=self.device)
        self.force_prob_range = to_torch(self.force_prob_range, dtype=torch.float, device=self.device)
        self.random_force_prob = torch.exp((torch.log(self.force_prob_range[0]) - torch.log(self.force_prob_range[1]))
                                           * torch.rand(self.num_envs, device=self.device) + torch.log(self.force_prob_range[1]))

        self.rb_forces = torch.zeros((self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device)

        self.control_steps = self.cfg['env']['controlStepsBegin']

        self.update_smoothing_factor = self.cfg['env']['adaptiveParameter']['enable']
        self.consecutive_successes_threshold = self.cfg['env']['adaptiveParameter']['consecutiveSuccessesThreshold']
        self.act_moving_average_lower, self.act_moving_average_upper = self.cfg['env']['adaptiveParameter']['rangeActionsMovingAverage']
        self.ap_update_step = self.cfg['env']['adaptiveParameter']['schedule_freq']
        self.ap_num_iter = self.cfg['env']['adaptiveParameter']['numIteration']
        self.ap_scale = self.cfg['env']['adaptiveParameter']['ScaleInitial']

        self.action_delta_penalty_scale_choice, self.joint_velocity_penalty_scale_choice = 0, 0

        self.update_action_moving_average()
        self.update_velocity_penalty_factor()

        self.enable_random_obs = self.cfg["env"]["random_cube_observation"]["enable"]
        self.random_cube_pose_prob = self.cfg["env"]["random_cube_observation"]["prob"]
        
        self.fingertip_kp_list = torch.zeros((self.num_envs, 3, 3), dtype=torch.float, device=self.device)
        self.valve_kp_list = torch.zeros((self.num_envs, 3, 3), dtype=torch.float, device=self.device)
        self.keypoints_closest_distance = torch.zeros((self.num_envs, 3), dtype=torch.double, device=self.device)

        self.fingertip_kp_offset_tensor = to_torch(self.hand_cfg['finger_bias'], device=self.device).repeat((self.num_envs, 3, 1))* 0.05

        self.end_rand = torch.randint(0, 599, (self.num_envs,), dtype=torch.int32, device=self.device)

    def create_sim(self):
        self.dt = self.sim_params.dt
        self.up_axis_idx = 2

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        if self.randomize:
            if self.asym_state_dr_params:
                self.num_system_params = 90
                self.sim_params_buf = torch.zeros((self.num_envs, 
                                                   self.num_system_params), 
                                                   dtype=torch.float, 
                                                   device=self.device)
            
            self.apply_randomizations(self.randomization_params)
            if self.asym_state_dr_params:
                self.update_sim_para(first_randomization=True)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_hand_asset_options(self):
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01

        asset_options.vhacd_enabled = False
        asset_options.vhacd_params = gymapi.VhacdParams()
        asset_options.vhacd_params.resolution = 4
        asset_options.vhacd_params.max_convex_hulls = 6

        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        return asset_options

    def _create_object_asset_options(self):
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.fix_base_link = True
        object_asset_options.vhacd_enabled = True
        object_asset_options.vhacd_params = gymapi.VhacdParams()
        object_asset_options.vhacd_params.resolution = 2
        object_asset_options.vhacd_params.max_convex_hulls = 50
        return object_asset_options

    def _configure_hand_dof_props(self, allegro_hand_asset):
        shadow_hand_dof_props = self.gym.get_asset_dof_properties(allegro_hand_asset)

        self.shadow_hand_dof_lower_limits = []
        self.shadow_hand_dof_upper_limits = []
        self.shadow_hand_dof_default_pos = []
        self.shadow_hand_dof_default_vel = []
        self.sensors = []

        for i in range(self.num_shadow_hand_dofs):
            self.shadow_hand_dof_lower_limits.append(shadow_hand_dof_props['lower'][i])
            self.shadow_hand_dof_upper_limits.append(shadow_hand_dof_props['upper'][i])
            self.shadow_hand_dof_default_pos.append(0.0)
            self.shadow_hand_dof_default_vel.append(0.0)

            shadow_hand_dof_props['effort'][i] = 0.5
            shadow_hand_dof_props['stiffness'][i] = 2
            shadow_hand_dof_props['damping'][i] = 0.1
            shadow_hand_dof_props['friction'][i] = 0.01
            shadow_hand_dof_props['armature'][i] = 0.001
            shadow_hand_dof_props['velocity'][i] = 3.14
            print("Max effort: ", shadow_hand_dof_props['effort'][i])

        print('shadow_hand_dof_lower_limits', self.shadow_hand_dof_lower_limits)
        print('shadow_hand_dof_upper_limits', self.shadow_hand_dof_upper_limits)

        self.actuated_dof_indices = to_torch(self.actuated_dof_indices, dtype=torch.long, device=self.device)
        self.shadow_hand_dof_lower_limits = to_torch(self.shadow_hand_dof_lower_limits, device=self.device)
        self.shadow_hand_dof_upper_limits = to_torch(self.shadow_hand_dof_upper_limits, device=self.device)
        self.shadow_hand_dof_default_pos = to_torch(self.shadow_hand_dof_default_pos, device=self.device)
        self.shadow_hand_dof_default_vel = to_torch(self.shadow_hand_dof_default_vel, device=self.device)

        return shadow_hand_dof_props

    def _configure_object_dof_props(self, current_object_asset, current_object_type):
        object_dof_props = self.gym.get_asset_dof_properties(current_object_asset)

        object_dof_props['stiffness'][0] = 0.001
        object_dof_props['damping'][0] = 0.001
        object_dof_props['friction'][0] = 0.001
        object_dof_props['effort'][0] = 0.001

        if self.twist_config[current_object_type]['xy_axis_limit'] > 0.1:
            object_dof_props['stiffness'][1:] = 0.0
            object_dof_props['damping'][1:] = 0.0
            object_dof_props['friction'][1:] = 0.001
            object_dof_props['effort'][1:] = 0.001
        else:
            object_dof_props['stiffness'][1:] = 1000
            object_dof_props['damping'][1:] = 1000
            object_dof_props['friction'][1:] = 1000
            object_dof_props['effort'][1:] = 1000

        return object_dof_props

    def _create_hand_start_pose(self):
        shadow_hand_start_pose = gymapi.Transform()
        shadow_hand_start_pose.p = gymapi.Vec3(*get_axis_params(0.7, self.up_axis_idx))
        shadow_hand_start_pose.r = (
            gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), 0.0*np.pi)
            * gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), 1.15 * np.pi)
            * gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 0.0 * np.pi)
        )
        return shadow_hand_start_pose

    def _create_object_start_pose(self, shadow_hand_start_pose, current_object_type):
        local_object_start_pose = gymapi.Transform()
        local_object_start_pose.p = gymapi.Vec3()
        local_object_start_pose.p.x = shadow_hand_start_pose.p.x

        twist_pose = self.twist_config[current_object_type]['pose']

        noise_y = (torch.rand(1) * (0.02 - (-0.02))) - 0.02
        noise_z = (torch.rand(1) * (0.0 - (-0.03))) - 0.03

        self.object_pos_noise_list.append([noise_y, noise_z])

        local_object_start_pose.p.x = shadow_hand_start_pose.p.x + twist_pose[0]
        local_object_start_pose.p.y = shadow_hand_start_pose.p.y + twist_pose[1] + noise_y
        local_object_start_pose.p.z = shadow_hand_start_pose.p.z + twist_pose[2] + noise_z

        local_object_start_pose.r = (
            gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), 0.0*np.pi)
            * gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), 0.0 * np.pi)
            * gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 0.0 * np.pi)
        )
        return local_object_start_pose

    def _load_object_assets(self, asset_root):
        object_asset_options = self._create_object_asset_options()
        self.object_assets = {}
        for twist_cls in self.twist_cls_list:
            twist_object_file = self.twist_config[twist_cls]['urdf_path']
            self.object_assets[twist_cls] = self.gym.load_asset(
                self.sim,
                asset_root,
                twist_object_file,
                object_asset_options,
            )

    def _create_force_sensors(self, allegro_hand_asset):
        if self.obs_type == "full_state" or (self.asymmetric_obs and self.asym_state_fingertip_state):
            sensor_pose = gymapi.Transform()
            for ft_handle in self.fingertip_handles:
                print(f'create force sensor for {ft_handle}')
                self.gym.create_asset_force_sensor(allegro_hand_asset, ft_handle, sensor_pose)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../assets')
        
        allegro_hand_asset_file = self.hand_cfg['urdf_path']

        asset_options = self._create_hand_asset_options()

        allegro_hand_asset = self.gym.load_asset(self.sim, asset_root, allegro_hand_asset_file, asset_options)

        self.num_shadow_hand_bodies = self.gym.get_asset_rigid_body_count(allegro_hand_asset)
        self.num_shadow_hand_shapes = self.gym.get_asset_rigid_shape_count(allegro_hand_asset)
        self.num_shadow_hand_dofs = self.gym.get_asset_dof_count(allegro_hand_asset)
        print("Num dofs: ", self.num_shadow_hand_dofs)
        self.num_shadow_hand_actuators = self.num_shadow_hand_dofs

        self.actuated_dof_indices = [i for i in range(self.num_shadow_hand_dofs)]

        shadow_hand_dof_props = self._configure_hand_dof_props(allegro_hand_asset)

        self.fingertip_handles = [self.gym.find_asset_rigid_body_index(allegro_hand_asset, name) for name in self.fingertips]

        self._create_force_sensors(allegro_hand_asset)
        self._load_object_assets(asset_root)
        shadow_hand_start_pose = self._create_hand_start_pose()

        max_agg_bodies = self.num_shadow_hand_bodies + 2*100
        max_agg_shapes = self.num_shadow_hand_shapes + 2*100

        self.object_types = np.random.choice(self.twist_cls_list, self.num_envs)

        self.allegro_hands = []
        self.envs = []

        self.object_init_state = []
        self.hand_start_states = []

        self.hand_indices = []
        self.object_indices = []

        self.object_handles = []

        shadow_hand_rb_count = self.gym.get_asset_rigid_body_count(allegro_hand_asset)

        self.object_cls_number_list = []
        
        self.valve_kp_bias_tensor = []
        
        self.points_feature_tensor_list = []
        self.rotation_direction_tensor_list = []
        
        self.object_pos_noise_list = []
        
        for i in range(self.num_envs):
            current_object_type = self.object_types[i]

            current_object_asset = self.object_assets[current_object_type]
            self.object_cls_number_list.append(self.twist_all_cls_list.index(current_object_type))
            self.valve_kp_bias_tensor.append(self.twist_config[current_object_type]['kp_bais'])
            
            self.points_feature_tensor_list.append(self.twist_config[current_object_type]['points_feature'])
            self.rotation_direction_tensor_list.append(self.twist_config[current_object_type]['axis'])

            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            allegro_hand_actor = self.gym.create_actor(env_ptr, allegro_hand_asset, shadow_hand_start_pose, "hand", i, -1, 0)
            self.hand_start_states.append([shadow_hand_start_pose.p.x, shadow_hand_start_pose.p.y, shadow_hand_start_pose.p.z,
                                           shadow_hand_start_pose.r.x, shadow_hand_start_pose.r.y, shadow_hand_start_pose.r.z, shadow_hand_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])
            self.gym.set_actor_dof_properties(env_ptr, allegro_hand_actor, shadow_hand_dof_props)
            hand_idx = self.gym.get_actor_index(env_ptr, allegro_hand_actor, gymapi.DOMAIN_SIM)
            self.hand_indices.append(hand_idx)

            if self.obs_type == "full_state" or self.asymmetric_obs:
                self.gym.enable_actor_dof_force_sensors(env_ptr, allegro_hand_actor)

            local_object_start_pose = self._create_object_start_pose(shadow_hand_start_pose, current_object_type)

            object_handle = self.gym.create_actor(env_ptr, current_object_asset, local_object_start_pose, "object", i, 2, 0)
                
            self.object_init_state.append([local_object_start_pose.p.x, local_object_start_pose.p.y, local_object_start_pose.p.z,
                                           local_object_start_pose.r.x, local_object_start_pose.r.y, local_object_start_pose.r.z, local_object_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])

            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)

            if self.asym_state_dr_params:
                self.object_handles.append(object_handle)

            object_dof_props = self._configure_object_dof_props(current_object_asset, current_object_type)
            self.gym.set_actor_dof_properties(env_ptr, object_handle, object_dof_props)

            self.object_indices.append(object_idx)


            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.allegro_hands.append(allegro_hand_actor)

        
        
        self.object_cls_number_tensor = to_torch(self.object_cls_number_list, device=self.device)
        
        self.valve_kp_bias_tensor =  to_torch(self.valve_kp_bias_tensor, device=self.device)

        self.points_feature_tensor_list = to_torch(self.points_feature_tensor_list, device=self.device) 
        
        self.rotation_direction_tensor_list = to_torch(self.rotation_direction_tensor_list, device=self.device)[:, None]


        first_object_asset = self.object_assets[self.object_types[0]]
        object_rb_count = self.gym.get_asset_rigid_body_count(first_object_asset)
        self.object_rb_handles = list(range(shadow_hand_rb_count, shadow_hand_rb_count + object_rb_count))

        object_rb_props = self.gym.get_actor_rigid_body_properties(self.envs[0], self.object_handles[0] if hasattr(self, 'object_handles') and self.object_handles else self.allegro_hands[0])
        self.object_rb_masses = [prop.mass for prop in object_rb_props]
        self.object_init_state = to_torch(self.object_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)

        self.prev_object_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)

        self.goal_states = self.object_init_state.clone()
        self.goal_init_state = self.goal_states.clone()
        self.hand_start_states = to_torch(self.hand_start_states, device=self.device).view(self.num_envs, 13)

        self.object_rb_handles = to_torch(self.object_rb_handles, dtype=torch.long, device=self.device)
        self.object_rb_masses = to_torch(self.object_rb_masses, dtype=torch.float, device=self.device)

        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)

        self.obs_nan_reset_index = torch.isnan(self.goal_init_state).any(1)
        print(self.obs_nan_reset_index.shape)

        self._init_random_network_adversary()
        self.random_cube_poses = torch.zeros(self.num_envs, 7, device=self.device)

    def _init_random_network_adversary(self):
        if self.enable_rna:
            softmax_bins = 32 
            num_dofs = len(self.shadow_hand_dof_lower_limits)
            self.discretised_dofs = torch.zeros((num_dofs, softmax_bins)).to(self.device)

            for i in range(0, len(self.shadow_hand_dof_lower_limits)):
                self.discretised_dofs[i] = torch.linspace(self.shadow_hand_dof_lower_limits[i], 
                                                          self.shadow_hand_dof_upper_limits[i], steps=softmax_bins).to(self.device)

            self.rna_network = RandomNetworkAdversary(num_envs=self.num_envs, in_dims=num_dofs+7, \
                out_dims=num_dofs, softmax_bins=softmax_bins, device=self.device)

    def get_rna_alpha(self):
        if self.randomize:
            return torch.rand(self.num_envs, 1, device=self.device)
        return torch.zeros(self.num_envs, 1, device=self.device)

    def get_random_network_adversary_action(self, canonical_action):

        if self.enable_rna:

            if self.last_step > 0 and self.last_step % self.random_adversary_weight_sample_freq == 0:
                self.rna_network._refresh()

            object_pose = self.root_state_tensor[self.object_indices, 0:7]

            rand_action_softmax = self.rna_network(torch.cat([self.shadow_hand_dof_pos, object_pose], axis=-1))
            rand_action_inds    = torch.argmax(rand_action_softmax, axis=-1)

            rand_action_inds  = torch.permute(rand_action_inds, (1, 0))
            rand_perturbation = torch.gather(self.discretised_dofs, 1, rand_action_inds)
            rand_perturbation = torch.permute(rand_perturbation, (1, 0))

            rand_perturbation = unscale(rand_perturbation, 
                                        self.shadow_hand_dof_lower_limits[self.actuated_dof_indices],
                                        self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])

            action_perturb_mask = torch.rand(self.num_envs, 1, device=self.device) < self.action_perturb_prob                                        
            rand_perturbation = ~action_perturb_mask * canonical_action + action_perturb_mask * rand_perturbation

            rna_alpha = self.get_rna_alpha()

            rand_perturbation = rna_alpha * rand_perturbation + (1 - rna_alpha) * canonical_action

            return rand_perturbation

        else:
            return canonical_action

    def compute_reward(self, actions):
        if self.random_first_epoch_train:
            self.progress_buf = torch.randint(0, self.max_episode_length-1, self.progress_buf.shape, dtype=torch.int32, device=self.device)
            self.random_first_epoch_train = False

        self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:], self.progress_buf[:], self.successes[:], self.consecutive_successes[:], reward_component_dict, successes_raw = compute_hand_reward(
            self.obs_nan_reset_index, self.rew_buf, self.reset_buf, self.reset_goal_buf, self.progress_buf, self.successes, self.consecutive_successes,
            self.max_episode_length, self.valve_dof_state, self.object_pos, self.object_rot, self.goal_pos, self.goal_rot,
            self.cur_targets , self.prev_targets, self.shadow_hand_dof_vel.clone(), self.keypoints_closest_distance,
            self.dist_reward_scale, self.rot_reward_scale, self.rot_eps, self.actions, self.action_penalty_scale,
            self.action_delta_penalty_scale_choice, self.joint_velocity_penalty_scale_choice,
            self.success_tolerance, self.reach_goal_bonus, self.fall_dist, self.fall_penalty,
            self.max_consecutive_successes, self.av_factor, self.ignore_z,
        )

        if self.calculate_cls_metric:
            self._update_class_metrics(successes_raw)

        self.extras['consecutive_successes'] = self.consecutive_successes.mean()
        self.extras['keypoints_closest_distance'] = self.keypoints_closest_distance.mean()
        
        for key in reward_component_dict:
            self.extras[key] = reward_component_dict[key]

        if self.print_success_stat:
            self.total_resets = self.total_resets + self.reset_buf.sum()
            direct_average_successes = self.total_successes + self.successes.sum()
            self.total_successes = self.total_successes + (self.successes * self.reset_buf).sum()

            print("Direct average consecutive successes = {:.1f}".format(direct_average_successes/(self.total_resets + self.num_envs)))
            if self.total_resets > 0:
                print("Post-Reset average consecutive successes = {:.1f}".format(self.total_successes/self.total_resets))

    def _update_class_metrics(self, successes_raw):
        for cls_idx, cls_key in enumerate(self.twist_object_used_index_list):
            cls_mask = (self.object_cls_number_tensor == cls_key)
            finished_cons_successes_cls = torch.sum(successes_raw * self.reset_buf * cls_mask)
            num_resets_cls = torch.sum(self.reset_buf * cls_mask)
            cons_successes_cls = torch.where(
                num_resets_cls > 0,
                finished_cons_successes_cls / num_resets_cls,
                self.cons_successes_cls_list[cls_idx],
            )
            self.cons_successes_cls_list[cls_idx] = cons_successes_cls

        self.cons_successes_cls_list_traj.append(self.cons_successes_cls_list[None].clone())

        print('len(self.cons_successes_cls_list_traj): ', len(self.cons_successes_cls_list_traj))

        self.csc_list.append(successes_raw.clone())
        self.done_list.append(self.reset_buf.clone())
        self.reset_goal_list.append(self.reset_goal_buf.clone())

        if len(self.cons_successes_cls_list_traj) % self.METRIC_SAVE_INTERVAL == 0:
            self._save_class_metrics()

        if len(self.cons_successes_cls_list_traj) > self.METRIC_EXIT_STEPS:
            exit()

    def _save_class_metrics(self):
        np.save(
            f'{self.METRIC_SAVE_DIR}/object_pos_noise_list.npy',
            torch.tensor(self.object_pos_noise_list).cpu().numpy(),
        )

        print('save the cons suc traj!', len(self.cons_successes_cls_list_traj))
        np.save(
            f'{self.METRIC_SAVE_DIR}/cons_successes_cls_list_traj.npy',
            torch.concatenate(self.cons_successes_cls_list_traj).cpu().numpy(),
        )
        np.save(
            f'{self.METRIC_SAVE_DIR}/cons_successes_cls_list_traj_cls_name.npy',
            np.array(self.twist_cls_list),
        )
        np.save(
            f'{self.METRIC_SAVE_DIR}/csc_list.npy',
            torch.concatenate(self.csc_list).cpu().numpy(),
        )
        np.save(
            f'{self.METRIC_SAVE_DIR}/done_list.npy',
            torch.concatenate(self.done_list).cpu().numpy(),
        )
        np.save(
            f'{self.METRIC_SAVE_DIR}/reset_goal_list.npy',
            torch.concatenate(self.reset_goal_list).cpu().numpy(),
        )
        np.save(
            f'{self.METRIC_SAVE_DIR}/object_types.npy',
            np.array(self.object_types),
        )

    def get_random_quat(self, env_ids):
        uvw = torch_rand_float(0, 1.0, (len(env_ids), 3), device=self.device)
        q_w = torch.sqrt(1.0 - uvw[:, 0]) * (torch.sin(2 * np.pi * uvw[:, 1]))
        q_x = torch.sqrt(1.0 - uvw[:, 0]) * (torch.cos(2 * np.pi * uvw[:, 1]))
        q_y = torch.sqrt(uvw[:, 0]) * (torch.sin(2 * np.pi * uvw[:, 2]))
        q_z = torch.sqrt(uvw[:, 0]) * (torch.cos(2 * np.pi * uvw[:, 2]))
        new_rot = torch.cat((q_x.unsqueeze(-1), q_y.unsqueeze(-1), q_z.unsqueeze(-1), q_w.unsqueeze(-1)), dim=-1)

        return new_rot

    def get_random_cube_observation(self, current_cube_pose):
        '''
        This function replaces cube pose in some environments
        with a random cube pose to simulate noisy perception
        estimates in the real world.

        It is also called random cube pose injection.
        '''

        env_ids = np.arange(0, self.num_envs)

        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), 5), device=self.device)

        self.random_cube_poses[:, 0:2] = self.object_init_state[env_ids, 0:2] + \
                                         0.15 * rand_floats[:, 0:2]
        
        self.random_cube_poses[:, 2] = self.object_init_state[env_ids, 2] + \
                                       0.15 * rand_floats[:, 2]

        new_object_rot = self.get_random_quat(env_ids)

        self.random_cube_poses[:, 3:7] = new_object_rot

        random_cube_pose_mask = torch.rand(len(env_ids), 1, device=self.device) < self.random_cube_pose_prob

        current_cube_pose = current_cube_pose * ~random_cube_pose_mask + self.random_cube_poses * random_cube_pose_mask

        return current_cube_pose

    def compute_observations(self): 
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        if self.obs_type == "full_state" or self.asymmetric_obs:
            self.gym.refresh_force_sensor_tensor(self.sim)
            self.gym.refresh_dof_force_tensor(self.sim)
            self.gym.refresh_net_contact_force_tensor(self.sim)

        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

        self.goal_pose = self.goal_states[:, 0:7]
        self.goal_pos = self.goal_states[:, 0:3]
        self.goal_rot = self.goal_states[:, 3:7]

        if self.enable_random_obs:
            self.object_pose = self.get_random_cube_observation(self.object_pose)

        if self.obs_type == "full_no_vel":
            self.compute_full_observations(True)
        elif self.obs_type == "full":
            self.compute_full_observations()
        elif self.obs_type == "full_state":
            self.compute_full_state()
        else:
            print("Unknown observations type!")

        if self.asymmetric_obs:
            self.compute_full_state(True)

        self._sanitize_observation_buffers()

    def _sanitize_observation_buffers(self):
        obs_nan_mask = torch.isnan(self.obs_buf)
        if obs_nan_mask.any():
            self.obs_nan_reset_index = obs_nan_mask.any(1)
            print(torch.where(self.obs_nan_reset_index))

            if (self.control_steps // 8)%1000==0:
                print('obs_nan_reset_index: ', self.obs_nan_reset_index)

            self.obs_buf = torch.where(obs_nan_mask, torch.zeros_like(self.obs_buf), self.obs_buf)

        obs_inf_mask = torch.isinf(self.obs_buf)
        if obs_inf_mask.any():
            print("Inf found in obs_buf, replacing with zeros")
            self.obs_buf = torch.where(obs_inf_mask, torch.zeros_like(self.obs_buf), self.obs_buf)

        states_nan_mask = torch.isnan(self.states_buf)
        if states_nan_mask.any():
            self.obs_nan_reset_index = states_nan_mask.any(1)

            if (self.control_steps // 8)%1000==0:
                print('obs_nan_reset_index: ', self.obs_nan_reset_index)

            self.states_buf = torch.where(states_nan_mask, torch.zeros_like(self.states_buf), self.states_buf)

        states_inf_mask = torch.isinf(self.states_buf)
        if states_inf_mask.any():
            print("Inf found in states_buf, replacing with zeros")
            self.states_buf = torch.where(states_inf_mask, torch.zeros_like(self.states_buf), self.states_buf)

    def compute_full_observations(self, no_vel=False):
        if no_vel:
            self.obs_buf[:, 0:10] = unscale(self.shadow_hand_dof_pos,
                                                                   self.shadow_hand_dof_lower_limits, self.shadow_hand_dof_upper_limits)
            self.obs_buf[:, 10:20] = self.actions
            
            self.obs_buf[:, 20:29] = self.vec_sensor_tensor[:, self.FINGERTIP_FORCE_MASK]

            self.keypoints_fingertip_raw = self.rigid_body_states[:, self.hand_cfg['finger_tips_idx'], :7]
            self.keypoints_valve_raw = self.rigid_body_states[:, [13], :7]

            
            finger_kp_i = (self.keypoints_fingertip_raw[:, :, :3] +\
                quat_apply(self.keypoints_fingertip_raw[:, :, 3:], self.fingertip_kp_offset_tensor))

            self.fingertip_kp_list = finger_kp_i
            self.valve_kp_list = self.valve_kp_bias_tensor+self.keypoints_valve_raw[:, :, :3]
            self.keypoints_closest_distance = torch.norm(self.fingertip_kp_list - self.valve_kp_list, p=2, dim=-1)

            self.obs_buf[:, 29:38] = self.keypoints_fingertip_raw[:, :, :3].reshape((self.obs_buf.shape[0], -1))
            self.obs_buf[:, 172:181] = self.valve_kp_list[:, :, :3].reshape((self.obs_buf.shape[0], -1))

            self.obs_buf[:, 39] = to_torch(self.object_cls_number_list, 
                                            dtype=torch.float32,
                                            device=self.device)/100
            
            self.obs_buf[:, 40:168] = self.points_feature_tensor_list.clone()

            self.obs_buf[:, 168:171] = self.valve_dof_state[:, :, 0]
            
            self.obs_buf[:, 171:172] = self.rotation_direction_tensor_list

        else:
            self.obs_buf[:, 0:self.num_shadow_hand_dofs] = unscale(self.shadow_hand_dof_pos,
                                                                   self.shadow_hand_dof_lower_limits, self.shadow_hand_dof_upper_limits)
            self.obs_buf[:, self.num_shadow_hand_dofs:2*self.num_shadow_hand_dofs] = self.vel_obs_scale * self.shadow_hand_dof_vel

            self.obs_buf[:, 32:39] = self.object_pose
            self.obs_buf[:, 39:42] = self.object_linvel
            self.obs_buf[:, 42:45] = self.vel_obs_scale * self.object_angvel

            self.obs_buf[:, 45:52] = self.goal_pose
            self.obs_buf[:, 52:56] = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))

            self.obs_buf[:, 56:72] = self.actions

    def compute_full_state(self, asymm_obs=False):
        if asymm_obs:
            self.states_buf[:, 0:200] = self.obs_buf * 1.0 
            
            self.states_buf[:, 200:210] = self.vel_obs_scale * self.shadow_hand_dof_vel
            
            self.states_buf[:, 210:223] = self.force_torque_obs_scale * self.dof_force_tensor
            
            self.states_buf[:, 223:244] = self.keypoints_fingertip_raw.reshape((-1, 21))
            
            if self.asym_state_dr_params:
                self.states_buf[:, 250:340] = self.sim_params_buf
            
            self.states_buf[:, 340:343] = self.root_state_tensor[self.object_indices, 0:3]
            
            self.states_buf[:, 343] = self.act_moving_average


    def reset_target_pose(self, env_ids, apply_reset=False):
        object_indices = self.object_indices[env_ids].to(torch.int32)
        self.valve_dof_state[env_ids, :, :] *= 0

        if apply_reset:
            self.gym.set_dof_state_tensor_indexed(self.sim,
                                                gymtorch.unwrap_tensor(self.dof_state),
                                                gymtorch.unwrap_tensor(object_indices), 
                                                len(object_indices)
                                                )
    
        self.reset_goal_buf[env_ids] = 0
        

    def reset_idx(self, env_ids, goal_env_ids):

        if self.randomize:
            self.apply_randomizations(self.randomization_params)
            if self.asym_state_dr_params:
                self.update_sim_para()

        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_shadow_hand_dofs * 2 + 5), device=self.device)

        self.reset_target_pose(env_ids)

        self.rb_forces[env_ids, :, :] = 0.0

        self.root_state_tensor[self.object_indices[env_ids]] = self.object_init_state[env_ids].clone()
        self.root_state_tensor[self.object_indices[env_ids], 0:2] = self.object_init_state[env_ids, 0:2] + \
            self.reset_position_noise * rand_floats[:, 0:2]
        self.root_state_tensor[self.object_indices[env_ids], self.up_axis_idx] = self.object_init_state[env_ids, self.up_axis_idx] + \
            self.reset_position_noise * rand_floats[:, self.up_axis_idx]

        object_indices = torch.unique(torch.cat([self.object_indices[env_ids]]).to(torch.int32))
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(object_indices), len(object_indices))

        self.random_force_prob[env_ids] = torch.exp((torch.log(self.force_prob_range[0]) - torch.log(self.force_prob_range[1]))
                                                    * torch.rand(len(env_ids), device=self.device) + torch.log(self.force_prob_range[1]))

        delta_max = self.shadow_hand_dof_upper_limits - self.shadow_hand_dof_default_pos
        delta_min = self.shadow_hand_dof_lower_limits - self.shadow_hand_dof_default_pos
        rand_delta = delta_min + (delta_max - delta_min) * 0.3 * (rand_floats[:, 5:5+self.num_shadow_hand_dofs] + 1)

        pos = self.shadow_hand_default_dof_pos + self.reset_dof_pos_noise * rand_delta
        self.shadow_hand_dof_pos[env_ids, :] = pos
        self.shadow_hand_dof_vel[env_ids, :] = self.shadow_hand_dof_default_vel + \
            self.reset_dof_vel_noise * rand_floats[:, 5+self.num_shadow_hand_dofs:5+self.num_shadow_hand_dofs*2]
        self.prev_targets[env_ids, :self.num_shadow_hand_dofs] = pos
        self.cur_targets[env_ids, :self.num_shadow_hand_dofs] = pos
        
        self.valve_dof_state[env_ids, :, :] *= 0

        dof_reset_indices = torch.unique(torch.cat([self.hand_indices[env_ids],
                                                    self.object_indices[env_ids],
                                                    self.object_indices[goal_env_ids]
                                                 ]).to(torch.int32))

        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.prev_targets),
                                                        gymtorch.unwrap_tensor(dof_reset_indices), len(dof_reset_indices))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(dof_reset_indices), len(dof_reset_indices))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0


    def pre_physics_step(self, actions):
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(goal_env_ids) > 0 and len(env_ids) == 0:
            self.reset_target_pose(goal_env_ids, apply_reset=True)

        elif len(goal_env_ids) > 0:
            self.reset_target_pose(goal_env_ids)

        if len(env_ids) > 0:
            self.reset_idx(env_ids, goal_env_ids)

        num_dof = len(self.shadow_hand_dof_lower_limits)
        actions = actions[:, :num_dof]

        self.actions = actions.clone().to(self.device)

        action_apply = self.get_random_network_adversary_action(actions)

        if self.use_relative_control:
            targets = self.prev_targets[:, self.actuated_dof_indices] + self.shadow_hand_dof_speed_scale * self.dt * action_apply
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(targets,
                                                                          self.shadow_hand_dof_lower_limits[self.actuated_dof_indices],
                                                                          self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
        else:
            self.cur_targets[:, self.actuated_dof_indices] = scale(action_apply,
                                                                   self.shadow_hand_dof_lower_limits[self.actuated_dof_indices],
                                                                   self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])

            self.cur_targets[:, self.actuated_dof_indices] = self.act_moving_average * self.cur_targets[:,
                                                                                                        self.actuated_dof_indices] + (1.0 - self.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(self.cur_targets[:, self.actuated_dof_indices],
                                                                          self.shadow_hand_dof_lower_limits[self.actuated_dof_indices],
                                                                          self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])

        self.dof_delta = self.cur_targets[:, self.actuated_dof_indices] - self.prev_targets[:, self.actuated_dof_indices]

        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))

        if self.force_scale > 0.0:
            self.rb_forces *= torch.pow(self.force_decay, self.dt / self.force_decay_interval)

            force_indices = (torch.rand(self.num_envs, device=self.device) < self.random_force_prob).nonzero()

            self.rb_forces[force_indices, self.object_rb_handles, :] = torch.randn(
                self.rb_forces[force_indices, self.object_rb_handles, :].shape, device=self.device) * self.object_rb_masses[0] * self.force_scale
            
            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.rb_forces), None, gymapi.LOCAL_SPACE)


    def update_action_moving_average(self):
        act_moving_average_ptp = self.act_moving_average_upper-self.act_moving_average_lower
        self.act_moving_average = max( self.act_moving_average_upper-self.ap_scale*act_moving_average_ptp,
                                       self.act_moving_average_lower )

    def update_velocity_penalty_factor(self):
        self.action_delta_penalty_scale_choice = self.action_delta_penalty_scale*self.ap_scale
        self.joint_velocity_penalty_scale_choice = self.joint_velocity_penalty_scale*self.ap_scale

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        self.compute_observations()
        self.compute_reward(self.actions)

        if self.num_envs!=1 and not self.calculate_cls_metric:
            if self.update_smoothing_factor:
                cur_episode = self.control_steps // 8

                if cur_episode%self.ap_update_step==0 and self.control_steps%8==0 and cur_episode!=0 and self.consecutive_successes>self.consecutive_successes_threshold:
                    print(f'update the parameter, the consecutive_successes is {self.consecutive_successes}')
                    self.ap_scale = min(1.0, self.ap_scale+1/self.ap_num_iter)
                    self.update_action_moving_average()
                    self.update_velocity_penalty_factor()
        else:
            self.act_moving_average = 0.5
            
        self.extras['auto_para/act_moving_average'] = self.act_moving_average
        self.extras['auto_para/action_delta_penalty_scale_choice'] = self.action_delta_penalty_scale_choice
        self.extras['auto_para/joint_velocity_penalty_scale_choice'] = self.joint_velocity_penalty_scale_choice
        self.extras['auto_para/ap_scale'] = self.ap_scale

        self.extras['auto_para/last_step'] = self.last_step

        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]
        self.prev_object_rot = self.object_rot


    def update_sim_para(self, first_randomization=False):
        for env_ids in self.dr_env_ids:
            env_cur = self.envs[env_ids]
            obj_handle = self.object_handles[env_ids]
   
            obj_rigid_body_prop = self.gym.get_actor_rigid_body_properties(env_cur, obj_handle)
            obj_rigid_shape_prop = self.gym.get_actor_rigid_shape_properties(env_cur, obj_handle)

            obj_scale = self.gym.get_actor_scale(env_cur, obj_handle)

            obj_mass = obj_rigid_body_prop[4].mass

            obj_com = torch.tensor([obj_rigid_body_prop[4].com.x,
                                obj_rigid_body_prop[4].com.y,
                                obj_rigid_body_prop[4].com.z],
                                dtype=torch.float, device=self.device)

            obj_friction = torch.tensor([obj_rigid_shape_prop[j].friction for j in range(len(obj_rigid_shape_prop))],
                                        device='cuda:0').mean()
            obj_restitution = torch.tensor([obj_rigid_shape_prop[j].restitution for j in range(len(obj_rigid_shape_prop))],
                                           device='cuda:0').mean()

            if first_randomization:
                obj_pose = torch.zeros((3, ), dtype=torch.float, device=self.device)
            else:
                obj_pose = self.object_pos[env_ids].clone()


            self.sim_params_buf[env_ids, 0:3] = obj_pose
            self.sim_params_buf[env_ids, 3] = obj_scale
            self.sim_params_buf[env_ids, 4] = obj_mass

            self.sim_params_buf[env_ids, 8:11] = obj_com
            self.sim_params_buf[env_ids, 5] = obj_friction
            self.sim_params_buf[env_ids, 11] = obj_restitution

            hand_handle = self.allegro_hands[env_ids]

            hand_rigid_body_prop = self.gym.get_actor_rigid_body_properties(env_cur, hand_handle)
            hand_rigid_shape_prop = self.gym.get_actor_rigid_shape_properties(env_cur, hand_handle)

            hand_scale = self.gym.get_actor_scale(env_cur, hand_handle)

            hand_mass = torch.tensor([hand_rigid_body_prop[i].mass for i in range(len(hand_rigid_body_prop))], device=self.device)


            hand_com = torch.tensor([[hand_rigid_body_prop[i].com.x,
                                hand_rigid_body_prop[i].com.y,
                                hand_rigid_body_prop[i].com.z] for i in range(len(hand_rigid_body_prop))],
                                dtype=torch.float, device=self.device)

            hand_friction = torch.tensor([hand_rigid_shape_prop[j].friction for j in range(len(hand_rigid_body_prop))],
                                        device='cuda:0').mean()
            hand_restitution = torch.tensor([hand_rigid_shape_prop[j].restitution for j in range(len(hand_rigid_body_prop))],
                                           device='cuda:0').mean()


            
            self.sim_params_buf[env_ids, 15] = hand_scale
            self.sim_params_buf[env_ids, 16:27] = hand_mass
            self.sim_params_buf[env_ids, 41:74] = hand_com.flatten()
            self.sim_params_buf[env_ids, 27] = hand_friction 
            self.sim_params_buf[env_ids, 74] = hand_restitution

@torch.jit.script
def compute_hand_reward(
    obs_nan_reset_index, rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, consecutive_successes,
    max_episode_length: float, valve_dof_state, object_pos, object_rot, target_pos, target_rot,
    hand_cur_targets, hand_prev_targets, shadow_hand_dof_vel, keypoints_closest_distance, 
    dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
    actions, action_penalty_scale: float, action_delta_penalty_scale: float, joint_velocity_penalty_scale: float,
    success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
    fall_penalty: float, max_consecutive_successes: int, av_factor: float, ignore_z_rot: bool, 
):
    
    # Distance from the hand to the object
    goal_dist = torch.norm(object_pos - target_pos, p=2, dim=-1)

    if ignore_z_rot:
        success_tolerance = 2.0 * success_tolerance

    valve_pos = valve_dof_state[:, 0, 0]

    action_penalty = torch.sum(actions ** 2, dim=-1)

    action_detal_free = 0.2
    dof_vel_free = 1

    action_delta = hand_cur_targets-hand_prev_targets
    action_delta[torch.abs(action_delta)<action_detal_free] = 0

    shadow_hand_dof_vel[torch.abs(shadow_hand_dof_vel)<dof_vel_free] = 0

    action_delta_penalty = torch.sum((action_delta) ** 2, dim=-1)
    joint_velocity_penalty = torch.sum((shadow_hand_dof_vel) ** 2, dim=-1)

    dist_rew = 0
    
    rot_rew = 1.0*valve_pos

    keypoints_reward_scale = -50
    keypoints_closest_distance[keypoints_closest_distance<0.03] = 0
    keypoints_reward = keypoints_reward_scale*(keypoints_closest_distance.mean(dim=1))
    
    obj_fall = (valve_dof_state[:, 1:, 0].abs()>0.3).any(dim=1)
    fall_reward = torch.clamp(0.1/valve_dof_state[:, 1:, 0].abs().sum(dim=1), max=2.0)
    
    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    reward = rot_rew + action_penalty * action_penalty_scale + action_delta_penalty*action_delta_penalty_scale\
             + joint_velocity_penalty*joint_velocity_penalty_scale + keypoints_reward + fall_reward
    
    reward_component_dict = {
        "reward_component/dist_rew": dist_rew,
        "reward_component/rot_rew": (rot_rew).mean(),
        "reward_component/action_penalty": (action_penalty_scale * action_penalty).mean(),
        "reward_component/action_delta_penalty": (action_delta_penalty_scale*action_delta_penalty).mean(),
        "reward_component/joint_velocity_penalty": (joint_velocity_penalty_scale*joint_velocity_penalty).mean(),
        "reward_component/keypoints_reward": (keypoints_reward).mean(),
        "reward_component/fall_reward": (fall_reward).mean()
    }
    
    if reward.max().item()>15000:
        print('print abnormal reward: ', reward.max().item(), dist_rew, rot_rew.max().item(), action_penalty.max().item(),
              action_delta_penalty.max().item(), joint_velocity_penalty.max().item(), keypoints_reward.max().item())

    # Find out which envs hit the goal and update successes count
    goal_resets = torch.where(valve_pos>4, torch.ones_like(reset_goal_buf), reset_goal_buf)
    successes = successes + goal_resets

    # Success bonus
    reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)

    resets = torch.zeros_like(reset_buf)

    timed_out = progress_buf >= (max_episode_length - 1)
    
    resets = torch.where(timed_out, torch.ones_like(resets), resets)

    resets = torch.where(obj_fall, torch.ones_like(resets), resets)

    # Apply penalty for not reaching the goal
    if max_consecutive_successes > 0:
        reward = torch.where(timed_out, reward + 0.5 * fall_penalty, reward)

    if obs_nan_reset_index.any():
        print("Nan detected in obs, triggering reset")
        resets = torch.where(obs_nan_reset_index, torch.ones_like(resets), resets)
        reward = torch.where(torch.isnan(reward), torch.zeros_like(reward), reward)

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())
    cons_successes = torch.where(num_resets > 0,
                                 av_factor*finished_cons_successes/num_resets + (1.0 - av_factor)*consecutive_successes,
                                 consecutive_successes)

    return reward, resets, goal_resets, progress_buf, successes, cons_successes, reward_component_dict, successes
