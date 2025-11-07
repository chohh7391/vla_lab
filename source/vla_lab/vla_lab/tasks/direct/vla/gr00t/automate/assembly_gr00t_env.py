# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch

import isaacsim.core.utils.torch as torch_utils
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sensors import TiledCamera

from vla_lab.tasks.direct.base_line.automate import industreal_algo_utils as industreal_algo
from vla_lab.tasks.direct.base_line.automate import automate_algo_utils as automate_algo
from vla_lab.tasks.direct.base_line.factory import factory_utils
from vla_lab.tasks.direct.base_line.forge import forge_utils

from vla_lab.tasks.direct.vla.gr00t.automate.assembly_gr00t_env_cfg import AssemblyEnvCfg
from vla_lab.tasks.direct.base_line.automate.soft_dtw_cuda import SoftDTW
from vla_lab.tasks.direct.vla.gr00t.automate.base.assembly_env import AssemblyEnv
from vla_lab.tasks.direct.base_line.automate.assembly_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG
from vla_lab.envs import DirectRLGr00tEnv

class AssemblyGr00tEnv(AssemblyEnv):
    cfg: AssemblyEnvCfg

    def __init__(self, cfg: AssemblyEnvCfg, render_mode: str | None = None, **kwargs):

        # Update number of obs/states
        cfg.observation_space = sum([OBS_DIM_CFG[obs] for obs in cfg.obs_order])
        cfg.state_space = sum([STATE_DIM_CFG[state] for state in cfg.state_order])
        cfg.observation_space += cfg.action_space
        cfg.state_space += cfg.action_space
        self.cfg_task = cfg.tasks[cfg.task_name]

        DirectRLGr00tEnv.__init__(self, cfg, render_mode, **kwargs)

        self._set_body_inertias()
        self._init_tensors()
        self._set_default_dynamics_parameters()
        self._compute_intermediate_values(dt=self.physics_dt)

        # Load asset meshes in warp for SDF-based dense reward
        wp.init()
        self.wp_device = wp.get_preferred_device()
        self.plug_mesh, self.plug_sample_points, self.socket_mesh = industreal_algo.load_asset_mesh_in_warp(
            self.cfg_task.assembly_dir + self.cfg_task.held_asset_cfg.obj_path,
            self.cfg_task.assembly_dir + self.cfg_task.fixed_asset_cfg.obj_path,
            self.cfg_task.num_mesh_sample_points,
            self.wp_device,
        )

        # Get the gripper open width based on plug object bounding box
        self.gripper_open_width = automate_algo.get_gripper_open_width(
            self.cfg_task.assembly_dir + self.cfg_task.held_asset_cfg.obj_path
        )

        # Create criterion for dynamic time warping (later used for imitation reward)
        self.soft_dtw_criterion = SoftDTW(use_cuda=True, gamma=self.cfg_task.soft_dtw_gamma)

        # Evaluate
        if self.cfg_task.if_logging_eval:
            self._init_eval_logging()

        if self.cfg_task.sample_from != "rand":
            self._init_eval_loading()

        # Force sensor information.
        self.force_sensor_body_idx = self._robot.body_names.index("force_sensor")
        self.force_sensor_smooth = torch.zeros((self.num_envs, 6), device=self.device)
        self.force_sensor_world_smooth = torch.zeros((self.num_envs, 6), device=self.device)


    def _setup_scene(self):
        """Initialize simulation scene."""
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -0.4))

        # spawn a usd file of a table into the scene
        cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
        cfg.func(
            "/World/envs/env_.*/Table", cfg, translation=(0.55, 0.0, 0.0), orientation=(0.70711, 0.0, 0.0, 0.70711)
        )

        self._robot = Articulation(self.cfg.robot)
        self._fixed_asset = Articulation(self.cfg_task.fixed_asset)
        self._held_asset = RigidObject(self.cfg_task.held_asset)

        self._wrist_camera = TiledCamera(self.cfg.wrist_camera)
        self._left_camera = TiledCamera(self.cfg.left_camera)
        self._right_camera = TiledCamera(self.cfg.right_camera)

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions()

        self.scene.articulations["robot"] = self._robot
        self.scene.articulations["fixed_asset"] = self._fixed_asset
        self.scene.rigid_objects["held_asset"] = self._held_asset

        self.scene.sensors["wrist_camera"] = self._wrist_camera
        self.scene.sensors["left_camera"] = self._left_camera
        self.scene.sensors["right_camera"] = self._right_camera

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _reset_idx(self, env_ids):
        """We assume all envs will always be reset at the same time."""
        super()._reset_idx(env_ids)

        contact_rand = torch.rand((self.num_envs,), dtype=torch.float32, device=self.device)
        contact_lower, contact_upper = [5.0, 10.0]
        self.contact_penalty_thresholds = contact_lower + contact_rand * (contact_upper - contact_lower)

    def _apply_action(self):
        super()._apply_action()

        # Step (0): Scale actions to allowed range.
        pos_actions = self.actions[:, 0:3]
        pos_actions = pos_actions @ torch.diag(torch.tensor(self.cfg.ctrl.pos_action_bounds, device=self.device))

        rot_actions = self.actions[:, 3:6]
        rot_actions = rot_actions @ torch.diag(torch.tensor(self.cfg.ctrl.rot_action_bounds, device=self.device))

        # Step (1): Compute desired pose targets in EE frame.
        # (1.a) Position. Action frame is assumed to be the top of the bolt (noisy estimate).
        fixed_pos_action_frame = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
        ctrl_target_fingertip_preclipped_pos = fixed_pos_action_frame + pos_actions
        # (1.b) Enforce rotation action constraints.
        rot_actions[:, 0:2] = 0.0

        # Assumes joint limit is in (+x, -y)-quadrant of world frame.
        rot_actions[:, 2] = np.deg2rad(-180.0) + np.deg2rad(270.0) * (rot_actions[:, 2] + 1.0) / 2.0  # Joint limit.
        # (1.c) Get desired orientation target.
        bolt_frame_quat = torch_utils.quat_from_euler_xyz(
            roll=rot_actions[:, 0], pitch=rot_actions[:, 1], yaw=rot_actions[:, 2]
        )

        rot_180_euler = torch.tensor([np.pi, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        quat_bolt_to_ee = torch_utils.quat_from_euler_xyz(
            roll=rot_180_euler[:, 0], pitch=rot_180_euler[:, 1], yaw=rot_180_euler[:, 2]
        )

        ctrl_target_fingertip_preclipped_quat = torch_utils.quat_mul(quat_bolt_to_ee, bolt_frame_quat)

        # Step (2): Clip targets if they are too far from current EE pose.
        # (2.a): Clip position targets.
        self.delta_pos = ctrl_target_fingertip_preclipped_pos - self.fingertip_midpoint_pos  # Used for action_penalty.
        pos_error_clipped = torch.clip(self.delta_pos, -self.pos_threshold, self.pos_threshold)
        ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_error_clipped

        # (2.b) Clip orientation targets. Use Euler angles. We assume we are near upright, so
        # clipping yaw will effectively cause slow motions. When we clip, we also need to make
        # sure we avoid the joint limit.

        # (2.b.i) Get current and desired Euler angles.
        curr_roll, curr_pitch, curr_yaw = torch_utils.get_euler_xyz(self.fingertip_midpoint_quat)
        desired_roll, desired_pitch, desired_yaw = torch_utils.get_euler_xyz(ctrl_target_fingertip_preclipped_quat)

        # (2.b.ii) Correct the direction of motion to avoid joint limit.
        # Map yaws between [-125, 235] degrees (so that angles appear on a continuous span uninterrupted by the joint limit).
        curr_yaw = factory_utils.wrap_yaw(curr_yaw)
        desired_yaw = factory_utils.wrap_yaw(desired_yaw)

        # (2.b.iii) Clip motion in the correct direction.
        self.delta_yaw = desired_yaw - curr_yaw  # Used later for action_penalty.


    def _get_observations(self):
        """Get actor/critic inputs using asymmetric critic."""

        prev_actions = self.actions.clone()
        prev_actions[:, 3:5] = 0.0

        noisy_force, force_sensor_smooth = self.get_ft_force()

        obs_dict = {
            "joint_pos": self.joint_pos[:, 0:7],
            "fingertip_pos": self.fingertip_midpoint_pos,
            "fingertip_quat": self.fingertip_midpoint_quat,
            "fingertip_goal_pos": self.gripper_goal_pos,
            "fingertip_goal_quat": self.gripper_goal_quat,
            "delta_pos": self.gripper_goal_pos - self.fingertip_midpoint_pos,
            # added
            "force_threshold": self.contact_penalty_thresholds[:, None],
            "ft_force": noisy_force,
            "prev_actions": prev_actions,
        }

        state_dict = {
            "joint_pos": self.joint_pos[:, 0:7],
            "joint_vel": self.joint_vel[:, 0:7],
            "fingertip_pos": self.fingertip_midpoint_pos,
            "fingertip_quat": self.fingertip_midpoint_quat,
            "ee_linvel": self.fingertip_midpoint_linvel,
            "ee_angvel": self.fingertip_midpoint_angvel,
            "fingertip_goal_pos": self.gripper_goal_pos,
            "fingertip_goal_quat": self.gripper_goal_quat,
            "held_pos": self.held_pos,
            "held_quat": self.held_quat,
            "delta_pos": self.gripper_goal_pos - self.fingertip_midpoint_pos,
            # added
            "ft_force": force_sensor_smooth[:, 0:3],
            "force_threshold": self.contact_penalty_thresholds[:, None],
            "prev_actions": prev_actions,
        }
        obs_tensors = [obs_dict[obs_name] for obs_name in self.cfg.obs_order + ['prev_actions']]
        obs_tensors = torch.cat(obs_tensors, dim=-1)
        
        state_tensors = [state_dict[state_name] for state_name in self.cfg.state_order + ['prev_actions']]
        state_tensors = torch.cat(state_tensors, dim=-1)

        return {"policy": obs_tensors, "critic": state_tensors}
    
    def get_ft_force(self):

        self.force_sensor_world = self._robot.root_physx_view.get_link_incoming_joint_force()[
            :, self._robot.body_names.index("force_sensor")
        ]
        alpha = 0.25
        self.force_sensor_world_smooth = alpha * self.force_sensor_world + (1 - alpha) * self.force_sensor_world_smooth

        force_sensor_smooth = torch.zeros_like(self.force_sensor_world)
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        force_sensor_smooth[:, :3], force_sensor_smooth[:, 3:6] = forge_utils.change_FT_frame(
            self.force_sensor_world_smooth[:, 0:3],
            self.force_sensor_world_smooth[:, 3:6],
            (identity_quat, torch.zeros((self.num_envs, 3), device=self.device)),
            (identity_quat, self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise),
        )

        # Compute noisy force values.
        force_noise = torch.randn((self.num_envs, 3), dtype=torch.float32, device=self.device)
        force_noise *= 1.0
        noisy_force = force_sensor_smooth[:, 0:3] + force_noise

        return noisy_force, force_sensor_smooth[:, 0:3]
    
    def get_sparse_rewards(self) -> torch.Tensor:
        
        rew_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        rew_dict, rew_scales = {}, {}

        # Calculate action penalty for the asset-relative action space.
        pos_error = torch.norm(self.delta_pos, p=2, dim=-1) / self.cfg.ctrl.pos_action_threshold[0]
        rot_error = torch.abs(self.delta_yaw) / self.cfg.ctrl.rot_action_threshold[0]
        # Contact penalty.
        noisy_force, force_sensor_smooth = self.get_ft_force()
        contact_force = torch.norm(force_sensor_smooth[:, 0:3], p=2, dim=-1, keepdim=False)
        contact_penalty = torch.nn.functional.relu(contact_force - self.contact_penalty_thresholds)
        # Add success prediction rewards.
        curr_successes = automate_algo.check_plug_inserted_in_socket(
            self.held_pos,
            self.fixed_pos,
            self.disassembly_dists,
            self.keypoints_held,
            self.keypoints_fixed,
            self.cfg_task.close_error_thresh,
            self.episode_length_buf,
        )

        rew_dict = {
            "curr_success": curr_successes.float(),
            "action_penalty_asset": pos_error + rot_error,
            "contact_penalty": contact_penalty,
        }
        rew_scales = {
            "curr_success": 1.0,
            "action_penalty_asset": -self.cfg_task.action_penalty_asset_scale,
            "contact_penalty": -self.cfg_task.contact_penalty_scale,
        }

        for rew_name in rew_dict.keys():
            rew = rew_dict[rew_name] * rew_scales[rew_name]
            rew_buf += rew

        return rew_buf


    def _get_gr00t_observations(self):
        # This is for gr00t observations

        observations = {
            "video.left_view": np.expand_dims(self._left_camera.data.output["rgb"].cpu().numpy().astype(np.uint8), axis=1),
            "video.right_view": np.expand_dims(self._right_camera.data.output["rgb"].cpu().numpy().astype(np.uint8), axis=1),
            "video.wrist_view": np.expand_dims(self._wrist_camera.data.output["rgb"].cpu().numpy().astype(np.uint8), axis=1),
            "state.eef_position": np.expand_dims(self.fingertip_midpoint_pos.cpu().numpy().astype(np.float64), axis=1),
            "state.eef_quaternion": np.expand_dims(self.fingertip_midpoint_quat.cpu().numpy().astype(np.float64), axis=1),
            "state.gripper_qpos": np.expand_dims(self.joint_pos[:, 7:9].cpu().numpy().astype(np.float64), axis=1),
        }
        return observations