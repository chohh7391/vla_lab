# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch

import carb
import isaacsim.core.utils.torch as torch_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat
from isaaclab.sensors import TiledCamera

from vla_lab.tasks.direct.base_line.factory import factory_control, factory_utils
from vla_lab.tasks.direct.base_line.factory.factory_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG

from vla_lab.tasks.direct.vla.gr00t.factory.factory_gr00t_env_cfg import FactoryGr00tEnvCfg
from vla_lab.tasks.direct.vla.gr00t.factory.base.factory_not_parallel_env import FactoryEnv
from vla_lab.tasks.direct.base_line.forge import forge_utils


class FactoryGr00tNotParallelEnv(FactoryEnv):
    cfg: FactoryGr00tEnvCfg

    def __init__(self, cfg: FactoryGr00tEnvCfg, render_mode: str | None = None, **kwargs):
        # Update number of obs/states
        super().__init__(cfg, render_mode, **kwargs)

        # Force sensor information.
        self.force_sensor_body_idx = self._robot.body_names.index("force_sensor")
        self.force_sensor_smooth = torch.zeros((self.num_envs, 6), device=self.device)
        self.force_sensor_world_smooth = torch.zeros((self.num_envs, 6), device=self.device)

    def _setup_scene(self):
        """Initialize simulation scene."""
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -1.05))

        # spawn a usd file of a table into the scene
        cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
        cfg.func(
            "/World/envs/env_.*/Table", cfg, translation=(0.55, 0.0, 0.0), orientation=(0.70711, 0.0, 0.0, 0.70711)
        )

        self._robot = Articulation(self.cfg.robot)
        self._fixed_asset = Articulation(self.cfg_task.fixed_asset)
        self._held_asset = Articulation(self.cfg_task.held_asset)
        if self.cfg_task.name == "gear_mesh":
            self._small_gear_asset = Articulation(self.cfg_task.small_gear_cfg)
            self._large_gear_asset = Articulation(self.cfg_task.large_gear_cfg)

        self._wrist_camera = TiledCamera(self.cfg.wrist_camera)
        self._left_camera = TiledCamera(self.cfg.left_camera)
        self._right_camera = TiledCamera(self.cfg.right_camera)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            # we need to explicitly filter collisions for CPU simulation
            self.scene.filter_collisions()

        self.scene.articulations["robot"] = self._robot
        self.scene.articulations["fixed_asset"] = self._fixed_asset
        self.scene.articulations["held_asset"] = self._held_asset
        if self.cfg_task.name == "gear_mesh":
            self.scene.articulations["small_gear"] = self._small_gear_asset
            self.scene.articulations["large_gear"] = self._large_gear_asset

        self.scene.sensors["wrist_camera"] = self._wrist_camera
        self.scene.sensors["left_camera"] = self._left_camera
        self.scene.sensors["right_camera"] = self._right_camera

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _get_observations(self):
        """Get actor/critic inputs using asymmetric critic."""
        obs_dict, state_dict = self._get_factory_obs_state_dict()

        obs_dict.update({
            "ft_force": self.get_ft_force()[0], # added
        })

        state_dict.update({
            "ft_force": self.get_ft_force()[1], # added
        })

        obs_tensors = factory_utils.collapse_obs_dict(obs_dict, self.cfg.obs_order + ["prev_actions"])
        state_tensors = factory_utils.collapse_obs_dict(state_dict, self.cfg.state_order + ["prev_actions"])
        return {"policy": obs_tensors, "critic": state_tensors}

    def _get_rewards(self):
        """Update rewards and compute success statistics."""
        # Get successful and failed envs at current timestep
        check_rot = self.cfg_task.name == "nut_thread"
        curr_successes = self._get_curr_successes(
            success_threshold=self.cfg_task.success_threshold, check_rot=check_rot
        )

        rew_buf = torch.where(curr_successes, 1.0, 0.0)

        force = self.get_ft_force()[1]
        force_mag = torch.norm(force, dim=1)
        force_penalty = torch.where(force_mag > 0.2, -0.1, 0.0)

        rew_buf = rew_buf + force_penalty

        return rew_buf
    
    def get_ft_force(self):

        self.force_sensor_world = self._robot.root_physx_view.get_link_incoming_joint_force()[
            :, self._robot.body_names.index("force_sensor")
        ]
        alpha = 0.25
        self.force_sensor_world_smooth = alpha * self.force_sensor_world + (1 - alpha) * self.force_sensor_world_smooth

        self.force_sensor_smooth = torch.zeros_like(self.force_sensor_world)
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.force_sensor_smooth[:, :3], self.force_sensor_smooth[:, 3:6] = forge_utils.change_FT_frame(
            self.force_sensor_world_smooth[:, 0:3],
            self.force_sensor_world_smooth[:, 3:6],
            (identity_quat, torch.zeros((self.num_envs, 3), device=self.device)),
            (identity_quat, self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise),
        )

        # Compute noisy force values.
        force_noise = torch.randn((self.num_envs, 3), dtype=torch.float32, device=self.device)
        force_noise *= 1.0
        self.noisy_force = self.force_sensor_smooth[:, 0:3] + force_noise

        return self.noisy_force, self.force_sensor_smooth[:, 0:3]
    
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