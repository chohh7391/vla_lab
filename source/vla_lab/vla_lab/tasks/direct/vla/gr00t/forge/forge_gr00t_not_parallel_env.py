# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import torch

from vla_lab.tasks.direct.base_line.factory import factory_utils
from vla_lab.tasks.direct.vla.gr00t.forge.base.forge_not_parallel_env import ForgeEnv
from vla_lab.tasks.direct.vla.gr00t.forge.forge_gr00t_env_cfg import ForgeGr00tEnvCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab.sensors import TiledCamera
import wandb
import time



class ForgeGr00tNotParallelEnv(ForgeEnv):
    cfg: ForgeGr00tEnvCfg

    def __init__(self, cfg: ForgeGr00tEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize additional randomization and logging tensors."""
        super().__init__(cfg, render_mode, **kwargs)

        if wandb.run is None:
            wandb.init(project=f"vla-rl-forge-{cfg.task_name}", name=time.strftime('%m%d-%H:%M:%S'))

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

    def _get_rewards(self):
        """FORGE reward includes a contact penalty and success prediction error."""
        # Add success prediction rewards.
        check_rot = self.cfg_task.name == "nut_thread"
        true_successes = self._get_curr_successes(
            success_threshold=self.cfg_task.success_threshold, check_rot=check_rot
        )

        rew_buf = torch.where(true_successes, 1.0, 0.0)

        force = self.get_ft_force()[1]
        force_mag = torch.norm(force, dim=1)
        force_penalty = torch.where(force_mag > 0.2, -0.1, 0.0)

        rew_buf = force_penalty + rew_buf
        
        # Log Reward
        wandb.log({
            "success_rate": true_successes.sum().item() / self.num_envs,
        })

        return rew_buf
    
    def _pre_physics_step(self, action):
        """Apply policy actions with smoothing."""
        # Save last actions for gr00t data logging
        self.last_actions = action.clone()
        super()._pre_physics_step(action) # This changes action to smoothed action

    def _get_observations(self):
        """Add additional FORGE observations."""
        obs_dict, state_dict = self._get_factory_obs_state_dict()

        noisy_fixed_pos = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
        prev_actions = self.actions.clone()
        prev_actions[:, 3:5] = 0.0

        obs_dict.update({
            "fingertip_pos": self.noisy_fingertip_pos,
            "fingertip_pos_rel_fixed": self.noisy_fingertip_pos - noisy_fixed_pos,
            "fingertip_quat": self.noisy_fingertip_quat,
            "force_threshold": self.contact_penalty_thresholds[:, None],
            "ft_force": self.noisy_force,
            "prev_actions": prev_actions,
        })

        state_dict.update({
            "ema_factor": self.ema_factor,
            "ft_force": self.force_sensor_smooth[:, 0:3],
            "force_threshold": self.contact_penalty_thresholds[:, None],
            "prev_actions": prev_actions,
        })

        obs_tensors = factory_utils.collapse_obs_dict(obs_dict, self.cfg.obs_order + ["prev_actions"])
        state_tensors = factory_utils.collapse_obs_dict(state_dict, self.cfg.state_order + ["prev_actions"])

        return {"policy": obs_tensors, "critic": state_tensors}


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