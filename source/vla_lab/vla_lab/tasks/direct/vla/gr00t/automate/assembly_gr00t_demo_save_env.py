# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sensors import TiledCamera

from vla_lab.tasks.direct.base_line.automate import automate_algo_utils as automate_algo
from vla_lab.tasks.direct.base_line.automate.assembly_env import AssemblyEnv

from vla_lab.tasks.direct.vla.gr00t.automate.assembly_gr00t_env_cfg import AssemblyEnvCfg

import wandb
import time

import os
import pandas as pd
import sys
from torchvision.io import write_video

class AssemblyGr00tDemoSaveEnv(AssemblyEnv):
    cfg: AssemblyEnvCfg

    def __init__(self, cfg: AssemblyEnvCfg, render_mode: str | None = None, **kwargs):

        super().__init__(cfg, render_mode, **kwargs)

        if wandb.run is None:
            wandb.init(project=f"vla-rl-automate-{cfg.task_name}", name=time.strftime('%m%d-%H:%M:%S'))

        self.demo_save_cfg = self.cfg.demo_save_cfg

        if self.demo_save_cfg:
            self.gr00t_data_buffers = [{} for _ in range(self.num_envs)]
            self.gr00t_img_buffers = [{} for _ in range(self.num_envs)]

            self._initialize_parquet_buffers()
            self._initialize_image_buffers()


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


    def _get_observations(self):
        """Get actor/critic inputs using asymmetric critic."""

        # TODO: Add F/T Sensor obs
        obs_dict = {
            "joint_pos": self.joint_pos[:, 0:7],
            "fingertip_pos": self.fingertip_midpoint_pos,
            "fingertip_quat": self.fingertip_midpoint_quat,
            "fingertip_goal_pos": self.gripper_goal_pos,
            "fingertip_goal_quat": self.gripper_goal_quat,
            "delta_pos": self.gripper_goal_pos - self.fingertip_midpoint_pos,
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
        }
        # obs_tensors = [obs_dict[obs_name] for obs_name in self.cfg.obs_order + ['prev_actions']]
        obs_tensors = [obs_dict[obs_name] for obs_name in self.cfg.obs_order]
        obs_tensors = torch.cat(obs_tensors, dim=-1)

        # state_tensors = [state_dict[state_name] for state_name in self.cfg.state_order + ['prev_actions']]
        state_tensors = [state_dict[state_name] for state_name in self.cfg.state_order]
        state_tensors = torch.cat(state_tensors, dim=-1)

        if self.cfg.is_demo_save:

            self._update_gr00t_observations(
                eef_position=obs_dict["fingertip_pos"],
                eef_quaternion=obs_dict["fingertip_quat"],
                gripper_qpos=self.joint_pos[:, 7:9]
            )

        return {"policy": obs_tensors, "critic": state_tensors}
    
    def _reset_idx(self, env_ids):
        """Perform additional randomizations."""
        super()._reset_idx(env_ids)

        if self.cfg.is_demo_save:
            self._save_gr00t_observations(
                data_dir=self.demo_save_cfg.data_dir,
                video_dir=self.demo_save_cfg.video_dir,
            )
            self._initialize_parquet_buffers()


    def _pre_physics_step(self, action):
        """Apply policy actions with smoothing."""
        # Save last actions for gr00t data logging
        self.last_actions = action.clone()
        super()._pre_physics_step(action) # This changes action to smoothed action

    def _get_rewards(self):
        """Update rewards and compute success statistics."""
        # Get successful and failed envs at current timestep

        curr_successes = automate_algo.check_plug_inserted_in_socket(
            self.held_pos,
            self.fixed_pos,
            self.disassembly_dists,
            self.keypoints_held,
            self.keypoints_fixed,
            self.cfg_task.close_error_thresh,
            self.episode_length_buf,
        )
        rew_buf = torch.where(curr_successes, 1.0, 0.0)

        self.prev_actions = self.actions.clone()
        return rew_buf

    ''' GR00T Helper Functions '''
    def _initialize_parquet_buffers(self):
        """Clears and initializes the data buffer for a specific environment."""
        for env_id in range(self.num_envs):
            self.gr00t_data_buffers[env_id] = {
                "observation.state": [],
                "action": [],
                "timestamp": [],
                "annotation.human.action.task_description": [],
                "task_index": [],
                "annotation.human.validity": [],
                "episode_index": [],
                "index": [],
                "next.reward": [],
                "next.done": [],
            }

    def _initialize_image_buffers(self):
        """Clears and initializes the image buffer for a specific environment."""
        for env_id in range(self.num_envs):
            self.gr00t_img_buffers[env_id] = {
                "left_camera": [],
                "right_camera": [],
                "wrist_camera": [],
            }

    def _save_gr00t_observations(self, data_dir: str, video_dir: str):
        """Saves the collected gr00t data and videos to the specified directories."""
        # Prevent saving if no data collected: Initial reset
        if len(self.gr00t_data_buffers[0]["observation.state"]) != self.max_episode_length - 2:
            print("No data collected for this episode. Skipping save.")
            return

        for env_id in range(self.num_envs):
            # Save Datas
            df = pd.DataFrame(self.gr00t_data_buffers[env_id])

            gr00t_data_path = os.path.join(
                data_dir, f"episode_{env_id:06d}.parquet"
            )
            os.makedirs(os.path.dirname(gr00t_data_path), exist_ok=True)

            df.to_parquet(gr00t_data_path, index=False)

            # Save Videos
            for camera_name, img_list in self.gr00t_img_buffers[env_id].items():

                img_tensor = torch.stack(img_list, dim=0).to(device=self.device, dtype=torch.uint8) # (T, C, H, W)

                gr00t_video_path = os.path.join(
                    video_dir[camera_name], f"episode_{env_id:06d}.mp4"
                )
                os.makedirs(os.path.dirname(gr00t_video_path), exist_ok=True)

                write_video(gr00t_video_path, img_tensor, fps=int(1/self.step_dt), video_codec="h264")


        print(f"Saved {self.num_envs} episodes to {data_dir} and {video_dir}")
        print(f"number of frames: {len(self.gr00t_data_buffers[0]['observation.state'])}")
        print(f"Exiting program after saving gr00t data...")

        # Quit Program
        sys.exit(0)

    def _update_gr00t_observations(self, eef_position, eef_quaternion, gripper_qpos):

        if self.episode_length_buf[0].item() == 0:
            return # Skip the first step (reset step)

        curr_successes = automate_algo.check_plug_inserted_in_socket(
            self.held_pos,
            self.fixed_pos,
            self.disassembly_dists,
            self.keypoints_held,
            self.keypoints_fixed,
            self.cfg_task.close_error_thresh,
            self.episode_length_buf,
        )

        state = torch.cat((eef_position, eef_quaternion, gripper_qpos), dim=-1)
        arm_action = self.last_actions[:, :6]
        gripper_action = torch.ones((self.num_envs, 1), device=self.device) # This Task forces gripper to be closed
        action = torch.cat((arm_action, gripper_action), dim=-1) # rescale success_pred to [0, 1]

        # Get Camera Datas -> # (num_envs, height, width, channel)
        left_camera_data = self._left_camera.data.output["rgb"].clone()
        right_camera_data = self._right_camera.data.output["rgb"].clone()
        wrist_camera_data = self._wrist_camera.data.output["rgb"].clone()

        print("================== Parquet Data Stats =================")
        print(f"current / max: {self.episode_length_buf[0].item()} / {self.max_episode_length - 2}")
        print(f"success / total: {curr_successes.sum().item()} / {self.num_envs}")

        for i in range(self.num_envs):
            current_step_in_ep = self.episode_length_buf[i].item()

            # update data buffers
            self.gr00t_data_buffers[i]["observation.state"].append(state[i].cpu().numpy().tolist())
            self.gr00t_data_buffers[i]["action"].append(action[i].cpu().numpy().tolist()) # action[6] is success_pred
            self.gr00t_data_buffers[i]["timestamp"].append(round(current_step_in_ep * self.step_dt, 3))
            self.gr00t_data_buffers[i]["annotation.human.action.task_description"].append(0) # Task index 0
            self.gr00t_data_buffers[i]["task_index"].append(0)
            self.gr00t_data_buffers[i]["annotation.human.validity"].append(curr_successes[i].item())
            self.gr00t_data_buffers[i]["episode_index"].append(i)
            self.gr00t_data_buffers[i]["index"].append(i * (self.max_episode_length - 1) + current_step_in_ep - 1)
            
            if curr_successes[i]:
                self.gr00t_data_buffers[i]["next.reward"].append(1.0)
            else:
                self.gr00t_data_buffers[i]["next.reward"].append(0.0)
            
            if current_step_in_ep == self.max_episode_length - 2: # Final Step of episode
                self.gr00t_data_buffers[i]["next.done"].append(True)
            else:
                self.gr00t_data_buffers[i]["next.done"].append(False)

            # update image buffers
            self.gr00t_img_buffers[i]["left_camera"].append(
                left_camera_data[i]
            )
            self.gr00t_img_buffers[i]["right_camera"].append(
                right_camera_data[i]
            )
            self.gr00t_img_buffers[i]["wrist_camera"].append(
                wrist_camera_data[i]
            )

