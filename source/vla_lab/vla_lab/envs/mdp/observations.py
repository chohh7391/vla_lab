from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCollection
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_tasks.manager_based.manipulation.stack import mdp

import numpy as np


def state(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    
    ee_frame_pos = mdp.ee_frame_pos(env, ee_frame_cfg)
    ee_frame_quat = mdp.ee_frame_quat(env, ee_frame_cfg)
    gripper_pos = mdp.gripper_pos(env, robot_cfg)

    return torch.cat((ee_frame_pos, ee_frame_quat, gripper_pos), dim=-1)


def task_description(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    return torch.zeros((env.num_envs, 1))


def task_index(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    return torch.zeros((env.num_envs, 1))


def episode_index(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    return torch.zeros((env.num_envs, 1)) # TODO: need to edit


def index(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    return torch.zeros((env.num_envs, 1)) # TODO: need to edit


def next_reward(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    
    success = mdp.cubes_stacked(env)
    next_reward = torch.where(
        success, torch.ones((env.num_envs, 1)), torch.zeros((env.num_envs, 1))
    )

    return next_reward


def next_done(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    
    is_last_step = env.episode_length_buf.unsqueeze(1) == env.max_episode_length_s - 1
    next_done = torch.where(
        is_last_step, torch.ones((env.num_envs, 1), dtype=torch.bool), torch.zeros((env.num_envs, 1), dtype=torch.bool)
    )
    
    return next_done


def get_gr00t_observations(
    env: ManagerBasedRLEnv,
    left_camera_cfg: SceneEntityCfg = SceneEntityCfg("left_camera"),
    right_camera_cfg: SceneEntityCfg = SceneEntityCfg("right_camera"),
    wrist_camera_cfg: SceneEntityCfg = SceneEntityCfg("wrist_camera"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):

    left_camera_img = mdp.image(env=env, sensor_cfg=left_camera_cfg, data_type="rgb", normalize=False)
    right_camera_img = mdp.image(env=env, sensor_cfg=right_camera_cfg, data_type="rgb", normalize=False)
    wrist_camera_img = mdp.image(env=env, sensor_cfg=wrist_camera_cfg, data_type="rgb", normalize=False)

    state_ = state(env=env, ee_frame_cfg=ee_frame_cfg, robot_cfg=robot_cfg)

    observations = {
        "video.left_view": np.expand_dims(left_camera_img.cpu().numpy().astype(np.uint8), axis=1),
        "video.right_view": np.expand_dims(right_camera_img.cpu().numpy().astype(np.uint8), axis=1),
        "video.wrist_view": np.expand_dims(wrist_camera_img.cpu().numpy().astype(np.uint8), axis=1),
        "state.eef_position": np.expand_dims(state_[:, 0:3].cpu().numpy().astype(np.float64), axis=1),
        "state.eef_quaternion": np.expand_dims(state_[:, 3:7].cpu().numpy().astype(np.float64), axis=1),
        "state.gripper_qpos": np.expand_dims(state_[:, 7:9].cpu().numpy().astype(np.float64), axis=1),
    }

    return observations
