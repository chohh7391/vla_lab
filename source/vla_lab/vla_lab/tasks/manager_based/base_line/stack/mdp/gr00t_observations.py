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
    if success:
        return torch.ones((env.num_envs, 1))
    else:
        return torch.zeros((env.num_envs, 1))


def next_done(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    
    is_last_step = env.episode_length_buf.unsqueeze(1) == env.max_episode_length_s - 1
    
    if is_last_step:
        return torch.ones((env.num_envs, 1), dtype=torch.bool)
    else:
        return torch.zeros((env.num_envs, 1), dtype=torch.bool)
