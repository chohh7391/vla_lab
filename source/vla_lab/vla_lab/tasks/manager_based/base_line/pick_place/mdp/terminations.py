# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _object_is_upright(object: RigidObject, upright_error_threshold: float) -> torch.Tensor:
    quat = object.data.root_quat_w
    local_z_dot_world_z = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    tilt = torch.acos(torch.clamp(local_z_dot_world_z, -1.0, 1.0))
    return tilt < upright_error_threshold


def object_placed_success(
    env: ManagerBasedRLEnv,
    threshold: float = 0.05,
    vel_threshold: float = 0.05,
    upright_error_threshold: float = 0.35,
    open_threshold: float = 0.035,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    green_cube_cfg: SceneEntityCfg = SceneEntityCfg("green_cube"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    green_half_height: float = 0.0203,
    white_half_height: float = 0.021,
) -> torch.Tensor:
    """Terminate successfully when the white cube is stably placed on the green cube with the gripper open."""
    white: RigidObject = env.scene[object_cfg.name]
    green: RigidObject = env.scene[green_cube_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    target_pos = green.data.root_pos_w.clone()
    target_pos[:, 2] += green_half_height + white_half_height

    dist = torch.norm(white.data.root_pos_w[:, :3] - target_pos, dim=1)
    near_target = dist < threshold

    stable = torch.norm(white.data.root_lin_vel_w, dim=1) < vel_threshold
    upright = _object_is_upright(white, upright_error_threshold)

    finger_ids, _ = robot.find_joints("panda_finger_joint.*")
    finger_pos = robot.data.joint_pos[:, finger_ids]
    gripper_open = torch.mean(finger_pos, dim=1) > open_threshold

    return near_target & stable & upright & gripper_open
