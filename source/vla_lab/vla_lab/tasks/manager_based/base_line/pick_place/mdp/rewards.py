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


def object_upright_error(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Return tilt angle between the cube's local z-axis and the world z-axis."""
    white: RigidObject = env.scene[object_cfg.name]
    quat = white.data.root_quat_w
    local_z_dot_world_z = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    return torch.acos(torch.clamp(local_z_dot_world_z, -1.0, 1.0))


def object_uprightness(
    env: ManagerBasedRLEnv,
    std: float = 0.35,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the cube for staying close to its initial upright orientation."""
    return 1 - torch.tanh(object_upright_error(env, object_cfg=object_cfg) / std)


def object_placed_at_goal(
    env: ManagerBasedRLEnv,
    threshold: float = 0.05,
    upright_error_threshold: float = 0.35,
    open_threshold: float = 0.035,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    green_cube_cfg: SceneEntityCfg = SceneEntityCfg("green_cube"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    green_half_height: float = 0.0203,
    white_half_height: float = 0.055,
) -> torch.Tensor:
    """Binary reward for placing the white cube on the green cube with the gripper open."""
    white: RigidObject = env.scene[object_cfg.name]
    green: RigidObject = env.scene[green_cube_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    target_pos = green.data.root_pos_w.clone()
    target_pos[:, 2] += green_half_height + white_half_height

    dist = torch.norm(white.data.root_pos_w[:, :3] - target_pos, dim=1)
    near_target = (dist < threshold).float()
    upright = (object_upright_error(env, object_cfg=object_cfg) < upright_error_threshold).float()

    finger_ids, _ = robot.find_joints("panda_finger_joint.*")
    finger_pos = robot.data.joint_pos[:, finger_ids]
    gripper_open = (torch.mean(finger_pos, dim=1) > open_threshold).float()

    return near_target * upright * gripper_open


def gripper_open_at_goal(
    env: ManagerBasedRLEnv,
    std: float = 0.08,
    minimal_height: float = 0.07,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    green_cube_cfg: SceneEntityCfg = SceneEntityCfg("green_cube"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    green_half_height: float = 0.0203,
    white_half_height: float = 0.055,
) -> torch.Tensor:
    """Encourage gripper to open when the held cube is near the stacking target.

    Uses a tanh proximity kernel so gradient flows continuously rather than switching
    on/off at a hard threshold. std=0.08 limits the effective range to ~15 cm, so the
    reward is negligible during the carry phase (> 20 cm from target) and does not
    interfere with the grasping/lifting reward signal.

    Reward = lifted * (1 - tanh(dist / std)) * gripper_openness, in [0, 1].
    """
    white: RigidObject = env.scene[object_cfg.name]
    green: RigidObject = env.scene[green_cube_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    target_pos = green.data.root_pos_w.clone()
    target_pos[:, 2] += green_half_height + white_half_height

    dist = torch.norm(white.data.root_pos_w[:, :3] - target_pos, dim=1)
    proximity = 1 - torch.tanh(dist / std)
    lifted = (white.data.root_pos_w[:, 2] > minimal_height).float()
    upright = object_uprightness(env, object_cfg=object_cfg)

    finger_ids, _ = robot.find_joints("panda_finger_joint.*")
    finger_pos = robot.data.joint_pos[:, finger_ids]
    gripper_openness = torch.clamp(torch.mean(finger_pos, dim=1) / 0.04, 0.0, 1.0)

    return lifted * proximity * upright * gripper_openness
