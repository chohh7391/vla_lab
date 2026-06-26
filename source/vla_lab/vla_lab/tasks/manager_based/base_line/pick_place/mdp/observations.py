# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def green_cube_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    green_cube_cfg: SceneEntityCfg = SceneEntityCfg("green_cube"),
) -> torch.Tensor:
    """Position of the green (target) cube in the robot's root frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    green: RigidObject = env.scene[green_cube_cfg.name]
    green_pos_w = green.data.root_pos_w[:, :3]
    green_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, green_pos_w)
    return green_pos_b
