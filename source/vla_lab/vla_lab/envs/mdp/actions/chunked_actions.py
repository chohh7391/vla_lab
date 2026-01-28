# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any, Dict
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from pxr import UsdPhysics

import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.operational_space import OperationalSpaceController
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.sensors import ContactSensor, ContactSensorCfg, FrameTransformer, FrameTransformerCfg
from isaaclab.sim.utils import find_matching_prims

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.envs.utils.io_descriptors import GenericActionIODescriptor

    from . import actions_cfg

from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from ...utils import AsyncGr00tInferenceClient
from ..observations import get_gr00t_observations

# import logger
logger = logging.getLogger(__name__)


class DifferentialInverseKinematicsChunkedAction(DifferentialInverseKinematicsAction):

    def __init__(self, cfg: actions_cfg.DifferentialInverseKinematicsChunkedActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)

        self._chunk_size = self.cfg.chunk_size
        self._gr00t_policy = AsyncGr00tInferenceClient(host="localhost", port=self.cfg.vla_server_port)
        self._gr00t_observations: Dict[str, Any] | None = None
        self._gr00t_actions: Dict[str, Any] | None = None
        self._processed_gr00t_actions: torch.Tensor | None = None
        self._episode_length: int = 0


    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self._ik_controller.action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        
        # process gr00t actions
        if self._episode_length % self._chunk_size == 0:
            if self._episode_length != 0:
                try:
                    self._gr00t_actions = self._gr00t_policy.get_result()
                except Exception as e:
                    print(f"Error getting Gr00t action: {e}")
                    self._gr00t_actions = torch.zeros((self.num_envs, self._chunk_size, self.action_dim), device=self.device)
            
            gr00t_actions_pos = torch.tensor(self.gr00t_actions["action.eef_position_delta"], dtype=torch.float32, device=self.device)
            gr00t_actions_rot = torch.tensor(self.gr00t_actions["action.eef_rotation_delta"], dtype=torch.float32, device=self.device)
            gr00t_actions_delta_pose = torch.cat(
                (gr00t_actions_pos, gr00t_actions_rot), dim=-1
            )
            self._processed_gr00t_actions = gr00t_actions_delta_pose

        elif self._episode_length % self._chunk_size == int(self._chunk_size / 2):
            self._gr00t_policy.request_action(self.get_gr00t_observations(env=self._env))

        self._processed_actions[:] = self._scale * (
            self.raw_actions + self._processed_gr00t_actions[:, self._episode_length % self._chunk_size, :]
        )
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )
        # obtain quantities from simulation
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()

        # set command into controller
        self._ik_controller.set_command(self._processed_actions, ee_pos_curr, ee_quat_curr)
        self._episode_length += 1

    def apply_actions(self):
        # obtain quantities from simulation
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        # compute the delta in joint-space
        if ee_quat_curr.norm() != 0:
            jacobian = self._compute_frame_jacobian()
            joint_pos_des = self._ik_controller.compute(ee_pos_curr, ee_quat_curr, jacobian, joint_pos)
        else:
            joint_pos_des = joint_pos.clone()
        # set the joint position command
        self._asset.set_joint_position_target(joint_pos_des, self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0
        self._episode_length = 0
        self._gr00t_observations = get_gr00t_observations(env=self._env)
        self._gr00t_actions = self._gr00t_policy.get_action_sync(self._gr00t_observations)

        
