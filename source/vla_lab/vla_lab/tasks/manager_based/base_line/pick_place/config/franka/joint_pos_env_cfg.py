# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

import vla_lab.tasks.manager_based.base_line.pick_place.mdp as mdp
from vla_lab.tasks.manager_based.base_line.lift.config.franka.joint_pos_env_cfg import FrankaCubeLiftEnvCfg

# Green cube geometry (green_block.usd at scale=1.0)
_GREEN_HALF_HEIGHT = 0.0203
_WHITE_HALF_HEIGHT = 0.055
# White cube center z when stacked on green cube: 0.0203 + 0.0203 + 0.055 = 0.0956
_PLACE_TARGET_Z = _GREEN_HALF_HEIGHT + _GREEN_HALF_HEIGHT + _WHITE_HALF_HEIGHT  # 0.0956


@configclass
class FrankaPickPlaceEnvCfg(FrankaCubeLiftEnvCfg):
    def __post_init__(self):
        # Inherits full Franka Lift setup: robot, white cube, ee_frame, actions, command body_name
        super().__post_init__()

        # Longer episode: pick-and-place requires more steps than pure lifting
        self.episode_length_s = 8.0

        # ── Scene: add kinematic green target cube ────────────────────────────
        # Placed at positive y so white cube spawn range (y ≤ 0.05) never overlaps
        self.scene.green_cube = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/GreenCube",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.5, 0.15, _GREEN_HALF_HEIGHT], rot=[1, 0, 0, 0]
            ),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/green_block.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=True,
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_depenetration_velocity=5.0,
                ),
                semantic_tags=[("class", "green_cube")],
            ),
        )

        # ── Command: fix target to the stacking position ──────────────────────
        # Robot frame ≈ world frame (robot at origin). Target = white cube center
        # when resting on green cube.
        self.commands.object_pose.ranges.pos_x = (0.5, 0.5)
        self.commands.object_pose.ranges.pos_y = (0.15, 0.15)
        self.commands.object_pose.ranges.pos_z = (_PLACE_TARGET_Z, _PLACE_TARGET_Z)
        self.commands.object_pose.ranges.roll = (0.0, 0.0)
        self.commands.object_pose.ranges.pitch = (0.0, 0.0)
        self.commands.object_pose.ranges.yaw = (0.0, 0.0)

        # ── Events: prevent white cube from spawning on top of green cube ─────
        # White cube init_state is [0.5, 0, 0.055]. Offset y kept ≤ 0.05 so
        # world y ≤ 0.05, giving ≥ 0.10 m gap from green cube at y = 0.15.
        self.events.reset_object_position = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.05), "z": (0.0, 0.0)},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("object", body_names="Object"),
            },
        )

        # ── Rewards: tighten goal-tracking gate to prevent table-sliding ──────
        # White cube rests at z = 0.055. Using minimal_height = 0.07 forces a
        # real lift before the goal-tracking reward activates, while still
        # allowing reward at the target (z = 0.0956 > 0.07).
        self.rewards.object_goal_tracking.params["minimal_height"] = 0.07
        self.rewards.object_goal_tracking_fine_grained.params["minimal_height"] = 0.07

        # Continuous release guidance: gradient toward opening gripper near target.
        # std=0.08 limits effective range to ~15 cm so carry-phase grasping is unaffected.
        # At hover 5 cm above target (dist≈0.054): proximity≈0.41 → ~4/step signal.
        # At carry 20 cm from target: proximity≈0.02 → negligible (no interference).
        self.rewards.gripper_open_at_goal = RewTerm(
            func=mdp.gripper_open_at_goal,
            params={
                "std": 0.08,
                "minimal_height": 0.07,
                "green_half_height": _GREEN_HALF_HEIGHT,
                "white_half_height": _WHITE_HALF_HEIGHT,
            },
            weight=10.0,
        )

        # Binary place bonus: cube at stacking position AND gripper open
        self.rewards.place_bonus = RewTerm(
            func=mdp.object_placed_at_goal,
            params={
                "threshold": 0.05,
                "upright_error_threshold": 0.35,
                "open_threshold": 0.035,
                "green_half_height": _GREEN_HALF_HEIGHT,
                "white_half_height": _WHITE_HALF_HEIGHT,
            },
            weight=20.0,
        )

        # ── Observations: add green cube position ────────────────────────────
        # Gives the policy a direct signal for where to place the cube,
        # complementing the command (target_object_position) already in the obs.
        self.observations.policy.green_cube_position = ObsTerm(
            func=mdp.green_cube_position_in_robot_root_frame
        )

        # ── Terminations: success when cube placed stably, gripper open ───────
        self.terminations.task_success = DoneTerm(
            func=mdp.object_placed_success,
            params={
                "threshold": 0.05,
                "vel_threshold": 0.05,
                "upright_error_threshold": 0.35,
                "open_threshold": 0.035,
                "green_half_height": _GREEN_HALF_HEIGHT,
                "white_half_height": _WHITE_HALF_HEIGHT,
            },
        )


@configclass
class FrankaPickPlaceEnvCfg_PLAY(FrankaPickPlaceEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
